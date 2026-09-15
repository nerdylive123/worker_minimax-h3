"""Supervisor entrypoint for the MiniMax-H3 RunPod worker (ComfyUI backend).

Boot sequence:
  1. Resolve the cached Comfy-Org/MiniMax-H3 snapshot (cache-first; falls back
     to a Hub download into the same cache dir when absent).
  2. Map the chosen variant/precision single-file components into ComfyUI's
     models/ tree (diffusion_models, text_encoders, vae) via symlinks.
  3. Launch ComfyUI headless and poll its health endpoint.
  4. Start the RunPod serverless loop with the async proxy handler.

On startup failure the worker stays alive and answers every job with the cause
instead of crash-looping the container.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from typing import Dict, Optional

import aiohttp

import runpod

import model_cache
import model_files

# --- Configuration (env-overridable) -----------------------------------------

MODEL_ID = os.getenv("MODEL_NAME", "Comfy-Org/MiniMax-H3")
MODEL_VARIANT = os.getenv("MODEL_VARIANT", "fl2va")  # fl2va | ref2va
PRECISION = os.getenv("PRECISION", "bf16")  # bf16 | fp8_scaled | int8_convrot | pruned_*
COMFYUI_DIR = os.getenv("COMFYUI_DIR", "/ComfyUI")
COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8188"))
COMFYUI_STARTUP_TIMEOUT = float(os.getenv("COMFYUI_STARTUP_TIMEOUT", "900"))
HEALTH_POLL_INTERVAL = 2.0


def _pick_files(snapshot: str) -> Dict[str, str]:
    """Resolve the on-disk component files for the configured variant/precision."""
    names = model_files.resolve_filenames(MODEL_VARIANT, PRECISION)
    files = {
        "diffusion_models": os.path.join(snapshot, "diffusion_models", names["unet_name"]),
        "text_encoders": os.path.join(snapshot, "text_encoders", names["clip_name"]),
        "vae": os.path.join(snapshot, "vae", names["vae_name"]),
        "audio_vae": os.path.join(snapshot, "vae", names["audio_vae_name"]),
        "loras": os.path.join(snapshot, "loras", names["turbo_lora"]),
    }
    # LoRA is optional (turbo mode); only hard-require the core components.
    required = {k: v for k, v in files.items() if k != "loras"}
    missing = [p for p in required.values() if not os.path.isfile(p)]
    if missing:
        raise RuntimeError(
            "Expected model component(s) missing from cache: " + ", ".join(missing)
        )
    return files


def _link_into_comfyui(files: Dict[str, str]) -> None:
    """Symlink the chosen component files into ComfyUI's models tree."""
    targets = {
        "diffusion_models": os.path.join(COMFYUI_DIR, "models", "diffusion_models"),
        "text_encoders": os.path.join(COMFYUI_DIR, "models", "text_encoders"),
        "vae": os.path.join(COMFYUI_DIR, "models", "vae"),
        "audio_vae": os.path.join(COMFYUI_DIR, "models", "vae"),
        "loras": os.path.join(COMFYUI_DIR, "models", "loras"),
    }
    for key, src in files.items():
        if not os.path.isfile(src):
            continue  # optional components (e.g. turbo LoRA) may be absent
        dest_dir = targets[key]
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(src))
        if os.path.islink(dest) or os.path.exists(dest):
            os.remove(dest)
        os.symlink(src, dest)
        print(f"[main] linked {src} -> {dest}", flush=True)


def _needed_patterns() -> list:
    """Allowlist of component files for the configured variant/precision.

    The Comfy-Org repo ships every quant of every component (~32 files,
    hundreds of GB). Downloading all of them blows the container disk (the
    observed failure). Fetch only what this variant/precision actually loads.
    """
    names = model_files.resolve_filenames(MODEL_VARIANT, PRECISION)
    patterns = [
        f"diffusion_models/{names['unet_name']}",
        f"text_encoders/{names['clip_name']}",
        f"vae/{names['vae_name']}",
        f"vae/{names['audio_vae_name']}",
    ]
    if os.getenv("ENABLE_TURBO", "1") not in ("0", "false", "False"):
        patterns.append(f"loras/{names['turbo_lora']}")
    return patterns


def _resolve_model() -> str:
    """Return the local snapshot path, downloading only the needed components."""
    if model_cache.snapshot_available(MODEL_ID):
        model_cache.set_offline()
        snapshot = model_cache.resolve_snapshot_path(MODEL_ID)
        print(f"[main] using cached model at {snapshot}", flush=True)
        return snapshot

    # Cache miss: download only the needed components into the HF cache
    # (HF_HOME points at the same root). A full snapshot_download would pull
    # every quant (~hundreds of GB) and fill the disk.
    patterns = _needed_patterns()
    print(f"[main] cache miss; downloading {len(patterns)} components from {MODEL_ID} ...", flush=True)
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=MODEL_ID,
        token=os.environ.get("HF_TOKEN"),
        allow_patterns=patterns,
    )
    model_cache.set_offline()
    print(f"[main] downloaded model to {snapshot}", flush=True)
    return snapshot


def _drain_log(proc: subprocess.Popen, max_chars: int = 20000) -> str:
    """Return recent child output without blocking on a live process.

    The launch starts a background thread that continuously drains the child's
    stdout into a bounded buffer, so this never waits on EOF and the child
    never blocks on a full pipe while logging.
    """
    buffer = getattr(proc, "_log_buffer", None)
    if buffer is not None:
        with buffer["lock"]:
            return buffer["data"][-max_chars:]
    return ""


def _launch_comfyui() -> subprocess.Popen:
    argv = [
        "python3",
        os.path.join(COMFYUI_DIR, "main.py"),
        "--listen",
        "0.0.0.0",
        "--port",
        str(COMFYUI_PORT),
        "--disable-auto-launch",
    ]
    extra = os.getenv("COMFYUI_EXTRA_ARGS", "").strip()
    if extra:
        import shlex

        argv.extend(shlex.split(extra))
    print(f"[main] launching: {' '.join(argv)}", flush=True)
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # Drain stdout in the background: tee it to the pod log (so ComfyUI's own
    # output is visible for debugging) and keep a bounded tail for _drain_log.
    import threading

    buffer = {"data": "", "lock": threading.Lock()}

    def _pump() -> None:
        try:
            for line in iter(proc.stdout.readline, ""):  # type: ignore[union-attr]
                if not line:
                    break
                print(f"[comfyui] {line.rstrip()}", flush=True)
                with buffer["lock"]:
                    buffer["data"] = (buffer["data"] + line)[-20000:]
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_pump, daemon=True).start()
    proc._log_buffer = buffer  # type: ignore[attr-defined]
    proc._start_time = time.monotonic()  # type: ignore[attr-defined]
    return proc


async def _wait_for_health(proc: subprocess.Popen) -> Optional[str]:
    """Poll until ComfyUI's HTTP server answers, tolerating long model loads.

    Returns None on success. Only fails early if the process provably crashed
    (exited within a short grace window); a long-running process that simply
    hasn't opened its port yet is treated as still-loading, not failed.
    """
    url = f"http://127.0.0.1:{COMFYUI_PORT}/"
    deadline = time.monotonic() + COMFYUI_STARTUP_TIMEOUT
    grace = float(os.getenv("COMFYUI_CRASH_GRACE_SECONDS", "20"))
    last_log = 0.0
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            rc = proc.poll()
            if rc is not None:
                uptime = time.monotonic() - getattr(proc, "_start_time", 0.0)
                log = _drain_log(proc)
                # A quick exit is a real crash; report it with whatever it logged.
                if uptime < grace or log:
                    return (
                        f"ComfyUI exited with code {rc} after {uptime:.1f}s. "
                        f"Output: {log or '(no output captured)'}"
                    )
                return f"ComfyUI exited with code {rc} after {uptime:.1f}s"
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status in (200, 404):  # server is up
                        return None
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            now = time.monotonic()
            if now - last_log >= 15:
                elapsed = now - getattr(proc, "_start_time", now)
                print(f"[main] waiting for ComfyUI ({elapsed:.0f}s elapsed)...", flush=True)
                last_log = now
            await asyncio.sleep(HEALTH_POLL_INTERVAL)
    return (
        f"ComfyUI did not open port {COMFYUI_PORT} within {COMFYUI_STARTUP_TIMEOUT}s. "
        f"Recent output: {_drain_log(proc) or '(none)'}"
    )


def _startup_error_handler_factory(message: str):
    # Echo the cause to the pod log so "see 'startup' for details" is actually
    # diagnosable from the logs, not just the job result payload.
    print(f"[main] STARTUP FAILURE: {message}", flush=True)

    def _handler(job):  # noqa: ARG001
        return {"error": "Worker failed to start; see 'startup' for details.", "startup": message}

    return _handler


async def _run() -> None:
    try:
        snapshot = _resolve_model()
        files = _pick_files(snapshot)
        _link_into_comfyui(files)
        proc = _launch_comfyui()
    except (ValueError, RuntimeError, OSError) as exc:
        runpod.serverless.start({"handler": _startup_error_handler_factory(str(exc))})
        return

    err = await _wait_for_health(proc)
    if err is not None:
        runpod.serverless.start({"handler": _startup_error_handler_factory(err)})
        return

    print("[main] ComfyUI healthy; starting RunPod serverless loop.", flush=True)
    import handler as _handler_mod

    # Inject the exact model component filenames we linked into the workflow.
    _handler_mod._MODEL_FILENAMES.update(
        model_files.resolve_filenames(MODEL_VARIANT, PRECISION)
    )

    runpod.serverless.start({"handler": _handler_mod.handler, "return_aggregate_stream": True})


if __name__ == "__main__":
    asyncio.run(_run())
