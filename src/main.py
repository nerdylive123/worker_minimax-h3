"""Supervisor entrypoint for the MiniMax-H3 RunPod worker.

1. Optionally switch to a baked-in offline model (/local_model_args.json).
2. Launch ``sglang serve`` as a subprocess.
3. Poll its /health endpoint until ready (or a classified startup failure).
4. Start the RunPod serverless loop with the async proxy handler.

On startup failure the worker stays alive and answers every job with the
classified cause instead of crash-looping the container.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from typing import Optional

import aiohttp

import runpod

from args_builder import build_serve_argv, get_config
from startup_errors import classify

LOCAL_MODEL_ARGS_PATH = "/local_model_args.json"
HEALTH_POLL_INTERVAL = 2.0


def _apply_baked_model() -> None:
    """If the model was baked into the image, point SGLang at it offline."""
    if not os.path.exists(LOCAL_MODEL_ARGS_PATH):
        return
    with open(LOCAL_MODEL_ARGS_PATH, "r", encoding="utf-8") as f:
        args = json.load(f)
    if args.get("model_path"):
        os.environ["SGLANG_MODEL_PATH"] = args["model_path"]
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        print(f"[main] Using baked model at {args['model_path']}", flush=True)


def _launch_sglang() -> subprocess.Popen:
    argv = build_serve_argv()
    print(f"[main] Launching: {' '.join(argv)}", flush=True)
    return subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _drain_log(proc: subprocess.Popen, max_chars: int = 20000) -> str:
    """Read whatever sglang has emitted so far (non-blocking best effort)."""
    if proc.stdout is None:
        return ""
    try:
        data = proc.stdout.read() or ""
    except Exception:  # noqa: BLE001
        data = ""
    return data[-max_chars:]


async def _wait_for_health(port: str, timeout: float, proc: subprocess.Popen) -> Optional[str]:
    """Poll /health until ready. Returns None on success, else an error log string."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                # Process exited before becoming healthy.
                return _drain_log(proc) or f"sglang exited with code {proc.returncode}"
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        return None
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(HEALTH_POLL_INTERVAL)
    return _drain_log(proc) or "sglang did not become healthy before the startup timeout"


def _startup_error_handler_factory(error: dict):
    """Build a handler that always returns the classified startup error."""

    def _handler(job):  # noqa: ARG001 - every job gets the same cause
        return {
            "error": "SGLang failed to start; see 'startup' for details.",
            "startup": error,
        }

    return _handler


async def _run() -> None:
    _apply_baked_model()
    cfg = get_config()

    try:
        proc = _launch_sglang()
    except (ValueError, OSError) as exc:
        error = classify(str(exc)).to_dict()
        runpod.serverless.start({"handler": _startup_error_handler_factory(error)})
        return

    startup_timeout = float(os.getenv("SGLANG_STARTUP_TIMEOUT", "1800"))
    err_log = await _wait_for_health(cfg["SGLANG_PORT"], startup_timeout, proc)

    if err_log is not None:
        error = classify(err_log).to_dict()
        print(f"[main] Startup failure: {error}", flush=True)
        runpod.serverless.start({"handler": _startup_error_handler_factory(error)})
        return

    print("[main] SGLang healthy; starting RunPod serverless loop.", flush=True)

    from handler import handler

    max_concurrency = int(os.getenv("MAX_CONCURRENCY", "4"))
    runpod.serverless.start(
        {
            "handler": handler,
            "return_aggregate_stream": True,
            "concurrency_modifier": lambda _: max_concurrency,
        }
    )


if __name__ == "__main__":
    asyncio.run(_run())
