"""RunPod serverless handler for MiniMax-H3 (ComfyUI backend).

The worker runs ComfyUI headless on localhost. This handler:
  * ``generate``  -- inject a prompt into the bundled official MiniMax-H3
    workflow (T2V or R2V), submit it to ``POST /prompt``, poll ``/history``,
    and return the generated video.
  * ``raw``       -- submit a caller-supplied API-format workflow verbatim.
  * ``proxy``     -- forward any ``{route, body, method}`` to ComfyUI's HTTP API.

The workflow JSONs are the official Comfy-Org templates
(workflows/t2v.json, workflows/r2v.json), converted GUI->API at runtime.
"""

from __future__ import annotations

import asyncio
import base64
import os
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

import aiohttp

import workflow_api

COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8188"))
COMFYUI_DIR = os.getenv("COMFYUI_DIR", "/ComfyUI")
BASE = f"http://127.0.0.1:{COMFYUI_PORT}"
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "3600"))
HISTORY_POLL_INTERVAL = 2.0
GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT", "1800"))

WORKFLOWS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workflows")
WORKFLOW_BY_ACTION = {
    "generate": os.path.join(WORKFLOWS_DIR, "t2v.json"),
    "t2v": os.path.join(WORKFLOWS_DIR, "t2v.json"),
    "r2v": os.path.join(WORKFLOWS_DIR, "r2v.json"),
    "ref2va": os.path.join(WORKFLOWS_DIR, "r2v.json"),
}

_gui_cache: Dict[str, Dict[str, Any]] = {}

# Model component filenames for the configured variant/precision, resolved at
# import time so the handler can inject them into each request's workflow.
_MODEL_FILENAMES: Dict[str, str] = {}


def _load_gui_workflow(path: str) -> Dict[str, Any]:
    """Return a deep copy of the cached GUI workflow template."""
    if path not in _gui_cache:
        _gui_cache[path] = workflow_api.load_workflow(path)
    import json as _json

    return _json.loads(_json.dumps(_gui_cache[path]))


def _build_prompt(path: str, params: Dict[str, Any], strip_media: bool = False) -> Dict[str, Any]:
    """Inject params into the GUI workflow, then convert GUI -> API.

    For R2V (top-level, no subgraph) the demo reference images are stripped so
    a bare text prompt runs; callers re-add reference media via a raw workflow.
    """
    gui = _load_gui_workflow(path)
    if strip_media:
        workflow_api.strip_unlinked_load_media(gui)
    workflow_api.set_params(gui, params)
    api = workflow_api.gui_to_api(gui)
    # Authoritative prompt set directly on the sampler node (robust across both
    # the subgraph T2V and top-level R2V layouts).
    if params.get("prompt"):
        workflow_api.set_prompt_text(api, params["prompt"])
    return api


async def _submit(session: aiohttp.ClientSession, prompt: Dict[str, Any]) -> str:
    payload = {"prompt": prompt, "client_id": str(uuid.uuid4())}
    async with session.post(f"{BASE}/prompt", json=payload) as resp:
        data = await resp.json(content_type=None)
    if "error" in data:
        raise RuntimeError(f"ComfyUI rejected the workflow: {data.get('error')} {data.get('node_errors')}")
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI returned no prompt_id: {data}")
    return prompt_id


async def _wait_outputs(session: aiohttp.ClientSession, prompt_id: str) -> Dict[str, Any]:
    deadline = asyncio.get_event_loop().time() + GENERATION_TIMEOUT
    while True:
        async with session.get(f"{BASE}/history/{prompt_id}") as resp:
            history = await resp.json(content_type=None)
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("completed") or entry.get("outputs"):
                return entry.get("outputs", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI generation failed: {entry}")
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError(f"Generation timed out after {GENERATION_TIMEOUT}s")
        await asyncio.sleep(HISTORY_POLL_INTERVAL)


async def _fetch_video(
    session: aiohttp.ClientSession, outputs: Dict[str, Any], as_base64: bool
) -> Dict[str, Any]:
    """Locate the saved video in node outputs and return it.

    A ComfyUI ``/view`` URL points at the worker's own loopback (127.0.0.1) and
    is unreachable by remote RunPod callers, so we always fetch the bytes and
    return base64 by default. With ``as_base64=False`` we still fetch the bytes
    (to confirm the artifact exists) but return the on-worker path plus a
    localhost URL, which is only useful for in-pod debugging.
    """
    videos = []
    for node_out in outputs.values():
        for key in ("videos", "gifs", "images"):
            for item in node_out.get(key, []) or []:
                videos.append(item)
    if not videos:
        raise RuntimeError(f"No video found in ComfyUI outputs: {list(outputs)}")

    item = videos[0]
    filename, subfolder, ftype = item.get("filename"), item.get("subfolder", ""), item.get("type", "output")
    params = {"filename": filename, "subfolder": subfolder, "type": ftype}
    async with session.get(f"{BASE}/view", params=params) as resp:
        blob = await resp.read()
    if as_base64:
        return {
            "video_base64": base64.b64encode(blob).decode("ascii"),
            "filename": filename,
        }
    return {
        "filename": filename,
        "worker_path": f"{COMFYUI_DIR}/output/{subfolder + '/' if subfolder else ''}{filename}",
        "video_url_loopback_only": f"{BASE}/view?filename={filename}&subfolder={subfolder}&type={ftype}",
        "size_bytes": len(blob),
        "note": "URL is loopback-only (unreachable remotely); use return_base64=true for delivery.",
    }


async def _run_generation(job_input: Dict[str, Any]) -> Dict[str, Any]:
    action = str(job_input.get("action", "generate")).lower()
    workflow_path = WORKFLOW_BY_ACTION.get(action, WORKFLOW_BY_ACTION["generate"])

    prompt_text = job_input.get("prompt")
    if not prompt_text:
        raise ValueError("'prompt' is required for generation")

    # Optional overrides injected into the workflow widgets. Model filenames
    # are fixed at container start (env variant/precision); callers may
    # override dimensions/duration/seed/turbo/ratio per request.
    params: Dict[str, Any] = {"prompt": prompt_text}
    for key in ("width", "height", "duration", "seed", "turbo", "turbo_steps", "turbo_strength", "ratio"):
        if job_input.get(key) is not None:
            params[key] = job_input[key]
    params.update(_MODEL_FILENAMES)

    # R2V's bundled workflow ships demo reference images that aren't in the
    # image; strip them for bare text prompts (media re-added via raw workflow).
    has_ref_media = bool(
        job_input.get("ref_images") or job_input.get("ref_videos") or job_input.get("ref_audios")
    )
    is_r2v = action in ("r2v", "ref2va")
    prompt = _build_prompt(workflow_path, params, strip_media=(is_r2v and not has_ref_media))

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        prompt_id = await _submit(session, prompt)
        outputs = await _wait_outputs(session, prompt_id)
        result = await _fetch_video(session, outputs, bool(job_input.get("return_base64", True)))
    result.update({"action": action, "prompt_id": prompt_id})
    return result


async def _run_raw(job_input: Dict[str, Any]) -> Dict[str, Any]:
    workflow = job_input.get("workflow")
    if not isinstance(workflow, dict):
        raise ValueError("'workflow' must be an API-format workflow object")
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        prompt_id = await _submit(session, workflow)
        outputs = await _wait_outputs(session, prompt_id)
        result = await _fetch_video(session, outputs, bool(job_input.get("return_base64", True)))
    result["prompt_id"] = prompt_id
    return result


async def _run_proxy(job_input: Dict[str, Any]) -> Any:
    route = job_input.get("route")
    if not isinstance(route, str) or not route.startswith("/"):
        raise ValueError("'route' must be a string starting with '/'")
    method = str(job_input.get("method", "GET")).upper()
    body = job_input.get("body")
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Preserve the caller's method even for bodyless requests (POST/DELETE/
        # HEAD/OPTIONS without a JSON body must not be rewritten to GET).
        if body is None:
            resp_ctx = session.request(method, f"{BASE}{route}")
        else:
            resp_ctx = session.request(method, f"{BASE}{route}", json=body)
        async with resp_ctx as resp:
            # Forward JSON natively; return non-JSON (HTML/text/binary) as text
            # so valid upstream responses don't surface as JSON parse errors.
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "application/json" in ctype:
                return await resp.json(content_type=None)
            text = await resp.text(errors="replace")
            try:
                import json as _json

                return _json.loads(text)  # JSON served with a loose content-type
            except (ValueError, TypeError):
                return {"status": resp.status, "content_type": ctype or None, "body": text}


async def handler(job: Dict[str, Any]) -> AsyncGenerator[Any, None]:
    """RunPod handler entrypoint."""
    job_input = job.get("input") or {}
    try:
        if not isinstance(job_input, dict):
            raise ValueError("job input must be an object")

        if "route" in job_input:
            result = await _run_proxy(job_input)
        elif "workflow" in job_input:
            result = await _run_raw(job_input)
        elif "action" in job_input or "prompt" in job_input:
            result = await _run_generation(job_input)
        else:
            raise ValueError(
                "Unrecognized input. Provide {'action':'generate','prompt':...}, "
                "{'workflow': {...}}, or {'route': '/...', 'method': ..., 'body': ...}."
            )
        yield result
    except asyncio.CancelledError:
        raise
    except (ValueError, RuntimeError, TimeoutError) as exc:
        yield {"error": str(exc)}
    except aiohttp.ClientError as exc:
        yield {"error": f"ComfyUI request failed: {exc}"}
    except Exception as exc:  # noqa: BLE001
        yield {"error": f"handler failure: {exc}"}
