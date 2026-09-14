"""RunPod serverless handler for MiniMax-H3.

Thin async proxy in front of a locally-running ``sglang serve`` instance.

Because SGLang's self-hosted video-generation HTTP route is not publicly
documented, the primary input mode is a generic ``{route, body, method}``
passthrough — correct regardless of the exact route SGLang exposes. A
convenience ``generate`` action and an OpenAI-style passthrough are also
accepted.

Accepted ``job["input"]`` shapes
--------------------------------
1. Generic proxy (primary)::

       {"route": "/v1/videos", "body": {...}, "method": "POST"}

2. Convenience generate::

       {"action": "generate", "prompt": "...", "duration": 4, "ratio": "16:9",
        "image_data": [...], "video_data": [...], "audio_data": [...]}

3. OpenAI-style passthrough::

       {"openai_route": "/v1/chat/completions", "openai_input": {...}}

Streaming: if ``body["stream"]`` is true, yields decoded SSE chunks as an async
generator; otherwise returns the parsed JSON once.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncGenerator, Dict, Tuple

import aiohttp

from args_builder import get_config

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "3600"))
# Candidate routes for the convenience generate action, tried in order. The
# first that does not 404 is used; override with SGLANG_GENERATE_ROUTE.
_GENERATE_ROUTE_CANDIDATES = [
    os.getenv("SGLANG_GENERATE_ROUTE", ""),
    "/v1/videos",
    "/v1/video/generations",
    "/generate",
]


def _base_url() -> str:
    cfg = get_config()
    return f"http://127.0.0.1:{cfg['SGLANG_PORT']}"


def _resolve_request(job_input: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any] | None]:
    """Map a job input to (method, path, body). Raises ValueError on bad input."""
    if not isinstance(job_input, dict):
        raise ValueError("job input must be an object")

    # 1. Generic proxy.
    if "route" in job_input:
        route = job_input["route"]
        if not isinstance(route, str) or not route.startswith("/"):
            raise ValueError("'route' must be a string starting with '/'")
        method = str(job_input.get("method", "POST")).upper()
        body = job_input.get("body")
        return method, route, body

    # 2. Convenience generate action.
    if job_input.get("action") == "generate":
        body = {
            k: v
            for k, v in {
                "prompt": job_input.get("prompt"),
                "duration": job_input.get("duration"),
                "ratio": job_input.get("ratio"),
                "image_data": job_input.get("image_data"),
                "video_data": job_input.get("video_data"),
                "audio_data": job_input.get("audio_data"),
            }.items()
            if v is not None
        }
        if "prompt" not in body:
            raise ValueError("'prompt' is required for action='generate'")
        # Route resolved at call time (see _post_generate) to allow fallback.
        return "GENERATE", "", body

    # 3. OpenAI-style passthrough.
    if "openai_route" in job_input:
        route = job_input["openai_route"]
        body = job_input.get("openai_input")
        return "POST", route, body

    raise ValueError(
        "Unrecognized input. Provide {'route','body','method'}, "
        "{'action':'generate',...}, or {'openai_route','openai_input'}."
    )


async def _post_generate(
    session: aiohttp.ClientSession, body: Dict[str, Any]
) -> aiohttp.ClientResponse:
    """POST a generate request, falling back across candidate routes."""
    base = _base_url()
    last_resp: aiohttp.ClientResponse | None = None
    for route in _GENERATE_ROUTE_CANDIDATES:
        if not route:
            continue
        resp = await session.post(f"{base}{route}", json=body)
        if resp.status != 404:
            return resp
        await resp.release()
        last_resp = resp
    if last_resp is not None:
        return last_resp
    raise RuntimeError("No generate route configured (SGLANG_GENERATE_ROUTE unset).")


async def handler(job: Dict[str, Any]) -> AsyncGenerator[Any, None]:
    """RunPod handler. Async generator so streaming jobs yield chunks."""
    try:
        method, route, body = _resolve_request(job.get("input"))
    except ValueError as exc:
        yield {"error": str(exc)}
        return

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    stream = isinstance(body, dict) and bool(body.get("stream"))

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if method == "GENERATE":
                resp = await _post_generate(session, body or {})
            elif method == "GET" or body is None:
                resp = await session.get(f"{_base_url()}{route}")
            else:
                resp = await session.request(method, f"{_base_url()}{route}", json=body)

            async with resp:
                if stream:
                    async for chunk in resp.content.iter_any():
                        if chunk:
                            yield chunk.decode("utf-8", errors="replace")
                else:
                    try:
                        yield await resp.json(content_type=None)
                    except Exception:
                        yield {"status": resp.status, "body": await resp.text()}
    except asyncio.CancelledError:
        # RunPod cancelled the job: clean up and propagate.
        raise
    except aiohttp.ClientError as exc:
        yield {"error": f"upstream request failed: {exc}"}
    except Exception as exc:  # noqa: BLE001 - surface unexpected proxy errors
        yield {"error": f"handler failure: {exc}"}
