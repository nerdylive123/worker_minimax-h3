"""RunPod Hub handler entrypoint for the MiniMax-H3 worker.

The RunPod Hub expects a discoverable ``handler.py``. The real implementation
lives in ``src/`` (supervisor ``src/main.py`` + proxy ``src/handler.py``).

Boot path: resolve + map the model, launch ComfyUI, then hand the proxy
handler to ``runpod.serverless.start``. On any startup failure we fall back to
a handler that returns the cause, so the worker answers jobs instead of
crash-looping.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import runpod  # noqa: E402


def _start(handler) -> None:
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})


async def _boot() -> None:
    import main as sup

    try:
        snapshot = sup._resolve_model()
        files = sup._pick_files(snapshot)
        sup._link_into_comfyui(files)
        proc = sup._launch_comfyui()
    except (ValueError, RuntimeError, OSError) as exc:
        _start(sup._startup_error_handler_factory(str(exc)))
        return

    err = await sup._wait_for_health(proc)
    if err is not None:
        _start(sup._startup_error_handler_factory(err))
        return

    print("[handler] ComfyUI healthy; starting RunPod serverless loop.", flush=True)
    import handler as _handler_mod
    import model_files

    _handler_mod._MODEL_FILENAMES.update(
        model_files.resolve_filenames(sup.MODEL_VARIANT, sup.PRECISION)
    )
    _start(_handler_mod.handler)


if __name__ == "__main__":
    try:
        asyncio.run(_boot())
    except Exception as exc:  # noqa: BLE001 - last-resort: report, don't crash-loop
        import main as sup

        _start(sup._startup_error_handler_factory(f"boot failure: {exc}"))
