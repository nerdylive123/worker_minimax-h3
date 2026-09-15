"""RunPod Hub handler entrypoint for the MiniMax-H3 worker.

The RunPod Hub expects a discoverable ``handler.py``. The real implementation
lives in ``src/`` (supervisor ``src/main.py`` + proxy ``src/handler.py``); this
shim simply adds ``src/`` to the path and runs the supervisor, which launches
ComfyUI and starts the RunPod serverless loop.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from main import _run  # noqa: E402  (path setup must precede import)

if __name__ == "__main__":
    import asyncio

    asyncio.run(_run())
