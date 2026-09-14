"""Download the MiniMax-H3 model into the local HF cache (used at build time).

Reads config from the environment:
  MODEL_NAME / SGLANG_MODEL_PATH   HF repo id or local path
  MODEL_REVISION                   optional git revision to pin
  HF_HOME                          cache root (network volume by default)
  HF_TOKEN                         token for gated models (passed via build secret)

Writes the resolved local path to /local_model_args.json so the supervisor can
switch to offline mode at boot.
"""

from __future__ import annotations

import json
import os

from huggingface_hub import snapshot_download

LOCAL_MODEL_ARGS_PATH = "/local_model_args.json"


def main() -> None:
    model_name = os.environ.get("MODEL_NAME") or os.environ.get(
        "SGLANG_MODEL_PATH", "MiniMaxAI/MiniMax-H3"
    )
    revision = os.environ.get("MODEL_REVISION") or None

    local_path = snapshot_download(
        repo_id=model_name,
        revision=revision,
        token=os.environ.get("HF_TOKEN"),
        ignore_patterns=["*.md", "*.gitattributes"],
    )

    with open(LOCAL_MODEL_ARGS_PATH, "w", encoding="utf-8") as f:
        json.dump({"model_path": local_path, "revision": revision}, f)

    print(f"[download_model] Baked {model_name} -> {local_path}")


if __name__ == "__main__":
    main()
