"""Resolve a RunPod-cached Hugging Face model snapshot path.

RunPod's endpoint "Model" caching feature places models under
``/runpod-volume/huggingface-cache/hub`` following Hugging Face cache
conventions (``models--{org}--{name}/snapshots/{hash}``). This worker loads
from that cache first; if the snapshot is absent it falls back to a normal
Hub download (which still lands in the same cache dir via ``HF_HOME``).

Resolver logic follows RunPod's documented helper:
https://docs.runpod.io/serverless/development/huggingface-models#use-cached-models
"""

from __future__ import annotations

import os

HF_CACHE_ROOT = os.environ.get(
    "HF_HUB_CACHE", "/runpod-volume/huggingface-cache/hub"
)


def resolve_snapshot_path(model_id: str, cache_root: str = HF_CACHE_ROOT) -> str:
    """Return the local snapshot dir for a cached model.

    Resolution order:
      1. ``refs/main`` -> the commit hash -> ``snapshots/{hash}``.
      2. Otherwise the first directory under ``snapshots/``.

    Raises:
        ValueError: if ``model_id`` is not an ``org/name`` id.
        RuntimeError: if the model is not present in the cache.
    """
    if "/" not in model_id:
        raise ValueError(
            f"model_id must be 'org/name', got {model_id!r}. "
            "Local paths are not supported by this resolver."
        )

    org, name = model_id.split("/", 1)
    model_root = os.path.join(cache_root, f"models--{org}--{name}")

    # 1. Prefer the commit pinned by refs/main.
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")
    if os.path.isfile(refs_main):
        with open(refs_main, "r", encoding="utf-8") as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            return candidate

    # 2. Fall back to the first available snapshot dir.
    if os.path.isdir(snapshots_dir):
        versions = [
            d
            for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        ]
        if versions:
            versions.sort()
            return os.path.join(snapshots_dir, versions[0])

    raise RuntimeError(f"Cached model not found: {model_id}")


def snapshot_available(model_id: str, cache_root: str = HF_CACHE_ROOT) -> bool:
    """True if the model is already present in the local cache."""
    try:
        resolve_snapshot_path(model_id, cache_root)
        return True
    except (ValueError, RuntimeError):
        return False


def set_offline() -> None:
    """Force offline mode once we're loading purely from the local cache."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
