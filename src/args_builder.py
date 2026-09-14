"""Build the ``sglang serve`` command line from environment variables.

Pure, side-effect-free helpers so they can be unit-tested without a GPU.
"""

from __future__ import annotations

import os
import shlex
from typing import List, Mapping


# Defaults are aligned with the MiniMax-H3 model card's reference serve command
# (which uses 4 GPUs + Ulysses degree 4). We default to single-GPU so the worker
# boots on a standard pod; operators scale up via env vars.
DEFAULTS = {
    "SGLANG_MODEL_PATH": "MiniMaxAI/MiniMax-H3",
    "SGLANG_MODEL_VARIANT": "fl2va",
    "SGLANG_NUM_GPUS": "1",
    "SGLANG_ULYSSES_DEGREE": "1",
    "SGLANG_PERFORMANCE_MODE": "speed",
    "SGLANG_PORT": "30010",
    "SGLANG_HOST": "0.0.0.0",
    "SGLANG_EXTRA_ARGS": "",
}

VALID_VARIANTS = ("fl2va", "ref2va")


def get_config(env: Mapping[str, str] | None = None) -> dict:
    """Return the resolved serving config from the environment."""
    env = env if env is not None else os.environ
    return {key: env.get(key, default) for key, default in DEFAULTS.items()}


def build_serve_argv(env: Mapping[str, str] | None = None) -> List[str]:
    """Build the ``sglang serve`` argv list from environment variables.

    Raises:
        ValueError: if SGLANG_MODEL_VARIANT is not a supported variant.
    """
    cfg = get_config(env)

    variant = cfg["SGLANG_MODEL_VARIANT"].strip().lower()
    if variant not in VALID_VARIANTS:
        raise ValueError(
            f"Unsupported SGLANG_MODEL_VARIANT={cfg['SGLANG_MODEL_VARIANT']!r}. "
            f"Expected one of {VALID_VARIANTS}."
        )

    argv = [
        "sglang",
        "serve",
        "--model-path",
        cfg["SGLANG_MODEL_PATH"],
        "--num-gpus",
        str(int(cfg["SGLANG_NUM_GPUS"])),
        "--ulysses-degree",
        str(int(cfg["SGLANG_ULYSSES_DEGREE"])),
        "--performance-mode",
        cfg["SGLANG_PERFORMANCE_MODE"],
        "--host",
        cfg["SGLANG_HOST"],
        "--port",
        str(int(cfg["SGLANG_PORT"])),
        "--model-variant",
        variant,
    ]

    extra = cfg["SGLANG_EXTRA_ARGS"].strip()
    if extra:
        argv.extend(shlex.split(extra))

    return argv
