"""Classify SGLang startup failures into actionable, user-facing causes.

The supervisor keeps the worker alive after a boot failure and answers every
job with the classified cause instead of crash-looping the container.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StartupError:
    """A classified startup failure."""

    category: str
    message: str
    hint: str

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "message": self.message,
            "hint": self.hint,
        }


# Ordered (signatures, classification) rules. First match wins.
_RULES = [
    (
        ("out of memory", "outofmemoryerror", "cuda out of memory", "oom"),
        StartupError(
            category="cuda_oom",
            message="GPU out of memory while loading MiniMax-H3.",
            hint=(
                "MiniMax-H3 is a 33B DiT with a Qwen3-VL-32B text encoder. Use a "
                "larger GPU (80GB class) or increase SGLANG_NUM_GPUS / "
                "SGLANG_ULYSSES_DEGREE for multi-GPU parallelism."
            ),
        ),
    ),
    (
        ("model-variant", "unrecognized arguments", "invalid choice"),
        StartupError(
            category="bad_variant",
            message="SGLang rejected the model variant or CLI arguments.",
            hint=(
                "SGLANG_MODEL_VARIANT must be 'fl2va' or 'ref2va'. Also check "
                "SGLANG_EXTRA_ARGS for typos."
            ),
        ),
    ),
    (
        ("gated repo", "access to model", "401", "403", "repository not found", "authorization"),
        StartupError(
            category="hf_auth",
            message="Could not download the model from Hugging Face (auth/gated).",
            hint=(
                "Set HF_TOKEN with access to MiniMaxAI/MiniMax-H3 and accept the "
                "MiniMax H3 Community License on the model page."
            ),
        ),
    ),
    (
        ("address already in use", "port is already allocated", "bind"),
        StartupError(
            category="port_conflict",
            message="SGLang could not bind its port.",
            hint="Change SGLANG_PORT to a free port.",
        ),
    ),
    (
        ("connection refused", "failed to connect", "name or service not known", "network"),
        StartupError(
            category="network",
            message="Network error while starting SGLang / downloading weights.",
            hint="Check outbound network access, or bake the model into the image.",
        ),
    ),
]


def classify(log_text: str) -> StartupError:
    """Return the best-matching StartupError for the given sglang stderr/stdout."""
    lowered = (log_text or "").lower()
    for signatures, error in _RULES:
        for sig in signatures:
            if sig in lowered:
                return error
    return StartupError(
        category="unknown",
        message="SGLang failed to start for an unclassified reason.",
        hint="Inspect the worker logs for the underlying sglang error.",
    )
