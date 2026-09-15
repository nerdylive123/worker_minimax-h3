"""Map a (variant, precision) to the Comfy-Org/MiniMax-H3 component filenames.

Shared by the supervisor (which symlinks the files into ComfyUI's models tree)
and the handler (which injects the same filenames into the workflow). The repo
ships single-file safetensors components under:

    diffusion_models/  text_encoders/  vae/  loras/

Precision guidance from the Comfy-Org README:
  * prefer ``int8_convrot`` on PyTorch with cu130;
  * ``fp8_scaled`` only when ``int8_convrot`` is unavailable;
  * ``bf16`` for full quality (needs most VRAM).
"""

from __future__ import annotations

import os
from typing import Dict

VALID_VARIANTS = ("fl2va", "ref2va")
VALID_PRECISIONS = ("bf16", "int8_convrot", "fp8_scaled", "nvfp4_awq")

DIFFUSION_FILE = {
    ("fl2va", "bf16"): "minimax_h3_fl2va_bf16.safetensors",
    ("fl2va", "int8_convrot"): "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    ("fl2va", "fp8_scaled"): "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
    ("ref2va", "bf16"): "minimax_h3_ref2va_bf16.safetensors",
    ("ref2va", "int8_convrot"): "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    ("ref2va", "fp8_scaled"): "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
}

TEXT_ENCODER_FILE = {
    "bf16": "qwen3vl_32b_minimax_h3_bf16.safetensors",
    "int8_convrot": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "nvfp4_awq": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
}

VIDEO_VAE_FILE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE_FILE = "minimax_h3_audio_vae_fp32.safetensors"

TURBO_LORA = {
    "fl2va": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    "ref2va": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
}


def resolve_filenames(
    variant: str = "fl2va", precision: str = "bf16"
) -> Dict[str, str]:
    """Return workflow widget filenames for the given variant/precision."""
    variant = variant.strip().lower()
    if variant not in VALID_VARIANTS:
        raise ValueError(f"variant must be one of {VALID_VARIANTS}, got {variant!r}")
    precision = precision.strip().lower()

    diffusion = DIFFUSION_FILE.get((variant, precision)) or DIFFUSION_FILE[(variant, "bf16")]
    # fp8_scaled has no dedicated text-encoder quant; fall back per README.
    encoder = TEXT_ENCODER_FILE.get(precision) or TEXT_ENCODER_FILE[
        "int8_convrot" if precision == "fp8_scaled" else "bf16"
    ]

    return {
        "unet_name": diffusion,
        "clip_name": encoder,
        "vae_name": VIDEO_VAE_FILE,
        "audio_vae_name": AUDIO_VAE_FILE,
        "turbo_lora": TURBO_LORA.get(variant, TURBO_LORA["fl2va"]),
    }


def filenames_from_env(env=None) -> Dict[str, str]:
    env = env if env is not None else os.environ
    return resolve_filenames(
        env.get("MODEL_VARIANT", "fl2va"),
        env.get("PRECISION", "bf16"),
    )
