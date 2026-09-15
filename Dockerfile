# syntax=docker/dockerfile:1
# RunPod serverless worker for MiniMax-H3 (video + audio generation), served
# via a headless ComfyUI backend. Loads Comfy-Org/MiniMax-H3 single-file
# components from RunPod's endpoint model cache (cache-first), with a Hub
# download fallback into the same cache dir.
#
# Base: runpod/pytorch with cu130 so the int8_convrot quant path works (the
# Comfy-Org README prefers int8_convrot on cu130; fp8_scaled / bf16 are the
# fallbacks). Override PRECISION at deploy time to switch.

ARG BASE_IMAGE=runpod/pytorch:1.3.0-cu1300-torch291-ubuntu2404
FROM ${BASE_IMAGE}

ARG COMFYUI_REPO=https://github.com/comfyanonymous/ComfyUI.git
ARG COMFYUI_REF=master

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# --- ComfyUI ---------------------------------------------------------------
# Clone ComfyUI and install its requirements. The MiniMax-H3 nodes
# (MiniMaxH3ImageToVideo, SaveVideo, CreateVideo, VAEDecodeAudio, ...) are in
# recent ComfyUI; pin COMFYUI_REF to a dated commit for reproducible builds.
# --ignore-installed avoids pip trying to uninstall Debian/apt-managed Python
# packages (e.g. cryptography) that have no RECORD file (uninstall-no-record-file).
RUN git clone --depth 1 --branch "${COMFYUI_REF}" "${COMFYUI_REPO}" /ComfyUI \
    && python3 -m pip install --no-cache-dir --ignore-installed -r /ComfyUI/requirements.txt

# Model-loader deps for MiniMax-H3: Qwen3-VL text encoder needs a recent
# transformers (>=4.51) plus safetensors/accelerate/diffusers. Pin floors so an
# older base image can't silently break model loading at startup.
RUN python3 -m pip install --no-cache-dir --ignore-installed \
    "transformers>=4.51.0" "accelerate>=1.0.0" "safetensors>=0.4.5" "diffusers>=0.33.0" "sentencepiece" "protobuf"

COPY builder/requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir --ignore-installed -r /requirements.txt

# --- Runtime configuration --------------------------------------------------
# HF cache root matches RunPod's endpoint model-cache mount so cached models
# are found first and downloads persist on the network volume.
ARG BASE_PATH="/runpod-volume"
ENV BASE_PATH=$BASE_PATH \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TOKENIZERS_PARALLELISM=false \
    MODEL_NAME="Comfy-Org/MiniMax-H3" \
    MODEL_VARIANT="fl2va" \
    PRECISION="int8_convrot" \
    COMFYUI_DIR="/ComfyUI" \
    COMFYUI_PORT="8188" \
    COMFYUI_STARTUP_TIMEOUT="900" \
    GENERATION_TIMEOUT="1800" \
    REQUEST_TIMEOUT="3600" \
    PYTORCH_ALLOC_CONF="expandable_segments:True"

COPY src /src
COPY workflows /workflows
COPY handler.py /handler.py

WORKDIR /src
# Root handler.py is the Hub-discoverable entrypoint; it adds src/ to the path
# and runs the supervisor (which launches ComfyUI + the serverless loop).
ENTRYPOINT ["python3", "-u", "/handler.py"]
