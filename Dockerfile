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
RUN git clone --depth 1 --branch "${COMFYUI_REF}" "${COMFYUI_REPO}" /ComfyUI \
    && python3 -m pip install --no-cache-dir -r /ComfyUI/requirements.txt

COPY builder/requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir -r /requirements.txt

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

WORKDIR /src
ENTRYPOINT ["python3", "-u", "/src/main.py"]
