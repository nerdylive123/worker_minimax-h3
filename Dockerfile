# syntax=docker/dockerfile:1
# RunPod serverless worker for MiniMaxAI/MiniMax-H3 (33B omni video+audio
# generation), served via SGLang. Wraps the upstream SGLang image — same
# pattern as runpod-workers/worker-vllm wrapping vllm/vllm-openai.

ARG SGLANG_VERSION=v0.5.19
FROM lmsysorg/sglang:${SGLANG_VERSION}
ARG SGLANG_VERSION

COPY builder/requirements.txt /requirements.txt
RUN python3 -m ensurepip --upgrade 2>/dev/null || true \
    && python3 -m pip install --no-cache-dir -r /requirements.txt

# Fail the build if the base image's sglang version drifted from the ARG.
RUN test "$(python3 -c 'import sglang; print(sglang.__version__)')" = "${SGLANG_VERSION#v}"

# Optional build-time model bake. Empty MODEL_NAME = download at runtime to the
# network volume (HF_HOME below). Gated models: pass HF_TOKEN as a build secret,
# never a layer (--secret id=HF_TOKEN).
ARG MODEL_NAME=""
ARG MODEL_REVISION=""
ARG BASE_PATH="/runpod-volume"

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    BASE_PATH=$BASE_PATH \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TOKENIZERS_PARALLELISM=false \
    SGLANG_MODEL_PATH="MiniMaxAI/MiniMax-H3" \
    SGLANG_MODEL_VARIANT="fl2va" \
    SGLANG_NUM_GPUS="1" \
    SGLANG_ULYSSES_DEGREE="1" \
    SGLANG_PERFORMANCE_MODE="speed" \
    SGLANG_PORT="30010" \
    SGLANG_STARTUP_TIMEOUT="1800" \
    MAX_CONCURRENCY="4" \
    REQUEST_TIMEOUT="3600" \
    PYTORCH_ALLOC_CONF="expandable_segments:True"

COPY src /src

# Bake the model only when MODEL_NAME build-arg is set.
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -n "$MODEL_NAME" ]; then \
        if [ -f /run/secrets/HF_TOKEN ]; then export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); fi && \
        python3 /src/download_model.py; \
    fi

ENTRYPOINT ["python3", "/src/main.py"]
