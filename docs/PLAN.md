# Plan: `worker_minimax-h3` — RunPod Serverless Worker for MiniMax-H3

> Status: **proposed**. Source research completed 2026-09-14 against the RunPod Hub docs and the `MiniMaxAI/MiniMax-H3` Hugging Face model card.

---

## 1. What this worker is

A RunPod **serverless GPU worker** that serves **[`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)** and is publishable to the RunPod Hub.

The repo is currently empty (fresh clone, no README), so everything below is built from scratch following the canonical `runpod-workers` layout.

### The model (verified — this changes the design)
- **Not a chat LLM.** MiniMax-H3 is a **33B omni-modal video + audio generation model** (diffusion transformer). Inputs: text / image(s) / video / audio → output: **video with synchronized stereo audio**.
- Two open-sourced task variants in one repo: **`fl2va`** (text-to-audio-video, optional first/last frame) and **`ref2va`** (reference-to-audio-video, up to 9 images + 3 videos + 3 audio clips).
- Serving stack per the model card: **SGLang** (`sglang serve --model-variant fl2va|ref2va`), diffusers, or ComfyUI. **Stock vLLM does NOT support it** (vLLM only lists MiniMax M2/M3 LLMs).
- Video: 4–15 s, 24 FPS, multiple aspect ratios; native 32 kHz stereo audio.
- License: **MiniMax H3 Community License** — non-EU/UK/Korea/US territory restriction, >$20M/yr revenue needs authorization, outputs may not train other models. This must be surfaced in the README and Hub description.

### Architecture decision
The public docs do **not** document the exact self-hosted SGLang HTTP request/response shape for this video model (only MiniMax's hosted API is documented: `POST /video-generation-v2-create`, etc.). So we will **not** hard-code a guessed video endpoint.

Instead we follow the **worker-vllm supervisor + generic-proxy pattern**:
- `main.py` launches `sglang serve` as a subprocess, health-polls it, then starts the RunPod serverless loop.
- `handler.py` is a thin async proxy that forwards `job["input"]` to the local SGLang server via a **generic `{route, body, method}` passthrough**, plus a convenience high-level generate action.
- This keeps the worker correct regardless of SGLang's exact video route, and is the pattern the official `worker-vllm` uses.

---

## 2. Repo structure (target tree)

```
worker_minimax-h3/
├── .github/
│   └── workflows/
│       ├── dev.yml            # build+push dev-<branch> image on PR
│       └── release.yml        # build+push vX.Y.Z image on tag/dispatch
├── .runpod/
│   ├── hub.json               # Hub listing metadata + env-var form schema
│   └── tests.json             # CI test inputs + GPU config (gates Hub release)
├── builder/
│   ├── requirements.txt       # runpod SDK + aiohttp + huggingface_hub
│   └── fetch_models.py        # optional build-time model bake (snapshot_download)
├── src/
│   ├── main.py                # supervisor: spawn sglang, health-poll, runpod.serverless.start
│   ├── handler.py             # async proxy handler (route/body passthrough + generate)
│   ├── args_builder.py        # build the `sglang serve` CLI from env vars
│   ├── startup_errors.py      # classify sglang boot failures (OOM, bad variant, gated)
│   └── download_model.py      # shared model-download helper used at bake time
├── public/                    # icon/banner assets referenced by hub.json
├── docs/
│   └── PLAN.md                # this file
├── Dockerfile
├── docker-bake.hcl            # bake targets for CI (linux/amd64)
├── test_input.json            # local-test input for `python src/handler.py --test_input`
├── .gitignore                 # runpod.toml, *.pyc, .env, .DS_Store, test/
├── LICENSE                    # MIT
└── README.md
```

Layout convention copied from `runpod-workers/worker-vllm` (production) and `worker-template` (minimal): code in `src/`, deps in `builder/`, hub metadata in `.runpod/`, CI in `.github/workflows/`.

---

## 3. File-by-file build spec

### `Dockerfile`
Pattern: **wrap the upstream SGLang inference image** (same idea as worker-vllm wrapping `vllm/vllm-openai`).

```dockerfile
ARG SGLANG_VERSION=v0.4.9.post2          # version pinned via ARG, verified after install
FROM sglang/sglang:${SGLANG_VERSION}
ARG SGLANG_VERSION

COPY builder/requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir -r /requirements.txt

# Fail the build if the base image version drifted from the ARG.
RUN test "$(python3 -c 'import sglang; print(sglang.__version__)')" = "${SGLANG_VERSION#v}"

# Optional build-time model bake. Empty default = download at runtime.
ARG MODEL_NAME=""
ARG MODEL_REVISION=""
ARG BASE_PATH="/runpod-volume"
ENV MODEL_NAME=$MODEL_NAME MODEL_REVISION=$MODEL_REVISION BASE_PATH=$BASE_PATH \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
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
    REQUEST_TIMEOUT="3600"

COPY src /src

# Bake model only when MODEL_NAME build-arg set; HF_TOKEN via build secret, never a layer.
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -n "$MODEL_NAME" ]; then \
        if [ -f /run/secrets/HF_TOKEN ]; then export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); fi && \
        python3 /src/download_model.py; \
    fi

ENTRYPOINT ["python3", "/src/main.py"]
```

Key choices and why:
- **Base = `sglang/sglang`** (the verified serving image for this model). Confirm the exact current tag at build time; SGLang tags are version-sensitive.
- **Dual-mode model handling**: runtime download to the network volume (`HF_HOME=/runpod-volume/...`) by default; optional `--build-arg MODEL_NAME=...` bake via `builder/fetch_models.py` (gated models use a build-secret `HF_TOKEN`).
- Layer order deps → code → optional bake so code edits don't bust the dep cache.

### `builder/requirements.txt`
```
runpod~=1.12.0
aiohttp~=3.10
huggingface_hub>=0.24
```
Pin with compatible-release (`~=`), matching the org convention.

### `src/main.py` (supervisor)
- Build the `sglang serve` argv from env via `args_builder.py`:
  `sglang serve --model-path $SGLANG_MODEL_PATH --model-variant $SGLANG_MODEL_VARIANT --num-gpus $SGLANG_NUM_GPUS --ulysses-degree $SGLANG_ULYSSES_DEGREE --performance-mode $SGLANG_PERFORMANCE_MODE --host 0.0.0.0 --port $SGLANG_PORT`
- Spawn subprocess, poll `GET /health` on `SGLANG_PORT` (2 s interval, `SGLANG_STARTUP_TIMEOUT` deadline).
- On startup failure, classify with `startup_errors.py` and keep the worker alive answering every job with the cause (no crash-loop).
- Once healthy: `runpod.serverless.start({"handler": handler, "return_aggregate_stream": True, "concurrency_modifier": lambda _: int(os.getenv("MAX_CONCURRENCY", "4"))})`.
- If `builder/fetch_models.py` wrote `/local_model_args.json` at bake time, read it and override env (offline mode: `HF_HUB_OFFLINE=1`).

### `src/handler.py` (async proxy)
Accepts `job["input"]` in these shapes:
1. **Generic proxy (primary):** `{"route": "<path>", "body": {...}, "method": "POST"}` → forward to local SGLang. This is the safe default given the undocumented video route.
2. **Convenience generate:** `{"action": "generate", "prompt": str, "duration": int, "ratio": "16:9", "image_data"?: [..], "video_data"?: [..], "audio_data"?: [..]}` → maps to the model's generate route (route name resolved at first real SGLang video-model test, with a clear error if unsupported).
3. **OpenAI-style passthrough:** `{"openai_route": ..., "openai_input": ...}` (harmless forward; only meaningful if SGLang exposes an OpenAI-compatible route for this model).
- Streaming: if `body.stream == true`, yield decoded SSE chunks (async generator); else return parsed JSON once.
- Catch `asyncio.CancelledError`, clean up, re-raise. Return `{"error": "..."}` on bad input; raise to mark job FAILED with details.

### `src/args_builder.py`
Pure function env → `sglang serve` argv list. Handles `--model-variant`, `--num-gpus`, `--ulysses-degree`, `--performance-mode`, `--port`, plus any extra `SGLANG_EXTRA_ARGS` passthrough string.

### `src/startup_errors.py`
Map sglang stderr signatures → friendly causes: CUDA OOM, unknown `--model-variant`, HF gated/auth error, port conflict, model-path download failure.

### `src/download_model.py`
`huggingface_hub.snapshot_download(SGLANG_MODEL_PATH, ...)` into `HF_HOME`, honoring `MODEL_REVISION` and `HF_TOKEN`; write resolved local path to `/local_model_args.json`.

### `.runpod/hub.json`
```json
{
  "title": "MiniMax-H3",
  "description": "Serverless worker for MiniMaxAI/MiniMax-H3 — a 33B omni-modal video+audio generation model (text/image/video/audio -> video + stereo audio), served via SGLang. NOTE: MiniMax H3 Community License applies (territory & usage restrictions; outputs may not train other models).",
  "type": "serverless",
  "category": "video",
  "iconUrl": "<public/icon URL>",
  "config": {
    "runsOn": "GPU",
    "containerDiskInGb": 150,
    "gpuIds": "ADA_80_PRO,AMPERE_80",
    "gpuCount": 1,
    "allowedCudaVersions": ["12.8", "12.7", "12.6", "12.5", "12.4"],
    "env": [
      {"key": "SGLANG_MODEL_VARIANT", "input": {"name": "Model variant", "type": "string", "description": "Which H3 task variant to serve.", "default": "fl2va", "options": [{"label": "FL2VA (text-to-audio-video)", "value": "fl2va"}, {"label": "Ref2VA (reference-to-audio-video)", "value": "ref2va"}], "required": true}},
      {"key": "SGLANG_NUM_GPUS", "input": {"name": "Num GPUs", "type": "number", "default": 1, "min": 1, "max": 8, "advanced": true}},
      {"key": "SGLANG_ULYSSES_DEGREE", "input": {"name": "Ulysses degree", "type": "number", "default": 1, "min": 1, "max": 8, "advanced": true}},
      {"key": "SGLANG_PERFORMANCE_MODE", "input": {"name": "Performance mode", "type": "string", "default": "speed", "advanced": true}},
      {"key": "MAX_CONCURRENCY", "input": {"name": "Max concurrency", "type": "number", "default": 4, "min": 1, "max": 64, "advanced": true}},
      {"key": "HF_TOKEN", "input": {"name": "Hugging Face token", "type": "string", "description": "Only if the model is gated for your account.", "advanced": true}}
    ]
  }
}
```
- `category: "video"` (correct Hub category; not `language`).
- `containerDiskInGb: 150` — the model is large (33B DiT + Qwen3-VL-32B encoder + VAEs); confirm against real artifact size and adjust.
- `gpuIds`: start with `ADA_80_PRO,AMPERE_80` (80 GB class, mirrors worker-vllm); the model card's reference is 4 GPUs w/ Ulysses degree 4, so we default `gpuCount: 1` + expose `SGLANG_NUM_GPUS`/`ULYSSES_DEGREE` as advanced inputs. Validate on hardware and widen/narrow the pool + `allowedCudaVersions` to match the actual SGLang base-image CUDA.

### `.runpod/tests.json`
Gate release on a cheap real request. Because video gen is heavy and the route is unverified, keep the test to a small, fast health/smoke input plus a tiny generation; tune `timeout` generously (video gen is slow).
```json
{
  "tests": [
    {
      "name": "health_smoke",
      "input": {"route": "/health", "method": "GET"},
      "timeout": 120000
    },
    {
      "name": "text_to_video_tiny",
      "input": {"action": "generate", "prompt": "a cat sitting on a windowsill", "duration": 4, "ratio": "1:1"},
      "timeout": 600000
    }
  ],
  "config": {
    "gpuTypeId": "NVIDIA A40",
    "gpuCount": 1,
    "env": [{"key": "SGLANG_MODEL_VARIANT", "value": "fl2va"}]
  }
}
```

### `test_input.json` (local testing)
Mirror the convenience shape for `python src/handler.py --test_input`:
```json
{"input": {"action": "generate", "prompt": "a red balloon drifting over a city", "duration": 4, "ratio": "16:9"}}
```

### `.github/workflows/dev.yml` + `release.yml`
Copy the worker-vllm pattern: PR → build+push `dev-<branch>`; tag `v[0-9]+.[0-9]+.[0-9]+*` or `workflow_dispatch` → build+push tagged image. Driven by `docker-bake.hcl`; image coordinates from repo vars (`DOCKERHUB_REPO`, `DOCKERHUB_IMG`) and secrets (`DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `HUGGINGFACE_ACCESS_TOKEN`). A fork only sets its own vars/secrets. (Note: the Hub itself builds from your Dockerfile on a GitHub release; these workflows are for self-publishing images to Docker Hub.)

### `docker-bake.hcl`
One target, `platforms = ["linux/amd64"]`, tags `${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}`, all overridable via bake args.

### `README.md`
Centered title + one-line description + banner, RunPod badge (`https://api.runpod.io/badge/nerdylive123/worker_minimax-h3`), "Current SGLang version: X" line, TOC, then:
- What MiniMax-H3 is (video+audio gen, not a chat LLM).
- **License & usage notice** (territory restriction, >$20M authorization, no training on outputs) — required given the model card.
- Request shapes (the three input modes) with examples.
- Env-var table (Variable / Description / Default / Options).
- Local testing (`--test_input`, `--rp_serve_api`).
- Deployment (Hub + self-hosted image) and GPU guidance.

### `LICENSE`, `.gitignore`
MIT license; `.gitignore` = `runpod.toml`, `*.pyc`, `.env`, `test/`, `.DS_Store`.

---

## 4. Environment variables (final set)

| Env var | Default | Purpose |
|---|---|---|
| `SGLANG_MODEL_PATH` | `MiniMaxAI/MiniMax-H3` | HF repo id or local path |
| `SGLANG_MODEL_VARIANT` | `fl2va` | `fl2va` \| `ref2va` |
| `SGLANG_NUM_GPUS` | `1` | GPUs for serving |
| `SGLANG_ULYSSES_DEGREE` | `1` | Ulysses sequence-parallel degree |
| `SGLANG_PERFORMANCE_MODE` | `speed` | SGLang perf mode |
| `SGLANG_PORT` | `30010` | internal SGLang server port |
| `SGLANG_STARTUP_TIMEOUT` | `1800` (s) | health-poll deadline |
| `SGLANG_EXTRA_ARGS` | `""` | extra CLI args passthrough |
| `MAX_CONCURRENCY` | `4` | runpod concurrency_modifier |
| `REQUEST_TIMEOUT` | `3600` (s) | per-request aiohttp timeout |
| `HF_TOKEN` | — | gated model auth (env or build secret) |
| `HF_HOME` | `/runpod-volume/huggingface-cache/hub` | HF cache on network volume |

---

## 5. Build order

1. Scaffold tree + `.gitignore` + `LICENSE` + `builder/requirements.txt`.
2. `src/args_builder.py`, `src/startup_errors.py`, `src/download_model.py` (pure, unit-testable).
3. `src/handler.py` (proxy + generate), `src/main.py` (supervisor).
4. `Dockerfile` + `docker-bake.hcl`.
5. `.runpod/hub.json` + `.runpod/tests.json` + `test_input.json`.
6. `.github/workflows/dev.yml` + `release.yml`.
7. `README.md` (last, once env vars are final).
8. `docs/PLAN.md` kept as the design record.

## 6. Validation before publishing
- **Local, no GPU:** `python src/handler.py --test_input "$(cat test_input.json)"` to smoke the handler logic (SGLang spawn will fail without GPU — assert the startup-error path returns a clean error, not a crash). `python src/handler.py --rp_serve_api` to exercise the HTTP loop.
- **Docker:** `docker build` for syntax; full run requires a GPU pod.
- **On a real GPU pod:** confirm the actual SGLang video route + request/response, then lock the `generate` action mapping and finalize `hub.json` (`gpuIds`, `allowedCudaVersions`, `containerDiskInGb`) and `tests.json`.
- **Publish:** push, cut a semver GitHub release (`v0.1.0`), let the Hub build + run `tests.json`, then submit for review.

## 7. Open items to verify at build time (version-sensitive)
- Exact current `sglang/sglang` base-image tag and its bundled CUDA → drives `allowedCudaVersions`.
- The real SGLang self-hosted **video generation HTTP route + body** for MiniMax-H3 (drives the `generate` action and `tests.json`). If SGLang's video API is not exposed self-hosted, fall back to serving via diffusers inside the handler instead of SGLang — same repo structure holds.
- Real on-disk model size → finalize `containerDiskInGb`.
- Whether single-GPU BF16 fits in 80 GB, or 4× GPUs are mandatory → finalize `gpuCount`/`gpuIds` defaults.
