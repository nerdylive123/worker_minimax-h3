<div align="center">

# MiniMax-H3 — RunPod Serverless Worker

Serverless GPU worker for [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3): a **33B omni-modal video + audio generation** model (text / image / video / audio in → **video with synchronized stereo audio** out), served via **SGLang**.

[![RunPod](https://api.runpod.io/badge/nerdylive123/worker_minimax-h3)](https://www.runpod.io/console/hub/nerdylive123/worker_minimax-h3)

**Current SGLang version:** `v0.5.19`

</div>

---

## ⚠️ Model license & usage notice

MiniMax-H3 is released under the **MiniMax H3 Community License Agreement** (not a permissive OSI license). Key restrictions, per the [model card](https://huggingface.co/MiniMaxAI/MiniMax-H3):

- **Territory-limited** — excludes the EU, UK, Korea, and the US.
- **Commercial use** is allowed, but **> $20M USD/year revenue requires written authorization** from MiniMax.
- **Outputs may not be used to train or improve other AI models.**
- You must display "MiniMax H3" in commercial products that use it.

By deploying this worker you are responsible for complying with that license. This repo's own code is MIT-licensed (see `LICENSE`).

---

## What this is (and isn't)

- ✅ A **video + audio generation** worker. Give it a prompt (and optional reference images/video/audio) and it returns a generated video with native stereo sound.
- ❌ **Not a chat / text LLM.** MiniMax's text models are the M-series (M1, M2, M3). If you want chat completions, use one of those instead.

Two open-sourced task variants are supported via the `SGLANG_MODEL_VARIANT` env var:

| Variant | Name | Input | 
|---|---|---|
| `fl2va` (default) | FL2VA | text → audio-video (optional first/last frame) |
| `ref2va` | Ref2VA | reference images/videos/audio → audio-video (up to 9 images, 3 videos, 3 audio clips) |

## Architecture

This worker follows the `runpod-workers/worker-vllm` **supervisor + proxy** pattern:

```
RunPod job ──► src/handler.py (async proxy) ──► sglang serve (subprocess, localhost)
                     ▲
              src/main.py (supervisor: launch, /health poll, serverless.start)
```

- `src/main.py` builds the `sglang serve` command from env vars, launches it, polls `/health`, then starts the RunPod serverless loop. On boot failure it stays alive and returns the classified cause (OOM, bad variant, gated model, …) instead of crash-looping.
- `src/handler.py` proxies each job to the local SGLang server.

## Request shapes

### 1. Generic proxy (primary — safest)
SGLang's self-hosted video-generation HTTP route is not yet publicly documented, so the recommended mode forwards any route/body directly:

```json
{
  "input": {
    "route": "/v1/videos",
    "method": "POST",
    "body": { "prompt": "a red balloon drifting over a city", "duration": 4, "ratio": "16:9" }
  }
}
```

### 2. Convenience `generate` action
```json
{
  "input": {
    "action": "generate",
    "prompt": "a red balloon drifting over a city",
    "duration": 4,
    "ratio": "16:9",
    "image_data": ["<optional base64 / url>"],
    "video_data": ["<optional>"],
    "audio_data": ["<optional>"]
  }
}
```
The worker tries candidate routes in order (`SGLANG_GENERATE_ROUTE`, `/v1/videos`, `/v1/video/generations`, `/generate`) and uses the first that doesn't 404.

### 3. OpenAI-style passthrough
```json
{ "input": { "openai_route": "/v1/chat/completions", "openai_input": { "...": "..." } } }
```
(Only meaningful if SGLang exposes an OpenAI-compatible route for this model.)

**Streaming:** set `"stream": true` in the body and the handler yields decoded SSE chunks (RunPod `/stream`).

## Environment variables

| Variable | Default | Description | Options |
|---|---|---|---|
| `SGLANG_MODEL_PATH` | `MiniMaxAI/MiniMax-H3` | HF repo id or local model path | |
| `SGLANG_MODEL_VARIANT` | `fl2va` | Task variant to serve | `fl2va`, `ref2va` |
| `SGLANG_NUM_GPUS` | `1` | GPUs for serving | 1–8 |
| `SGLANG_ULYSSES_DEGREE` | `1` | Ulysses sequence-parallel degree | 1–8 |
| `SGLANG_PERFORMANCE_MODE` | `speed` | SGLang performance mode | |
| `SGLANG_PORT` | `30010` | Internal SGLang server port | |
| `SGLANG_STARTUP_TIMEOUT` | `1800` | Health-poll deadline (s) | |
| `SGLANG_EXTRA_ARGS` | `""` | Extra `sglang serve` CLI args | |
| `SGLANG_GENERATE_ROUTE` | auto | Force a specific generate route | |
| `MAX_CONCURRENCY` | `4` | RunPod concurrency modifier | 1–64 |
| `REQUEST_TIMEOUT` | `3600` | Per-request timeout (s) | |
| `HF_TOKEN` | — | HF token for gated models | |
| `HF_HOME` | `/runpod-volume/huggingface-cache/hub` | HF cache root (network volume) | |

> **GPU note:** the model card's reference config uses **4 GPUs with Ulysses degree 4** (33B DiT + Qwen3-VL-32B text encoder). Single-GPU BF16 likely needs an 80 GB-class card; use `SGLANG_NUM_GPUS`/`SGLANG_ULYSSES_DEGREE` to scale. Consumer / lower-VRAM use is better served by the ComfyUI + GGUF/NVFP4 ecosystem.

## Local testing

```bash
# Smoke-test the handler logic (no GPU needed; SGLang spawn will fail and the
# worker should return a clean classified error rather than crash):
python src/handler.py --test_input "$(cat test_input.json)"

# Serve a local HTTP API on :8000 (POST /runsync):
python src/handler.py --rp_serve_api
```

## Build & deploy

```bash
# Runtime download (model fetched to the network volume on first boot):
docker build -t nerdylive123/worker-minimax-h3:latest .

# Bake the model into the image (gated model: pass HF token as a build secret):
docker build \
  --build-arg MODEL_NAME=MiniMaxAI/MiniMax-H3 \
  --secret id=HF_TOKEN,env=HF_TOKEN \
  -t nerdylive123/worker-minimax-h3:latest .
```

Deploy from the [RunPod Hub](https://www.runpod.io/console/hub/nerdylive123/worker_minimax-h3) or point a Serverless endpoint at the built image. The Hub builds from this repo's `Dockerfile` on each GitHub release and gates publication on `.runpod/tests.json`.

CI (`.github/workflows/`) builds `dev-<branch>` images on PRs and tagged `vX.Y.Z` images on release, via `docker-bake.hcl`. Set repo variables `DOCKERHUB_REPO` / `DOCKERHUB_IMG` and secrets `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `HUGGINGFACE_ACCESS_TOKEN`.

## Repo layout

```
├── .github/workflows/   # dev + release CI
├── .runpod/             # hub.json (listing) + tests.json (release gate)
├── builder/             # requirements.txt
├── src/                 # main.py, handler.py, args_builder.py, startup_errors.py, download_model.py
├── docs/PLAN.md         # design plan & research
├── Dockerfile
├── docker-bake.hcl
├── test_input.json
└── README.md
```

## Status / known gaps

- The exact self-hosted SGLang video-generation HTTP route is **unverified** (only MiniMax's hosted API is documented). The generic-proxy mode works regardless; the `generate` convenience mapping will be locked after on-hardware verification. If SGLang's video API is not exposed self-hosted, the fallback is serving via diffusers in-process.
- `containerDiskInGb`, `gpuIds`, and `allowedCudaVersions` in `.runpod/hub.json` are initial estimates to be validated on hardware.
