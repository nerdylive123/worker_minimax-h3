<div align="center">

# MiniMax-H3 — RunPod Serverless Worker

Serverless GPU worker for **MiniMax-H3**, a 33B omni-modal **video + audio generation** model (text / image → **video with synchronized stereo audio**), served via a **headless ComfyUI** backend and loaded from **RunPod's endpoint model cache (cache-first)**.

[![RunPod](https://api.runpod.io/badge/nerdylive123/worker_minimax-h3)](https://www.runpod.io/console/hub/nerdylive123/worker_minimax-h3)

**Model:** [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) (single-file ComfyUI build) · **Backend:** ComfyUI headless

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

- ✅ A **video + audio generation** worker. Give it a text prompt (and optional reference media) and it returns a generated video with native stereo sound.
- ❌ **Not a chat / text LLM.** MiniMax's text models are the M-series.

Two task variants are supported via `MODEL_VARIANT`:

| Variant | Workflow | Input |
|---|---|---|
| `fl2va` (default) | `workflows/t2v.json` | text/image → audio-video (optional first/last frame) |
| `ref2va` | `workflows/r2v.json` | reference images/videos/audio → audio-video |

## Architecture

```
RunPod job ──► src/handler.py ──► inject params into official workflow
                                       │ (workflow_api.gui_to_api)
                                       ▼
                          ComfyUI headless (localhost:8188)
                                       ▲
              src/main.py: resolve cached model, symlink components into
              ComfyUI/models/..., launch ComfyUI, poll health
```

- **`src/main.py`** (supervisor): resolves the cached `Comfy-Org/MiniMax-H3` snapshot (**cache-first**), symlinks the chosen variant/precision single-file components into ComfyUI's `models/` tree, launches ComfyUI headless, polls health, then starts the RunPod serverless loop. On startup failure it stays alive and returns the cause instead of crash-looping.
- **`src/handler.py`**: builds the per-request workflow and proxies to ComfyUI's `/prompt`, `/history`, and `/view` endpoints; returns the generated video (base64 by default).
- **`src/workflow_api.py`**: converts the official Comfy-Org GUI workflows to ComfyUI's API `/prompt` format (expanding the MiniMax-H3 subgraph) and injects per-request parameters.
- **`src/model_cache.py`**: RunPod's `resolve_snapshot_path` helper for `/runpod-volume/huggingface-cache/hub`; sets offline mode when loading purely from cache.
- **`src/model_files.py`**: maps `(variant, precision)` → the exact single-file component filenames.

## Cache-first model loading

Set the endpoint's **Model** field to `Comfy-Org/MiniMax-H3` (this is the default `MODEL_NAME`). RunPod then:

1. Places the model at `/runpod-volume/huggingface-cache/hub/models--Comfy-Org--MiniMax-H3/snapshots/{hash}/` on hosts that have it (cold starts in seconds, **download time is not billed**), or
2. Downloads it onto the target host on first use.

The worker resolves that snapshot first and loads offline. If it's absent (e.g. local dev), it downloads into the same cache dir via `HF_HOME`. 

> **Note (RunPod limitation):** RunPod currently downloads **all** quantization variants of a repo when caching it (per-model quantization selection is a planned feature). `Comfy-Org/MiniMax-H3` ships ~10 diffusion quants + 3 text-encoder quants + LoRAs, so the **first** cache pull is large (hundreds of GB); subsequent workers on cached hosts start fast. Size your container disk / network volume accordingly (`containerDiskInGb: 200` by default).

## Request shapes

### 1. Generate (high-level, recommended)
```json
{
  "input": {
    "action": "t2v",
    "prompt": "a red balloon drifting over a city at dusk, cinematic",
    "width": 736,
    "height": 416,
    "duration": 5,
    "seed": 12345,
    "turbo": true,
    "turbo_steps": 8,
    "return_base64": true
  }
}
```
- `action`: `t2v` / `generate` (uses `workflows/t2v.json`) or `r2v` / `ref2va` (uses `workflows/r2v.json`).
- Optional per-request overrides: `width`, `height`, `duration` (s), `seed`, `turbo`, `turbo_steps`, `turbo_strength`, `ratio`.
- `return_base64` (default `true`): returns the video as `video_base64` — **the only remotely-usable delivery mode**, since a ComfyUI `/view` URL is loopback-only (`127.0.0.1`) and unreachable by remote callers. With `false` the worker still fetches the bytes to confirm the artifact and returns the on-worker path + size (in-pod debugging only).
- Model component filenames (unet/clip/vae/lora) are fixed at container start from `MODEL_VARIANT` + `PRECISION`.
- **R2V reference media:** the bundled `r2v.json` ships demo reference images that aren't in the image, so for a bare text prompt the worker strips those `LoadImage` nodes and runs text-only. To use reference images/video/audio, supply them via a **raw workflow** (mode 2) with your own `LoadImage`/uploaded inputs.

Response:
```json
{ "video_base64": "<...>", "filename": "MiniMax_H3_00001.mp4", "action": "t2v", "prompt_id": "..." }
```

### 2. Raw workflow (full control)
Submit any API-format ComfyUI workflow verbatim:
```json
{ "input": { "workflow": { "<node_id>": { "class_type": "...", "inputs": { } } } } }
```

### 3. Proxy (arbitrary ComfyUI route)
```json
{ "input": { "route": "/system_stats", "method": "GET" } }
```

## Environment variables

| Variable | Default | Description | Options |
|---|---|---|---|
| `MODEL_NAME` | `Comfy-Org/MiniMax-H3` | HF repo to load from the endpoint model cache | |
| `MODEL_VARIANT` | `fl2va` | Task variant | `fl2va`, `ref2va` |
| `PRECISION` | `int8_convrot` | Weight quantization | `int8_convrot`, `bf16`, `fp8_scaled`, `nvfp4_awq` |
| `COMFYUI_PORT` | `8188` | Internal ComfyUI port | |
| `COMFYUI_STARTUP_TIMEOUT` | `900` | Health-poll deadline (s) | |
| `COMFYUI_EXTRA_ARGS` | `""` | Extra ComfyUI CLI args | |
| `GENERATION_TIMEOUT` | `1800` | Per-generation timeout (s) | |
| `REQUEST_TIMEOUT` | `3600` | HTTP client timeout (s) | |
| `HF_TOKEN` | — | HF token for gated models | |
| `HF_HOME` | `/runpod-volume/huggingface-cache/hub` | HF cache root (matches the endpoint model-cache mount) | |

> **Precision guide** (from the Comfy-Org README): prefer `int8_convrot` on PyTorch with cu130 (the default base image); use `fp8_scaled` only if `int8_convrot` is unavailable; `bf16` for full quality at the highest VRAM cost.

> **GPU note:** MiniMax-H3 is a 33B DiT with a Qwen3-VL-32B text encoder. Plan for an 80 GB-class GPU for bf16; the `int8_convrot` / `fp8_scaled` quants reduce VRAM. The default hub.json targets the 80 GB pool.

## Local testing

```bash
# Handler smoke-test (no GPU; ComfyUI won't be up, so the worker should return
# a clean error rather than crash):
python src/handler.py --test_input "$(cat test_input.json)"

# Serve a local HTTP API on :8000 (POST /runsync):
python src/handler.py --rp_serve_api
```

The workflow converter is pure Python and can be exercised without a GPU:
```bash
python -c "import sys; sys.path.insert(0,'src'); import workflow_api as w; \
  wf=w.load_workflow('workflows/t2v.json'); w.set_subgraph_params(wf,{'prompt':'x'}); \
  api=w.gui_to_api(wf); print(len(api), 'nodes')"
```

## Build & deploy

```bash
docker build -t nerdylive123/worker-minimax-h3:latest .
```

Deploy from the [RunPod Hub](https://www.runpod.io/console/hub/nerdylive123/worker_minimax-h3) or point a Serverless endpoint at the built image. Set the endpoint **Model** field to `Comfy-Org/MiniMax-H3` to enable cache-first loading. The Hub builds from this repo's `Dockerfile` on each GitHub release and gates publication on `.runpod/tests.json`.

CI (`.github/workflows/`) builds `dev-<branch>` images on PRs and tagged `vX.Y.Z` images on release, via `docker-bake.hcl`. Set repo variables `DOCKERHUB_REPO` / `DOCKERHUB_IMG` and secrets `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`.

## Repo layout

```
├── .github/workflows/   dev + release CI
├── .runpod/             hub.json (listing) + tests.json (release gate)
├── builder/             requirements.txt
├── src/                 main.py, handler.py, workflow_api.py, model_cache.py, model_files.py
├── workflows/           official Comfy-Org MiniMax-H3 templates (t2v.json, r2v.json)
├── docs/PLAN.md         design plan & research
├── Dockerfile
├── docker-bake.hcl
├── test_input.json
└── README.md
```

## Status / known gaps

- The workflow converter (`workflow_api._manual_gui_to_api`) is validated against the bundled official T2V/R2V templates (26 and 29 nodes; no dangling references). On a live ComfyUI worker, ComfyUI's own exporter is preferred automatically when importable; the manual flattener is the fallback. **First real generation on a GPU pod should confirm the flattened graph executes identically to the GUI template.**
- `containerDiskInGb` (200) is sized for RunPod's all-quants cache behavior; if RunPod adds per-quant selection this can shrink substantially.
- Output is returned via ComfyUI's `SaveVideo` node (`/view`); confirm the exact filename/format keys on first run.
