# vast-ltx-worker

An [LTX-2.3](https://github.com/Lightricks/LTX-2) image-to-video GPU worker, packaged to run
on **Vast.ai Serverless**. It takes a single start image + a motion prompt and returns a short
(~4 s) MP4 clip. CubeCrank uses it to add subtle motion to the AI b-roll stills in avatar-format
videos (the "make this still gently move" step).

This repo is **only the worker**. The thing that calls it — endpoint discovery, the route
handshake, retries, motion-QC — lives in the CubeCrank app at
[`lib/server/providers/vast-serverless.ts`](https://github.com/deni-nzt/cubecrank/blob/main/lib/server/providers/vast-serverless.ts).
Read [ARCHITECTURE.md](ARCHITECTURE.md) for the full request lifecycle and the deploy model.

---

## What it is (30-second version)

```
CubeCrank app                         Vast.ai                         one GPU worker
─────────────                         ───────                         ──────────────
vast-serverless.ts ──1. GET /endptjobs──▶ console.vast.ai   (get the endpoint's JWT)
                   ──2. POST /route/  ──▶ run.vast.ai        (wait for a warm worker → its URL)
                   ──3. POST /generate──────────────────────▶ worker.py  (PyWorker, the routed port)
                                                                  │ forwards payload
                                                                  ▼
                                                             model_server.py  (:8188, resident LTX 2.3)
                                                                  │ image_b64 → video_b64
                                                                  ▼
                   ◀────────────  { video_b64, timing }  ───────┘
```

Three processes cooperate on the worker box:

| Process | Port | Role |
|---|---|---|
| **`worker.py`** (Vast PyWorker) | the port Vast routes to | Wraps the model server so the Vast autoscaler can route + scale it. Validates the request envelope, reports a load metric, warms the model with one benchmark clip on boot. |
| **`model_server.py`** (FastAPI) | `127.0.0.1:8188` | The actual LTX 2.3 inference server. Loads the 22B transformer **once** and keeps it resident, then serves `POST /generate` (`image_b64` → `video_b64`). Localhost-only; the PyWorker on the same box is its only client. |
| **`provision.sh`** (boot script) | — | Runs once on cold start: installs the LTX-2 Python env, downloads the model weights, launches `model_server.py`. |

There is **no ComfyUI and no diffusers** here despite the `:8188` port (that's just ComfyUI's
conventional port). Inference calls the official Lightricks `ltx_pipelines` library directly.

---

## Repo layout

| File | What it is |
|---|---|
| `model_server.py` | Resident LTX 2.3 FastAPI server (`:8188`). `POST /generate`, `GET /health`. Two pipeline modes (`one_stage` / `two_stage`) via `LTX_PIPELINE`. |
| `worker.py` | The Vast PyWorker wrapper (routes `/generate`, autoscaling metric, startup benchmark). |
| `provision.sh` | Cold-start bootstrap: `uv sync` the LTX-2 repo, download weights (skips what exists), launch the server on `:8188`. |
| `onstart.sh` | The **canonical Vast endpoint "onstart" script** — the 3 lines the serverless template runs on every cold start (clone → provision → worker). Kept in-repo so the template is version-controlled, not tribal knowledge. |
| `test.png` | 768×512 dummy frame. Required at runtime — the pipeline builds its args against it at import, and the PyWorker uses it for the warm-up benchmark. |
| `requirements.txt` | The PyWorker's **only** direct dep (`vastai`). The heavy deps (torch, xformers, `ltx_pipelines`, fastapi, uvicorn) come from `provision.sh`, not pip — see ARCHITECTURE.md §Dependencies. |
| `.env.example` | Every environment variable the code reads, with defaults + meaning. |

---

## Deploying / creating the endpoint

### First-time setup (create the Vast Serverless endpoint)

The CubeCrank app finds the endpoint **by name** (`VAST_LTX_ENDPOINT`, default `ltx-i2v`), so the
name matters; the numeric id does not need to be hard-coded anywhere.

1. **Create a serverless endpoint** named `ltx-i2v` in the [Vast console](https://cloud.vast.ai/)
   (Serverless → Create). This is currently a **manual console step** — there is no
   create-endpoint script in this repo.
2. **Worker group / search query** — target an 80 GB GPU (H100/A100-80G) for the default bf16
   resident model. Set a **disk floor of ≥160 GB** (weights + venv + CUDA ≈ 90 GB; less ENOSPCs
   the cold start). Set `max_workers` to your concurrency ceiling and `cold_workers` (kept-warm
   count) per your latency/cost trade-off — every extra warm worker bills continuously.
3. **Onstart** — paste the contents of [`onstart.sh`](onstart.sh) into the template's onstart.
4. **Template env** — set at minimum `HF_TOKEN` (the Gemma text encoder is a **gated** model and
   will not download without it). Add `LTX_QUANT=fp8-cast` + an offload mode if you run on
   cheaper/smaller GPUs. See [.env.example](.env.example).

### Shipping a code change

**A plain `git push` does NOT reach the workers.** Production ships worker code as a
**sha256-pinned bootstrap tarball** that the template onstart fetches from the app host — `git
clone` is a *fallback only*, because anonymous GitHub clones of these repos 404 (the account is
shadow-flagged, which is the whole reason the tarball path exists). So the flow is:

1. Edit + commit + `git push` (keeps GitHub in sync — does not deploy).
2. Rebuild the tarball from `HEAD`, publish it to the app host, bump the pin (`BS_SHA`) in the
   template onstart. **Full runbook: [DEPLOY.md](DEPLOY.md).**
3. **Recycle workers.** A Vast *resume* of a stopped worker re-runs the OLD baked code — only a
   **fresh create** re-fetches the tarball. So destroy the current workers and let the autoscaler
   create fresh ones. (Warm workers never hot-reload — the transformer is resident in RAM.)

Reproducibility is pinned on both ends: the tarball is sha256-checked, and `provision.sh` pins the
upstream `Lightricks/LTX-2` checkout to `LTX2_REF` (a fixed commit) so an upstream change can't
break a fresh worker (it did once — see the `LTX2_REF` comment in `provision.sh`).

A fresh clone that is **not** shadow-flagged can ignore all of this and just use the plain
git-clone [`onstart.sh`](onstart.sh) — it's the standard, reproducible path.

---

## The HTTP contract

Two layers. The app talks to the **PyWorker envelope**; the PyWorker forwards the inner **payload**
to the model server. Full field-by-field detail (and the app-side route handshake) is in
[ARCHITECTURE.md](ARCHITECTURE.md).

**What the app POSTs to the routed worker** (`worker.py`'s `/generate`):

```jsonc
{
  "auth_data": { /* the entire body Vast's POST /route/ returned */ },
  "session_id": null,
  "payload": {                         // ← forwarded verbatim to model_server :8188
    "prompt": "the scene is almost perfectly still, only faint natural movement, camera locked",
    "negative_prompt": "worst quality, blurry, morphing, warping, distortion, ...",
    "image_b64": "<base64 PNG/JPEG, no data: prefix>",
    "steps": 8,
    "num_frames": 97,                  // ≈ 4.04 s @ 24-25 fps
    "height": 576,
    "width": 1024                      // app sends 1024×576 (16:9, /32-divisible)
  }
}
```

**What `model_server.py` returns** (top-level; the PyWorker may nest it under `response`):

```jsonc
{
  "video_b64": "<base64 MP4>",
  "timing": { "model_load_s": 44.0, "sample_decode_s": 9.1, "encode_s": 0.6, "total_s": 9.8 },
  "batch_avg_s": 9.8,
  "warm": true,                        // false only on the very first (cold) clip
  "mode": "one_stage"
}
```

On failure it returns `{ "error": "...", "tb": "..." }` (HTTP 200 with an error body — the app
treats a missing `video_b64` as a failure and keeps the beat's still).

`GET /health` → `{ ok, mode, resident, model_load_s, clips_done, batch_avg_s }`.

---

## Models

Downloaded by `provision.sh` into `$LTX_MODELS_DIR` (default `/workspace/models`). Point
`LTX_MODELS_DIR` at a **persistent `/data` volume** to keep weights warm across cold starts (the
existence checks skip anything already present — this is the single biggest cold-start win).

| Role | HF repo → file | Path | Notes |
|---|---|---|---|
| 22B transformer (dev) | `Lightricks/LTX-2.3` → `ltx-2.3-22b-dev.safetensors` | `checkpoints/` | The resident ~44 GB model. |
| Distilled LoRA | `Lightricks/LTX-2.3` → `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | `loras/` | 8-step distillation. |
| Text encoder | `google/gemma-3-12b-it-qat-q4_0-unquantized` | `text_encoders/` | **Gated** — needs `HF_TOKEN`. |
| Spatial upscaler | `Lightricks/LTX-2.3` → `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | `upscalers/` | Only used by `two_stage`. |

The VAE is not a separate download — it ships inside the LTX-2 pipeline library.

---

## Pipeline modes (`LTX_PIPELINE`)

- **`one_stage`** (default) — `TI2VidOneStagePipeline`, 8-step distilled, renders at the
  **request's** `height`/`width`/`steps`. This is what CubeCrank uses (1024×576).
- **`two_stage`** — `DistilledPipeline`: stage-1 at half res → 2× latent upscale → distilled
  refine at `LTX_TARGET_HEIGHT`×`LTX_TARGET_WIDTH` (default 1024×1536). **The request's dims and
  steps are ignored** — the operator picks the tier via env. Same checkpoint + LoRA, plus the
  spatial upscaler; one shared `DiffusionStage` keeps the resident cache a single ~44 GB
  transformer (no VRAM doubling).

---

## Environment variables

See [.env.example](.env.example) for the full annotated list. The ones you actually set on the
endpoint template:

| Var | Default | Why you'd set it |
|---|---|---|
| `HF_TOKEN` | — | **Required.** Gated Gemma text-encoder download. |
| `LTX_MODELS_DIR` | `/workspace/models` | Point at a persistent volume to cache weights. |
| `LTX_QUANT` | `""` (bf16) | `fp8-cast` to fit 48 GB GPUs. |
| `LTX_OFFLOAD` | `none` | Push text-encoder/blocks to CPU on smaller GPUs. |
| `LTX_PIPELINE` | `one_stage` | `two_stage` for the operator-tier upscale path. |

---

## Cold-start behaviour & readiness (read before you debug a "stuck" worker)

Cold start is **long** — first boot downloads ~90 GB of weights and loads a 44 GB transformer, so
a from-scratch worker can take **20–40 min** before its first clip. The app accounts for this: it
prewarms the endpoint at produce start and waits up to ~30–45 min on the route (see ARCHITECTURE.md).

Readiness is signalled to Vast by the **PyWorker** (`worker.py`), carefully sequenced to avoid a
cold-start crash-loop that used to kill most boots:
- `on_load = ["PIPELINE_READY"]` — the PyWorker fires its one warm-up benchmark clip the instant it
  sees this token, with **no wait/retry for the port**. So `model_server.py` emits `PIPELINE_READY`
  from a **background thread only after `GET :8188/health` actually answers**
  (`_announce_ready_when_serving`), guaranteeing the benchmark hits a live socket.
- `"Application startup complete"` was **removed** from `on_load` — uvicorn logs it *before* it
  binds the socket, so it raced the same way.
- `on_error = ["Traceback", "CUDA out of memory", "RuntimeError"]` — fail-fast tokens that abort a
  doomed boot instead of waiting out the whole model-load window.

This was the **2026-07-12 cold-start bug**: `PIPELINE_READY` printed at import (pre-bind) → the
benchmark POSTed a dead `:8188` → Vast marked the worker `error` and restart-looped it (~80% of
cold starts failed). The health-gated emit above fixed it (validated 2/2 clean boots). The signal
to trust when debugging is still a `200` from `GET :8188/health`.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Cold start never finishes / ENOSPC | Disk floor too low — raise the worker group to ≥160 GB. |
| Gemma download fails silently | `HF_TOKEN` missing or not accepted for the gated repo. |
| Worker "ready" but connection refused | The `PIPELINE_READY`-before-bind race (above) — verify `GET :8188/health` returns 200. |
| Clips come back morphed/melting | Expected for close-up liquids/eggs — the app's motion-QC drops these; not a worker bug. |
| Warm workers ignore a code push | By design — drain/restart workers to re-clone `main`. |
| `two_stage` ignores my width/height | By design — `two_stage` renders at `LTX_TARGET_*`, not the request dims. |

---

## Cloning this from scratch — checklist

1. Have a Vast.ai account with credit and an `HF_TOKEN` that has accepted the Gemma license.
2. Create the `ltx-i2v` serverless endpoint (above), disk ≥160 GB, an 80 GB GPU group.
3. Paste `onstart.sh` into the template; set `HF_TOKEN` (+ quant/offload env for smaller GPUs).
4. On the caller side, set `VAST_API_KEY` and `VAST_LTX_ENDPOINT=ltx-i2v` in the app and follow
   [`vast-serverless.ts`](https://github.com/deni-nzt/cubecrank/blob/main/lib/server/providers/vast-serverless.ts)
   for the route → generate handshake.
5. Trigger one clip and watch the model log (`/var/log/portal/ltx.log`) for `MODEL_LOAD … (resident)`
   then `CLIP … total=…`.
