# Architecture

How a single motion clip is produced, end to end, and why the pieces are shaped the way they are.
Start with the [README](README.md) for the component overview; this doc is the internals.

## The request lifecycle

The app never talks to a worker directly by IP. Vast Serverless is a router: you ask it for a
worker, it hands you a short-lived URL, you POST your job there. Three hops
(app code: [`cubecrank/lib/server/providers/vast-serverless.ts`](https://github.com/deni-nzt/cubecrank/blob/main/lib/server/providers/vast-serverless.ts)):

### 1. Resolve the endpoint's JWT — `GET console.vast.ai/api/v0/endptjobs/`
Authenticated with your account `VAST_API_KEY`. The app finds the endpoint whose
`endpoint_name === "ltx-i2v"` (env `VAST_LTX_ENDPOINT`) and reads its `api_key` — a per-endpoint
**JWT** used for the next two hops. Cached in-process so this only happens once per boot. The
numeric endpoint id is never needed by the app: **discovery is by name.**

### 2. Wait for a worker — `POST run.vast.ai/route/`
```jsonc
{ "endpoint": "ltx-i2v", "api_key": "<endpoint JWT>", "cost": 100, "request_idx": 0, "replay_timeout": 60 }
```
Polled with backoff (2 s → ×1.5 → 15 s cap) until the response contains a ready worker:
`{ "url": "https://<worker>", "request_idx": N, ... }`. On a **cold endpoint this poll is where the
20–40 min wait happens** — the app's route deadline is `VAST_ROUTE_WAIT_MS` (default 30 min, set to
45 min in prod), dropping to `VAST_ROUTE_WAIT_WARM_MS` (~8 min, fail-fast) when a pre-probe already
saw a warm worker. `cost:100` is a real clip; the app's `prewarmVastServerless()` sends `cost:1` at
produce start purely to nudge the autoscaler into booting a worker early.

### 3. Generate — `POST <worker url>/generate`
Authenticated `Authorization: Bearer <endpoint JWT>`. Body is the **PyWorker envelope**:
```jsonc
{ "auth_data": <the entire /route/ response body>, "session_id": null, "payload": <flat model_server Req> }
```
The call uses raw `node:https` with `rejectUnauthorized:false` because Vast fronts workers with a
**private CA** the app can't verify — the channel is still TLS, just not chain-validated. Timeout
`VAST_WORKER_TIMEOUT_MS` (15 min). The response's `video_b64` becomes a
`data:video/mp4;base64,…` URL the app can cache and render.

### On the worker
`worker.py` (the Vast PyWorker) receives hop 3, validates `auth_data`, and forwards `payload`
verbatim to `127.0.0.1:8188/generate` — the FastAPI in `model_server.py`. That server decodes the
image, runs the LTX pipeline under `torch.inference_mode()`, encodes an MP4, and returns
`{ video_b64, timing, … }`. The PyWorker relays it back up the chain.

## The resident-transformer trick — the whole reason this is a server

Loading the 44 GB transformer takes ~40 s. A naive "one CLI call per clip" design would pay that on
every clip. `model_server.py:21-31` monkeypatches the LTX-2 library so the built transformer is
**cached in a module dict and never torn down**:

```python
from ltx_pipelines.utils.blocks import DiffusionStage
_TCACHE = {}
@contextmanager
def _resident_tctx(self, **kwargs):
    k = id(self)
    if k not in _TCACHE:
        _TCACHE[k] = self._build_transformer(**kwargs)   # built once
    yield _TCACHE[k]                                       # reused, no teardown
DiffusionStage._transformer_ctx = _resident_tctx
```

So the process pays the load **once** on the first clip (`warm:false`), then every subsequent clip
is warm (`warm:true`, ~10 s). This is why the worker is a long-lived resident process, and why
`two_stage` deliberately reuses one `DiffusionStage` for both stages — so the cache stays a single
transformer instead of doubling VRAM.

## Dependencies — why `requirements.txt` is one line

The dependency set is **layered**, and only the thinnest layer is pip-installable here:

1. **`requirements.txt` = `vastai`** — the only thing `worker.py` imports directly.
2. **The LTX-2 library** (`ltx_pipelines`, `ltx_core`, torch, xformers) is installed by
   `provision.sh` via `uv sync --frozen --extra xformers` **inside a clone of
   `Lightricks/LTX-2`** — it is that repo's locked env, not ours. `model_server.py` is *copied
   into* the LTX-2 checkout and run from there so those imports resolve. The clone is **pinned to a
   fixed commit** (`LTX2_REF`, currently `7809842…`): upstream's 2026-07-07 change dropped the
   `xformers` extra and hard-killed every fresh worker (`uv sync --frozen --extra xformers` failed
   in ~2 s → the server never launched → Vast recycled the box forever). The sync also falls back
   to a plain `--frozen` sync if the extra vanishes again — xformers is an optimization, never
   worth a dead worker.
3. **`fastapi` + `uvicorn` + `huggingface_hub` + `hf_transfer`** are `uv pip install`ed on top.

This is why you can't `pip install -r requirements.txt && python model_server.py` from a bare
checkout — the server only runs after `provision.sh` has assembled the LTX-2 env around it. That's
a deliberate trade (ride LTX-2's exact locked deps) but it's the #1 thing that surprises a cloner.

## Why not ComfyUI / diffusers

Earlier LTX experiments used both. This worker calls the official Lightricks `ltx_pipelines`
Python API directly because: (a) it exposes the distilled one-stage and two-stage pipelines with
the exact arg parsers LTX ships, (b) it lets us monkeypatch the transformer lifecycle for
residency, and (c) it avoids a ComfyUI graph as an extra moving part. The `:8188` port is a
leftover convention from the ComfyUI days — nothing here is ComfyUI.

## Performance & cost model

The server reports timing on every clip (`timing.{model_load_s, sample_decode_s, encode_s,
total_s}` + `batch_avg_s`), so you can measure rather than guess. The shape of the cost:

- **Cold start** (first worker, from-scratch volume): weights download (~90 GB) + `uv sync` +
  first transformer load ≈ **20–40 min** of GPU wall-clock before clip #1. Amortized across a whole
  video's clips, and mostly avoided by a warm/persistent weights volume + kept-warm workers.
- **Warm per-clip**: `total_s` ≈ **~10 s** at 1024×576 / 97 frames / 8 steps on an 80 GB GPU.
- **The real cost driver at low volume is warm-idle**: a serverless worker bills for the whole
  time it's up, not just the seconds it's rendering. A 20-minute video is only a few dozen motion
  clips (30 % of beats × a few-hundred beats), so **idle time between clips and the kept-warm tail
  dominate** — tune `cold_workers` and the autoscaler's idle-stop, not per-clip speed, to cut cost.
  Per-clip $ = (GPU $/hr ÷ 3600) × (active seconds + its share of idle + amortized cold boot).

The app-side cost tracker + the `vast-watchdog` (alerts on runaway fleet/burn) live in the app repo
and `~/vast-watchdog.py`; this worker just reports timing.
