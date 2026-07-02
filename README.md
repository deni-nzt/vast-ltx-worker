# vast-ltx-worker

LTX 2.3 image-to-video model server + PyWorker for **Vast Serverless**.

- `model_server.py` — resident LTX 2.3 FastAPI on `:8188` (`POST /generate`: `image_b64` → `video_b64`). Checkpoint / quantization / offload via env (`LTX_CKPT` / `LTX_QUANT` / `LTX_OFFLOAD`), so the same image runs **bf16 on 80 GB** or **fp8-cast on cheaper GPUs** without a rebuild.
- `worker.py` — PyWorker wrapper (routes `/generate`, autoscaling metric, warms the model via a startup benchmark).
- `provision.sh` — sets up the LTX-2 env (`uv sync --frozen --extra xformers`) + downloads weights (skips what's present) + launches the model server.
- `test.png` — 768×512 benchmark image.

**Template onstart:**
```
cd /workspace && git clone https://github.com/deni-nzt/vast-ltx-worker.git
cd /workspace/vast-ltx-worker && bash provision.sh
pip install -q vastai && python /workspace/vast-ltx-worker/worker.py
```

**Env:** `HF_TOKEN` (gated Gemma), `LTX_QUANT=fp8-cast` (cheaper GPUs), `LTX_OFFLOAD`, `LTX_MODELS_DIR` (point at a persistent `/data` volume to keep weights warm).
