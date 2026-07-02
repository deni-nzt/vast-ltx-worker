#!/bin/bash
# Provision an LTX 2.3 model server on a fresh Vast Serverless worker, then leave it running on :8188.
# Skips clone/downloads that already exist (so weights on a persistent /data volume aren't re-pulled).
set -e
export PATH="/root/.local/bin:$PATH"
MODELS="${LTX_MODELS_DIR:-/workspace/models}"
export LTX_CKPT="${LTX_CKPT:-$MODELS/checkpoints/ltx-2.3-22b-dev.safetensors}"
export LTX_GEMMA="${LTX_GEMMA:-$MODELS/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized}"
export LTX_LORA="${LTX_LORA:-$MODELS/loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors}"
mkdir -p "$MODELS/checkpoints" "$MODELS/loras" "$MODELS/text_encoders"

echo "[provision] uv + LTX-2 repo"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
UV=$(command -v uv || echo /root/.local/bin/uv)
cd /workspace; [ -d ltx2repo ] || git clone --depth 1 https://github.com/Lightricks/LTX-2.git ltx2repo
cd /workspace/ltx2repo; $UV sync --frozen --extra xformers
.venv/bin/python -m pip install -q fastapi uvicorn "huggingface_hub"

echo "[provision] weights (skip if already present)"
[ -f "$LTX_CKPT" ]  || .venv/bin/huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-22b-dev.safetensors --local-dir "$MODELS/checkpoints/"
[ -f "$LTX_LORA" ]  || .venv/bin/huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-22b-distilled-lora-384-1.1.safetensors --local-dir "$MODELS/loras/"
[ -d "$LTX_GEMMA" ] || .venv/bin/huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir "$LTX_GEMMA/"

echo "[provision] launch model server on :8188 (resident model, log -> /workspace/model_server.log)"
cp /workspace/vast-ltx-worker/model_server.py /workspace/ltx2repo/model_server.py
cd /workspace/ltx2repo
setsid env LTX_CKPT="$LTX_CKPT" LTX_GEMMA="$LTX_GEMMA" LTX_LORA="$LTX_LORA" LTX_QUANT="${LTX_QUANT:-}" LTX_OFFLOAD="${LTX_OFFLOAD:-none}" \
  .venv/bin/python -m uvicorn model_server:app --host 127.0.0.1 --port 8188 > /workspace/model_server.log 2>&1 < /dev/null &
echo "[provision] model server launched"
