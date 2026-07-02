#!/bin/bash
# Provision an LTX 2.3 model server on a fresh Vast Serverless worker, then leave it running on :8188.
# Skips clone/downloads that already exist (so weights on a persistent /data volume aren't re-pulled).
set -e
export PATH="/root/.local/bin:$PATH"
MODELS="${LTX_MODELS_DIR:-/workspace/models}"
export LTX_CKPT="${LTX_CKPT:-$MODELS/checkpoints/ltx-2.3-22b-dev.safetensors}"
export LTX_GEMMA="${LTX_GEMMA:-$MODELS/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized}"
export LTX_LORA="${LTX_LORA:-$MODELS/loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors}"
export LTX_UPSCALER="${LTX_UPSCALER:-$MODELS/upscalers/ltx-2.3-spatial-upscaler-x2-1.1.safetensors}"
mkdir -p "$MODELS/checkpoints" "$MODELS/loras" "$MODELS/text_encoders" "$MODELS/upscalers"

echo "[provision] uv + LTX-2 repo"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
UV=$(command -v uv || echo /root/.local/bin/uv)
cd /workspace; [ -d ltx2repo ] || git clone --depth 1 https://github.com/Lightricks/LTX-2.git ltx2repo
cd /workspace/ltx2repo; $UV sync --frozen --extra xformers
$UV pip install -q fastapi uvicorn huggingface_hub 2>/dev/null || .venv/bin/python -m pip install -q fastapi uvicorn huggingface_hub

echo "[provision] weights (skip if already present)"
[ -f "$LTX_CKPT" ]  || .venv/bin/huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-22b-dev.safetensors --local-dir "$MODELS/checkpoints/"
[ -f "$LTX_LORA" ]  || .venv/bin/huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-22b-distilled-lora-384-1.1.safetensors --local-dir "$MODELS/loras/"
[ -d "$LTX_GEMMA" ] || .venv/bin/huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir "$LTX_GEMMA/"
[ -f "$LTX_UPSCALER" ] || .venv/bin/huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.1.safetensors --local-dir "$MODELS/upscalers/"

MODEL_LOG="${MODEL_LOG:-/var/log/portal/ltx.log}"; mkdir -p "$(dirname "$MODEL_LOG")"
echo "[provision] launch model server on :8188 (resident model, log -> $MODEL_LOG)"
cp /workspace/vast-ltx-worker/model_server.py /workspace/ltx2repo/model_server.py
# model_server builds its pipeline args against this dummy frame — must exist on a fresh worker
cp /workspace/vast-ltx-worker/test.png /workspace/test.png
cd /workspace/ltx2repo
setsid env LTX_CKPT="$LTX_CKPT" LTX_GEMMA="$LTX_GEMMA" LTX_LORA="$LTX_LORA" LTX_UPSCALER="$LTX_UPSCALER" LTX_QUANT="${LTX_QUANT:-}" LTX_OFFLOAD="${LTX_OFFLOAD:-none}" \
  LTX_PIPELINE="${LTX_PIPELINE:-one_stage}" LTX_TARGET_HEIGHT="${LTX_TARGET_HEIGHT:-1024}" LTX_TARGET_WIDTH="${LTX_TARGET_WIDTH:-1536}" \
  .venv/bin/python -m uvicorn model_server:app --host 127.0.0.1 --port 8188 > "$MODEL_LOG" 2>&1 < /dev/null &
echo "[provision] model server launched"
