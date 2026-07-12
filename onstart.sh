#!/bin/bash
# STANDARD / reference Vast Serverless onstart for the ltx-i2v endpoint — the clean, reproducible
# path: clone this repo -> provision.sh (env + weights + launch model server on :8188) -> worker.py
# (the Vast PyWorker that routes /generate and warms the model). Paste these lines into the endpoint
# template's onstart. Required template env: HF_TOKEN (gated Gemma). Optional: LTX_QUANT,
# LTX_OFFLOAD, LTX_MODELS_DIR, LTX_PIPELINE — see .env.example.
#
# NOTE — THIS account's PRODUCTION does NOT use the git clone below. Anonymous GitHub clones of
# deni-nzt repos 404 (the account is shadow-flagged), so production fetches a sha256-pinned tarball
# from the app host and uses this git clone only as a FALLBACK. See DEPLOY.md for the real deploy.
set -e

cd /workspace && git clone https://github.com/deni-nzt/vast-ltx-worker.git
cd /workspace/vast-ltx-worker && bash provision.sh
pip install -q vastai && python /workspace/vast-ltx-worker/worker.py
