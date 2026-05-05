#!/bin/bash
# Run once on a fresh 8xA100 instance to install deps and download assets.
# Usage: bash scripts/setup_env.sh <HF_TOKEN>
set -e

HF_TOKEN=${1:?"Usage: setup_env.sh <HF_TOKEN>"}

echo "=== Installing system deps ==="
apt-get update -qq && apt-get install -y -qq unzip wget git-lfs

echo "=== Installing Python deps ==="
pip install -q --upgrade pip
pip install -q \
  torch==2.5.1 torchvision==0.20.1 \
  transformers==4.49.0 accelerate safetensors sentencepiece \
  einops pyarrow scipy opencv-python-headless decord \
  ninja wheel setuptools triton \
  bitsandbytes wandb tensorboard \
  huggingface_hub

# flash_attn must be compiled for the local CUDA version
pip install -q flash-attn==2.5.8 --no-build-isolation || \
  echo "WARNING: flash_attn build failed — install manually if needed"

echo "=== Downloading sample dataset ==="
wget -q -O /tmp/bagel_example.zip \
  https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/bagel_example.zip
unzip -q /tmp/bagel_example.zip -d /data
echo "Dataset at /data/bagel_example"

echo "=== Downloading model weights ==="
python3 - <<EOF
from huggingface_hub import snapshot_download
import os
os.environ["HF_TOKEN"] = "$HF_TOKEN"

# Qwen2.5-0.5B as lightweight LLM backbone for profiling (not full 14B)
snapshot_download("Qwen/Qwen2.5-0.5B-Instruct",  local_dir="weights/llm")
snapshot_download("google/siglip-so400m-patch14-384", local_dir="weights/vit")
# VAE — download from BAGEL HF repo
snapshot_download("ByteDance-Seed/BAGEL-7B-MoT",
                  allow_patterns=["vae/*"], local_dir="weights")
EOF

echo ""
echo "=== Setup complete ==="
echo "Next: edit data/dataset_info.py to point to /data/bagel_example, then run scripts/profile.sh"
