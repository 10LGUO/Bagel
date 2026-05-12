#!/bin/bash
# Run once on a fresh GPU instance to install deps and download assets.
# Usage: bash scripts/setup_env.sh <HF_TOKEN>
#
# After this completes, run: bash scripts/profile.sh
set -e

HF_TOKEN=${1:?"Usage: setup_env.sh <HF_TOKEN>"}

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="/data/bagel_example"

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
if [ ! -d "$DATA_DIR" ]; then
  wget --progress=bar:force -O /tmp/bagel_example.zip \
    https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/bagel_example.zip
  unzip -q /tmp/bagel_example.zip -d /data
  echo "Dataset at $DATA_DIR"
else
  echo "Dataset already present at $DATA_DIR — skipping"
fi

echo "=== Downloading model weights ==="
python3 - <<EOF
import os, re
from huggingface_hub import snapshot_download

os.environ["HF_TOKEN"] = "$HF_TOKEN"
repo_dir = "$REPO_DIR"

# LLM backbone: Qwen2.5-0.5B (lightweight proxy for profiling — not the full 7B).
# The BAGEL tokenizer extended with image tokens is added at train time.
if not os.path.exists(f"{repo_dir}/weights/llm"):
    snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", local_dir=f"{repo_dir}/weights/llm")

# SigLIP VIT: the 384px base model (27×27 = 729 position slots).
# vit_max_num_patch_per_side must be ≤ 27 when using these weights.
if not os.path.exists(f"{repo_dir}/weights/vit"):
    snapshot_download("google/siglip-so400m-patch14-384", local_dir=f"{repo_dir}/weights/vit")

# VAE: ae.safetensors lives at the root of the BAGEL-7B-MoT repo.
if not os.path.exists(f"{repo_dir}/weights/vae/ae.safetensors"):
    os.makedirs(f"{repo_dir}/weights/vae", exist_ok=True)
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        "ByteDance-Seed/BAGEL-7B-MoT",
        filename="ae.safetensors",
        local_dir=f"{repo_dir}/weights/vae",
    )

print("Weights ready.")
EOF

echo "=== Patching dataset_info.py ==="
python3 - <<EOF
import re

info_path = "$REPO_DIR/data/dataset_info.py"
with open(info_path) as f:
    src = f.read()

patched = src.replace("your_data_path/bagel_example", "/data/bagel_example")
if patched == src:
    print("dataset_info.py already patched or uses different placeholder — skipping")
else:
    with open(info_path, "w") as f:
        f.write(patched)
    print("Patched dataset_info.py: your_data_path → /data/bagel_example")
EOF

echo ""
echo "=== Setup complete. Run: bash scripts/profile.sh ==="
