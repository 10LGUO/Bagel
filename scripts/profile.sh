#!/bin/bash
# Short profiling run — captures torch.profiler traces for 5 optimizer steps.
# Traces written to ./profiler_logs/rank<N>/ — view with TensorBoard.
#
# Usage: bash scripts/profile.sh
#   Optionally override paths via env vars:
#     LLM_PATH, VAE_PATH, VIT_PATH, DATA_CFG, OUTPUT_DIR, CKPT_DIR
#
# First-time setup: bash scripts/setup_env.sh <HF_TOKEN>
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

LLM_PATH=${LLM_PATH:-"$REPO_DIR/weights/llm"}
VAE_PATH=${VAE_PATH:-"$REPO_DIR/weights/vae/ae.safetensors"}
VIT_PATH=${VIT_PATH:-"$REPO_DIR/weights/vit"}
# Note: LLM_PATH must be BAGEL's own LLM weights (weights/llm from BAGEL-7B-MoT),
# not a standalone Qwen2.5 checkpoint — BAGEL extends the vocab with image tokens.
DATA_CFG=${DATA_CFG:-"$REPO_DIR/data/configs/example.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/results/profile_run"}
CKPT_DIR=${CKPT_DIR:-"$REPO_DIR/results/profile_run/checkpoints"}

# --- Preflight checks ---
missing=0
for path in "$LLM_PATH" "$VAE_PATH" "$VIT_PATH"; do
  if [ ! -e "$path" ]; then
    echo "ERROR: missing weight path: $path"
    echo "  Run: bash $REPO_DIR/scripts/setup_env.sh <HF_TOKEN>"
    missing=1
  fi
done
[ $missing -ne 0 ] && exit 1

if grep -q "your_data_path" "$REPO_DIR/data/dataset_info.py" 2>/dev/null; then
  echo "ERROR: data/dataset_info.py still contains 'your_data_path' placeholders."
  echo "  Run: bash $REPO_DIR/scripts/setup_env.sh <HF_TOKEN>"
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$CKPT_DIR" "$REPO_DIR/profiler_logs"

echo "=== Launching 4-GPU profiling run ==="
echo "Repo:          $REPO_DIR"
echo "LLM weights:   $LLM_PATH"
echo "VAE weights:   $VAE_PATH"
echo "VIT weights:   $VIT_PATH"
echo "Profiler traces -> $REPO_DIR/profiler_logs/rank*"
echo "To view: tensorboard --logdir $REPO_DIR/profiler_logs"
echo ""

BAGEL_PROFILE=1 \
WANDB_MODE=offline \
PYTHONPATH="$REPO_DIR" \
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  "$REPO_DIR/train/pretrain_unified_navit.py" \
    --dataset_config_file "$DATA_CFG" \
    --llm_path "$LLM_PATH" \
    --vae_path "$VAE_PATH" \
    --vit_path "$VIT_PATH" \
    --interpolate_pos False \
    --vit_max_num_patch_per_side 27 \
    --layer_module Qwen2MoTDecoderLayer \
    --use_flex True \
    --results_dir "$OUTPUT_DIR" \
    --checkpoint_dir "$CKPT_DIR" \
    --max_latent_size 32 \
    --num_shard 4 \
    --num_workers 1 \
    --total_steps 10 \
    --log_every 1 \
    --save_every 999999 \
    --expected_num_tokens 4096 \
    --max_num_tokens 5120 \
    --max_num_tokens_per_sample 4096 \
    --wandb_offline True

echo ""
echo "=== Done. View traces with: ==="
echo "  tensorboard --logdir $REPO_DIR/profiler_logs --bind_all"
