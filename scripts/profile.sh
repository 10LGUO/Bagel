#!/bin/bash
# Short profiling run — captures torch.profiler traces for 5 optimizer steps.
# Traces written to ./profiler_logs/rank<N>/ — view with TensorBoard.
#
# Usage: bash scripts/profile.sh
#   Optionally override paths via env vars:
#     LLM_PATH, VAE_PATH, VIT_PATH, DATA_CFG, OUTPUT_DIR, CKPT_DIR
set -e

LLM_PATH=${LLM_PATH:-"weights/llm"}
VAE_PATH=${VAE_PATH:-"weights/vae/ae.safetensors"}
VIT_PATH=${VIT_PATH:-"weights/vit"}
DATA_CFG=${DATA_CFG:-"data/configs/example.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"results/profile_run"}
CKPT_DIR=${CKPT_DIR:-"results/profile_run/checkpoints"}

mkdir -p "$OUTPUT_DIR" "$CKPT_DIR" profiler_logs

echo "=== Launching 8-GPU profiling run ==="
echo "Profiler traces -> ./profiler_logs/rank*"
echo "To view: tensorboard --logdir ./profiler_logs"
echo ""

BAGEL_PROFILE=1 \
WANDB_MODE=offline \
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  train/pretrain_unified_navit.py \
    --dataset_config_file "$DATA_CFG" \
    --llm_path "$LLM_PATH" \
    --vae_path "$VAE_PATH" \
    --vit_path "$VIT_PATH" \
    --layer_module Qwen2MoTDecoderLayer \
    --use_flex True \
    --results_dir "$OUTPUT_DIR" \
    --checkpoint_dir "$CKPT_DIR" \
    --max_latent_size 32 \
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
echo "  tensorboard --logdir ./profiler_logs --bind_all"
