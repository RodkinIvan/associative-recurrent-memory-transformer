#!/usr/bin/env bash
set -euo pipefail

# This script runs inside the container

# 1) Ensure env is active
source /opt/conda/etc/profile.d/conda.sh
conda activate pretrain

# 2) Optional: login to wandb if WANDB_API_KEY provided
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "wandb: logging enabled"
else
  echo "wandb: no API key provided; logging disabled"
fi

# 3) Pre-tokenize FineWeb-Edu (streaming → chunked on disk) from its folder
# With in-script streaming tokenization, no separate tokenization step is needed
cd scripts/pretrain

# 4) Launch multi-GPU training using all visible GPUs
# In Docker, pass --gpus all; here we compute NP based on nvidia-smi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)}
export NP=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# Update training launcher (must be run from its folder) to consume all GPUs automatically
sed -i 's/^export CUDA_VISIBLE_DEVICES=.*/export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'/' finetune_armt_inner_gemma3-1b_fineweb_deepspeed.sh

# 4) Launch training from scripts/pretrain
bash ./finetune_armt_inner_gemma3-1b_fineweb_deepspeed.sh


