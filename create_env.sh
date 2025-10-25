#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=pretrain
PYTHON_VERSION=3.11
CUDA_VERSION=12.4

# Create env if it doesn’t exist
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

# Install CUDA runtime (optional – PyTorch wheels already bundle CUDA)
conda install -y -n "${ENV_NAME}" nvidia/label/cuda-${CUDA_VERSION}.1::cuda || true

# Always run pip inside env to keep paths consistent
conda run -n "${ENV_NAME}" bash -c "
  set -euo pipefail
  python -m pip install --upgrade pip wheel packaging ninja

  # 1️⃣  Install PyTorch FIRST (official wheel for your CUDA version)
  pip install torch --index-url https://download.pytorch.org/whl/cu124

  # 2️⃣  Core deps
  pip install wandb transformers datasets accelerate deepspeed tensorboard munch \
              peft ipywidgets ipykernel bitsandbytes einops tqdm lightning fvcore trl pyyaml

  # 3️⃣  FlashAttention built cleanly *against this Torch*
  pip install --no-build-isolation --no-cache-dir flash-attn
  pip install liger-kernel==0.6.0
"

echo "✅ Environment '${ENV_NAME}' ready."
echo "Activate with: conda activate ${ENV_NAME}"