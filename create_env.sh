#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=pretrain
PYTHON_VERSION=3.10
CUDA_VERSION=12.1   # You use cu121 wheels for PyTorch below

# ---- Channel setup: avoid Anaconda "defaults" (no ToS prompts) ----
conda config --remove-key channels || true
conda config --add channels conda-forge
conda config --set channel_priority strict

# Create env if it doesn’t exist (from conda-forge only)
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" --override-channels -c conda-forge "python=${PYTHON_VERSION}"
fi

# (Optional) Install CUDA runtime *only if you really need it*.
# PyTorch cu124 wheels already bundle CUDA, so you can skip this line entirely.
# If you still want a CUDA runtime from NVIDIA, pin channels explicitly:
conda install -y -n "${ENV_NAME}" --override-channels \
  -c nvidia/label/cuda-${CUDA_VERSION}.1 -c conda-forge \
  cuda || true

# Always run pip inside env to keep paths consistent
conda run -n "${ENV_NAME}" bash -c "
  set -euo pipefail
  python -m pip install --upgrade pip wheel packaging ninja

  # 1) Install PyTorch FIRST (official cu124 wheel)
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

  # 2) Core deps
  python -m pip install \
    wandb transformers==4.57.1 datasets accelerate deepspeed==0.16.4 tensorboard munch \
    peft ipywidgets ipykernel bitsandbytes einops tqdm lightning fvcore trl pyyaml

  # 3) FlashAttention built cleanly *against this Torch*
  python -m pip install --no-build-isolation --no-cache-dir flash-attn
  python -m pip install liger-kernel==0.6.0
"

echo "✅ Environment '${ENV_NAME}' ready."
echo "Activate with: conda activate ${ENV_NAME}"