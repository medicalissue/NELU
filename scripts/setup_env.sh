#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Create conda environment for NELU experiments.
#
#  Usage:
#      bash scripts/setup_env.sh
# ═══════════════════════════════════════════════════════════════

set -e

ENV_NAME="nelu"

echo "Creating conda env: $ENV_NAME"

conda create -n $ENV_NAME python=3.11 -y
conda activate $ENV_NAME || source activate $ENV_NAME

# Core
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Training
pip install timm wandb tqdm scikit-learn

# Data
pip install datasets transformers

# CUDA kernel build
pip install ninja

# Apex (for LAMB optimizer — DeiT-III ImageNet)
# Optional: only needed for ImageNet experiments
pip install packaging
pip install --no-build-isolation apex -f https://github.com/NVIDIA/apex/releases || \
    echo "WARNING: apex install failed. ImageNet will fallback to AdamW."

echo ""
echo "Environment ready. Activate with:"
echo "  conda activate $ENV_NAME"
