#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  Bootstrap a fresh EC2 instance (Ubuntu 22.04 Deep Learning AMI)
#  for NELU training on H100×8.
#
#  Assumes you launched an AWS DL AMI that already has:
#    - NVIDIA driver + CUDA toolkit
#    - Docker
#    - Python / pip
#    - AWS CLI (optional — we install it if missing)
#
#  If you launched a plain Ubuntu 22.04 instead, run install_cuda.sh first.
#
#  Usage (on the new instance):
#      git clone <repo> ResAct && cd ResAct
#      bash scripts/bootstrap_instance.sh
#      conda activate nelu
#      tmux new -s train
#      bash run_h100.sh 2>&1 | tee logs/run_$(date +%Y%m%d_%H%M).log
#
#  What this does:
#    1. Installs miniconda if missing
#    2. Creates the `nelu` conda env (pytorch + timm + datasets + etc)
#    3. Installs AWS CLI if missing
#    4. Creates /data (big EBS or NVMe) and chowns to ubuntu
#    5. Syncs datasets from s3://nelu-datasets/ → /data/
#    6. Runs a quick smoke test (imports + 1-batch forward)
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Config ─────────────────────────────────────────────────────────
S3_BUCKET="${S3_BUCKET:-s3://nelu-datasets}"
DATA_DIR="${DATA_DIR:-/data}"
ENV_NAME="${ENV_NAME:-nelu}"
PY_VER="${PY_VER:-3.11}"
CUDA_WHL="${CUDA_WHL:-cu124}"

# Which datasets to pull. Comment out things you don't need.
SYNC_CIFAR10=true
SYNC_CIFAR100=true
SYNC_CIFAR100C=true       # OOD eval
SYNC_IMAGENET=false       # ~150GB — enable when running Phase 4+5
SYNC_IMAGENETC=false      # ~18GB  — enable when doing ImageNet OOD
SYNC_FINEWEB=false        # ~20GB  — enable when running Phase 3

# ── Helpers ────────────────────────────────────────────────────────
section() { echo -e "\n\033[1;34m══ $* ══\033[0m"; }
have() { command -v "$1" >/dev/null 2>&1; }

section "System info"
echo "  hostname : $(hostname)"
echo "  kernel   : $(uname -r)"
if have nvidia-smi && nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "  no GPU available (CPU-only builder). NELU CUDA kernel will JIT-build on H100."
fi

# ── 1. Miniconda ───────────────────────────────────────────────────
section "1. Miniconda"
if ! have conda; then
    echo "  Installing miniconda..."
    cd /tmp
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
    rm Miniconda3-latest-Linux-x86_64.sh
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda init bash
    cd -
else
    echo "  conda found: $(conda --version)"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# ── 2. nelu env ────────────────────────────────────────────────────
section "2. conda env '$ENV_NAME'"
if conda env list | grep -q "^$ENV_NAME "; then
    echo "  env '$ENV_NAME' already exists — skipping create"
else
    conda create -n "$ENV_NAME" "python=$PY_VER" -y
fi
conda activate "$ENV_NAME"

echo "  Installing torch ($CUDA_WHL)..."
pip install --quiet torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/$CUDA_WHL"

echo "  Installing training deps..."
pip install --quiet \
    timm==1.0.11 \
    wandb \
    tqdm \
    scikit-learn \
    scipy \
    datasets \
    transformers \
    tiktoken \
    ninja \
    matplotlib \
    pandas \
    sentencepiece

# Optional: apex for LAMB. Falls back to AdamW if this fails.
echo "  Installing apex (optional, for DeiT-III LAMB)..."
pip install --quiet packaging
pip install --quiet --no-build-isolation \
    git+https://github.com/NVIDIA/apex.git 2>/dev/null || \
    echo "  WARNING: apex build failed — LM + ImageNet will use AdamW fallback."

# ── 3. AWS CLI + s5cmd ─────────────────────────────────────────────
section "3. AWS CLI + s5cmd"
if ! have aws; then
    echo "  Installing awscli v2..."
    cd /tmp
    curl -sL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
    unzip -q awscliv2.zip
    sudo ./aws/install
    rm -rf aws awscliv2.zip
    cd -
else
    echo "  aws found: $(aws --version)"
fi

# s5cmd — 5-10x faster than 'aws s3 sync' for large datasets
if ! have s5cmd; then
    echo "  Installing s5cmd..."
    cd /tmp
    S5CMD_VER="2.2.2"
    curl -sL "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VER}/s5cmd_${S5CMD_VER}_Linux-64bit.tar.gz" -o s5cmd.tar.gz
    tar -xzf s5cmd.tar.gz s5cmd
    sudo mv s5cmd /usr/local/bin/
    rm -f s5cmd.tar.gz
    cd -
fi
echo "  s5cmd: $(s5cmd version 2>&1 | head -1)"

# Sanity check credentials — they should come from the instance role
# or a credentials file. If missing, data sync will fail.
if ! aws sts get-caller-identity --output text >/dev/null 2>&1; then
    echo ""
    echo "  \033[1;31mWARNING: no AWS credentials detected.\033[0m"
    echo "  Attach an IAM role with s3 read on $S3_BUCKET,"
    echo "  or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY."
fi

# ── 4. /data layout ────────────────────────────────────────────────
section "4. data directory"
if [ ! -d "$DATA_DIR" ]; then
    echo "  $DATA_DIR does not exist. Creating..."
    sudo mkdir -p "$DATA_DIR"
    sudo chown "$(whoami)":"$(whoami)" "$DATA_DIR"
else
    echo "  $DATA_DIR exists."
fi
df -h "$DATA_DIR"

# ── 5. Sync datasets from S3 (parallel s5cmd) ──────────────────────
section "5. sync datasets from $S3_BUCKET"

sync_one() {
    # args: s3_subfolder local_subfolder flag
    local sub="$1" local="$2" flag="$3"
    if ! $flag; then
        echo "  SKIP  $sub  (disabled in config)"
        return
    fi
    local dest="$DATA_DIR/$local"
    if [ -e "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "  OK    $sub  (already present at $dest)"
        return
    fi
    echo "  SYNC  $sub  →  $dest  (s5cmd)"
    mkdir -p "$dest"
    # s5cmd 'cp' with recursive glob — much faster than aws s3 sync
    # --concurrency: per-object parallelism for multipart downloads
    s5cmd --numworkers 64 cp --concurrency 8 \
        "$S3_BUCKET/$sub/*" "$dest/" \
        2>&1 | tail -3
}

# Run independent syncs in parallel where possible
# (small datasets in background, ImageNet sequential since it dominates)
echo "  → starting parallel small-dataset syncs..."
sync_one "cifar-10-batches-py"  "cifar-10-batches-py"  $SYNC_CIFAR10  &
sync_one "cifar-100-python"     "cifar-100-python"     $SYNC_CIFAR100 &
sync_one "CIFAR-100-C"          "CIFAR-100-C"          $SYNC_CIFAR100C &
sync_one "fineweb-edu"          "fineweb-edu"          $SYNC_FINEWEB &
wait
echo "  → small datasets done"

# Big datasets sequentially (avoid throttling, maximize per-sync throughput)
sync_one "imagenet"             "imagenet"             $SYNC_IMAGENET
sync_one "ImageNet-C"           "ImageNet-C"           $SYNC_IMAGENETC

echo ""
df -h "$DATA_DIR"

# ── 6. Link data into the repo so torchvision finds it ─────────────
section "6. symlinks → repo ./data/"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO_ROOT/data"
for sub in cifar-10-batches-py cifar-100-python CIFAR-100-C \
           imagenet ImageNet-C fineweb-edu; do
    if [ -e "$DATA_DIR/$sub" ]; then
        ln -snf "$DATA_DIR/$sub" "$REPO_ROOT/data/$sub"
        echo "  ln  data/$sub  →  $DATA_DIR/$sub"
    fi
done

# ── 7. Smoke test ──────────────────────────────────────────────────
section "7. smoke test"
cd "$REPO_ROOT"
python - <<'PY'
import sys, torch
print(f"  python     : {sys.version.split()[0]}")
print(f"  torch      : {torch.__version__}")
print(f"  cuda ok    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device cnt : {torch.cuda.device_count()}")
    print(f"  gpu 0      : {torch.cuda.get_device_name(0)}")

import timm, wandb, datasets as hfd  # noqa
print("  timm, wandb, datasets ok")

from nelu import NELU, nelu
import torch.nn as nn
x = torch.randn(2, 16, 8, 8, device="cuda" if torch.cuda.is_available() else "cpu")
y = NELU()(x)
print(f"  NELU forward ok  (in {tuple(x.shape)} → {tuple(y.shape)})")
PY

# ── 8. Done ────────────────────────────────────────────────────────
section "DONE"
cat <<EOF

Next steps:
  conda activate $ENV_NAME
  tmux new -s train
  # (optional) wandb login
  bash run_h100.sh 2>&1 | tee logs/run_\$(date +%Y%m%d_%H%M).log

To enable larger datasets, edit this script and flip the flags, then rerun:
  SYNC_IMAGENET   — for Phase 4+5
  SYNC_IMAGENETC  — for Phase 6
  SYNC_FINEWEB    — for Phase 3 (GPT-2 LM)

EOF
