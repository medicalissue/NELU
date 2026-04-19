#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  ONE-TIME setup: prepare an EBS volume with all dependencies,
#  data, and compiled CUDA kernels, then snapshot it.
#
#  Run this ONCE on a temporary p5.48xlarge instance. After it
#  finishes, create an EBS snapshot from the /data volume and use
#  that snapshot ID in launch_spot.sh.
#
#  Steps:
#    1. Launch p5.48xlarge with a 500GB EBS volume on /dev/sdf
#    2. SSH in and run this script
#    3. When done, create snapshot:
#       aws ec2 create-snapshot --volume-id vol-xxx --description "nelu-training-env"
#    4. Note the snapshot ID and set DATA_SNAPSHOT in launch_spot.sh
#    5. Terminate the instance
#
#  The snapshot will contain:
#    /data/
#    ├── imagenet/train/  imagenet/val/
#    ├── cifar-100-python/
#    ├── CIFAR-100-C/
#    ├── ImageNet-C/  (if available)
#    ├── env/          conda environment
#    ├── repos/
#    │   ├── NELU/     this repo
#    │   (no upstream repos needed — all training via timm)
#    └── cache/        compiled CUDA kernels
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
exec > >(tee /var/log/nelu-snapshot-setup.log) 2>&1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/aws_common.sh"

ENV_FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --env-file)
            if [ $# -lt 2 ]; then
                echo "ERROR: --env-file requires a path" >&2
                exit 1
            fi
            ENV_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--env-file FILE]"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

load_env_file "$ENV_FILE" "$REPO_ROOT"

S3_BUCKET="${S3_BUCKET:-s3://nelu-datasets}"

echo "═══════════════════════════════════════════════════════════"
echo "  NELU Training Environment Setup (for EBS snapshot)"
echo "  $(date -u)"
echo "═══════════════════════════════════════════════════════════"

# ── 1. Format and mount the data volume ──────────────────────
echo ""
echo "── 1. Setting up /data volume ──"

# Find the additional EBS volume (not the root)
# Find the EBS data volume.  We attached it as /dev/sdf, but NVMe
# instances rename it.  Strategy: find any block device that is NOT
# the root device and NOT already mounted.
DATA_DEV=""
ROOT_DEV=$(lsblk -no PKNAME $(findmnt -n -o SOURCE /) 2>/dev/null || echo "")
for dev in /dev/sdf /dev/xvdf /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1; do
    [ -b "$dev" ] || continue
    # Skip if this is the root device or a partition of it
    devbase=$(basename "$dev")
    [ "$devbase" = "$ROOT_DEV" ] && continue
    [[ "$devbase" == "${ROOT_DEV}"* ]] && continue
    # Skip if already mounted
    mountpoint -q "$dev" 2>/dev/null && continue
    findmnt -rn -S "$dev" >/dev/null 2>&1 && continue
    DATA_DEV="$dev"
    break
done

if [ -z "$DATA_DEV" ]; then
    echo "WARNING: No additional volume found. Using root volume /data/"
    mkdir -p /data
else
    # Format if not already formatted
    if ! blkid "$DATA_DEV" | grep -q ext4; then
        echo "  Formatting $DATA_DEV as ext4..."
        mkfs.ext4 -q "$DATA_DEV"
    fi
    mkdir -p /data
    mount "$DATA_DEV" /data
    echo "  Mounted $DATA_DEV on /data"
fi

mkdir -p /data/env /data/repos /data/cache

# ── 2. Conda environment ─────────────────────────────────────
echo ""
echo "── 2. Setting up conda environment ──"

# Use the system conda if available (Deep Learning AMI)
CONDA_SH=""
for p in /opt/conda/etc/profile.d/conda.sh \
         /home/ubuntu/miniconda3/etc/profile.d/conda.sh \
         ~/miniconda3/etc/profile.d/conda.sh; do
    [ -f "$p" ] && CONDA_SH="$p" && break
done

if [ -z "$CONDA_SH" ]; then
    echo "  Installing miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p /data/env/miniconda3
    CONDA_SH="/data/env/miniconda3/etc/profile.d/conda.sh"
fi

source "$CONDA_SH"

# Accept conda TOS (required since conda 25.x)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# Create env on the data volume so it's in the snapshot
if [ ! -d /data/env/nelu/bin ]; then
    echo "  Creating conda env at /data/env/nelu..."
    conda create -y -p /data/env/nelu python=3.11
fi
conda activate /data/env/nelu

echo "  Installing PyTorch + dependencies..."
pip install -q \
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -q \
    timm==1.0.11 wandb tqdm scikit-learn scipy ninja matplotlib \
    pyyaml autoattack fvcore

# ── 3. Clone repos ───────────────────────────────────────────
echo ""
echo "── 3. Cloning repos ──"

cd /data/repos

if [ ! -d NELU ]; then
    git clone https://github.com/medicalissue/NELU.git
fi

# No upstream repos needed — all models train via train_imagenet_timm.py (timm-based)

# ── 4. Pre-compile CUDA kernels ──────────────────────────────
echo ""
echo "── 4. Pre-compiling NELU CUDA kernels ──"

cd /data/repos/NELU
TORCH_EXTENSIONS_DIR=/data/cache/torch_extensions \
    python -c "from nelu.cuda_kernel import nelu_cuda; print('NELU CUDA kernel compiled OK')"
TORCH_EXTENSIONS_DIR=/data/cache/torch_extensions \
    python -c "from nelu.nilu_cuda_kernel import nilu_cuda; print('NiLU CUDA kernel compiled OK')"

# ── 5. Download datasets ─────────────────────────────────────
echo ""
echo "── 5. Downloading datasets ──"
DOWNLOAD_CMD=(bash "$REPO_ROOT/scripts/download_data.sh")
if [ -n "$ENV_FILE" ]; then
    DOWNLOAD_CMD+=(--env-file "$ENV_FILE")
fi
DOWNLOAD_CMD+=(/data)
"${DOWNLOAD_CMD[@]}"

# ── 6. Verify ────────────────────────────────────────────────
echo ""
echo "── 6. Verification ──"
echo "  Conda env:    $(which python)"
echo "  PyTorch:      $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA:         $(python -c 'import torch; print(torch.version.cuda)')"
echo "  timm:         $(python -c 'import timm; print(timm.__version__)')"
echo "  GPUs:         $(nvidia-smi -L | wc -l)"
echo "  ImageNet:     $(ls /data/imagenet/train/ 2>/dev/null | wc -l) classes"
echo "  CIFAR-100:    $(ls /data/cifar-100-python/ 2>/dev/null | wc -l) files"
echo "  ImageNet-C:   $(ls /data/ImageNet-C/ 2>/dev/null | wc -l) corruptions"
echo "  ImageNet-A:   $(ls /data/imagenet-a/ 2>/dev/null | wc -l) classes"
echo "  ImageNet-R:   $(ls /data/imagenet-r/ 2>/dev/null | wc -l) classes"
echo "  ImageNet-O:   $(ls /data/imagenet-o/ 2>/dev/null | wc -l) classes"
echo "  NELU kernel:  compiled"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Find the volume ID:"
echo "       INSTANCE_ID=\$(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
echo "       aws ec2 describe-instances --instance-ids \$INSTANCE_ID \\"
echo "         --query 'Reservations[].Instances[].BlockDeviceMappings[?DeviceName==\`/dev/sdf\`].Ebs.VolumeId' \\"
echo "         --output text"
echo ""
echo "    2. Create snapshot:"
echo "       aws ec2 create-snapshot --volume-id <vol-xxx> \\"
echo "         --description 'nelu-training-env-$(date +%Y%m%d)'"
echo ""
echo "    3. Set in launch_spot.sh:"
echo "       DATA_SNAPSHOT=snap-xxxxxxxx  # put this into .env"
echo ""
echo "    4. Terminate this instance"
echo "═══════════════════════════════════════════════════════════"
