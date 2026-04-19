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

S3_BUCKET="${S3_BUCKET:-s3://nelu-datasets}"

echo "═══════════════════════════════════════════════════════════"
echo "  NELU Training Environment Setup (for EBS snapshot)"
echo "  $(date -u)"
echo "═══════════════════════════════════════════════════════════"

# ── 1. Format and mount the data volume ──────────────────────
echo ""
echo "── 1. Setting up /data volume ──"

# Find the additional EBS volume (not the root)
DATA_DEV=""
for dev in /dev/nvme1n1 /dev/xvdf /dev/sdf; do
    if [ -b "$dev" ]; then
        DATA_DEV="$dev"
        break
    fi
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

# Create env on the data volume so it's in the snapshot
conda create -y -p /data/env/nelu python=3.11 2>/dev/null || true
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

if [ ! -d /data/imagenet/train ]; then
    echo "  Syncing ImageNet from S3..."
    aws s3 sync "${S3_BUCKET}/imagenet/" /data/imagenet/ --quiet
    echo "  ImageNet: $(find /data/imagenet/train -mindepth 1 -maxdepth 1 | wc -l) classes"
fi

if [ ! -d /data/cifar-100-python ]; then
    echo "  Syncing CIFAR-100..."
    aws s3 sync "${S3_BUCKET}/cifar-100-python/" /data/cifar-100-python/ --quiet
fi

if [ ! -d /data/CIFAR-100-C ]; then
    echo "  Syncing CIFAR-100-C..."
    aws s3 sync "${S3_BUCKET}/CIFAR-100-C/" /data/CIFAR-100-C/ --quiet
fi

# Robustness benchmarks (ImageNet-C, -A, -R, -O)
if [ ! -d /data/ImageNet-C ]; then
    echo "  Syncing ImageNet-C (~30GB)..."
    aws s3 sync "${S3_BUCKET}/ImageNet-C/" /data/ImageNet-C/ --quiet || \
        echo "  WARNING: ImageNet-C not on S3, download manually from hendrycks/robustness"
fi

if [ ! -d /data/imagenet-a ]; then
    echo "  Downloading ImageNet-A (~800MB)..."
    wget -q https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar -O /tmp/imagenet-a.tar && \
        tar xf /tmp/imagenet-a.tar -C /data/ && rm /tmp/imagenet-a.tar || \
        echo "  WARNING: ImageNet-A download failed"
fi

if [ ! -d /data/imagenet-r ]; then
    echo "  Downloading ImageNet-R (~2GB)..."
    wget -q https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar -O /tmp/imagenet-r.tar && \
        tar xf /tmp/imagenet-r.tar -C /data/ && rm /tmp/imagenet-r.tar || \
        echo "  WARNING: ImageNet-R download failed"
fi

if [ ! -d /data/imagenet-o ]; then
    echo "  Downloading ImageNet-O (~20MB)..."
    wget -q https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar -O /tmp/imagenet-o.tar && \
        tar xf /tmp/imagenet-o.tar -C /data/ && rm /tmp/imagenet-o.tar || \
        echo "  WARNING: ImageNet-O download failed"
fi

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
echo "       export DATA_SNAPSHOT=snap-xxxxxxxx"
echo ""
echo "    4. Terminate this instance"
echo "═══════════════════════════════════════════════════════════"
