#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  EC2 user data script — runs on instance first boot.
#
#  Bootstraps the environment, downloads code and data from S3,
#  then starts training. Auto-shuts down when complete.
#
#  Placeholders replaced by launch_spot.sh:
#    __S3_BUCKET__  — S3 bucket URI
#    __NODE_ID__    — Node number (1, 2, 3, ...)
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
exec > >(tee /var/log/nelu-setup.log) 2>&1

S3_BUCKET="__S3_BUCKET__"
NODE_ID="__NODE_ID__"

echo "════════════════════════════════════════════════════════"
echo "  NELU Training Node ${NODE_ID}"
echo "  $(date -u)"
echo "════════════════════════════════════════════════════════"

# ── 1. System info ──────────────────────────────────────────────

INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id || echo "unknown")
INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type || echo "unknown")
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"

if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

# ── 2. Install dependencies ────────────────────────────────────

echo ""
echo "── Installing dependencies ──"

# Activate conda (Deep Learning AMI has it pre-installed)
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || \
    source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true

# Create or activate environment
if ! conda env list | grep -q "^nelu "; then
    conda create -n nelu python=3.11 -y
fi
conda activate nelu

pip install --quiet \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install --quiet \
    timm==1.0.11 wandb tqdm scikit-learn scipy ninja matplotlib pyyaml autoattack

# Apex for LAMB optimizer (DeiT III)
pip install --quiet packaging
pip install --quiet --no-build-isolation \
    git+https://github.com/NVIDIA/apex.git 2>/dev/null || \
    echo "WARNING: apex build failed — will use AdamW fallback"

# ── 3. Download code ───────────────────────────────────────────

echo ""
echo "── Downloading code ──"

WORKSPACE="/workspace"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

aws s3 cp "${S3_BUCKET}/code/nelu-code.tar.gz" /tmp/nelu-code.tar.gz --quiet
tar xzf /tmp/nelu-code.tar.gz -C "$WORKSPACE"
rm /tmp/nelu-code.tar.gz

# Download this node's job file
aws s3 cp "${S3_BUCKET}/jobs/jobs_node${NODE_ID}.txt" \
    "${WORKSPACE}/scripts/jobs_node${NODE_ID}.txt" --quiet

# Clone upstream training repos and apply NELU patches
cd $HOME
git clone https://github.com/facebookresearch/ConvNeXt.git convnext-train
cd convnext-train && git apply ${WORKSPACE}/patches/convnext-train.patch && cd ..

git clone https://github.com/facebookresearch/deit.git deit-train
cd deit-train && git apply ${WORKSPACE}/patches/deit-train.patch && cd ..

# ── 4. Download data from S3 ──────────────────────────────────

echo ""
echo "── Downloading datasets ──"

mkdir -p /data

# Always need ImageNet for the main experiments
if [ ! -d /data/imagenet/train ]; then
    echo "  Syncing ImageNet from S3 (this takes a while)..."
    aws s3 sync "${S3_BUCKET}/data/imagenet/" /data/imagenet/ --quiet || \
        echo "  WARNING: ImageNet sync failed or not available"
fi

# CIFAR-100 (small, always grab it)
if [ ! -d /data/cifar-100-python ]; then
    aws s3 sync "${S3_BUCKET}/data/cifar-100-python/" /data/cifar-100-python/ --quiet || true
fi

# Symlink data into the repo
ln -snf /data "${WORKSPACE}/data"

# ── 5. Build CUDA kernel ──────────────────────────────────────

echo ""
echo "── Building CUDA kernel ──"

cd "$WORKSPACE"
python -c "from nelu.cuda_kernel import nelu_cuda; print('NELU CUDA kernel OK')" 2>/dev/null || \
    echo "WARNING: CUDA kernel build failed — will use Python fallback"

# ── 6. Start spot interruption handler ─────────────────────────

echo ""
echo "── Starting spot interrupt handler ──"

bash "${WORKSPACE}/scripts/infra/spot_interrupt_handler.sh" &
HANDLER_PID=$!
echo "  Handler PID: $HANDLER_PID"

# ── 7. Start training ─────────────────────────────────────────

echo ""
echo "── Starting training ──"

export S3_BUCKET
cd "$WORKSPACE"

bash scripts/run_all.sh "scripts/jobs_node${NODE_ID}.txt" 2>&1 | \
    tee "/workspace/train_node${NODE_ID}.log"

# Upload the full log
aws s3 cp "/workspace/train_node${NODE_ID}.log" \
    "${S3_BUCKET}/logs/node${NODE_ID}_${INSTANCE_ID}.log" --quiet 2>/dev/null || true

# ── 8. Shutdown ────────────────────────────────────────────────
# run_all.sh handles shutdown, but if it didn't for some reason:
echo ""
echo "Training complete. Instance will shut down."
