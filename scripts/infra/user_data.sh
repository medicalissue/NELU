#!/bin/bash
# User data for instances booted from the NELU training EBS snapshot.
# The snapshot at /data already contains: conda env, datasets, repos, compiled kernels.
# This script only needs to: pull latest code, set env vars, start training.

set -euo pipefail
exec > >(tee /var/log/nelu-userdata.log) 2>&1

S3_BUCKET="__S3_BUCKET__"
NODE_ID="__NODE_ID__"
export WANDB_API_KEY="__WANDB_API_KEY__"

echo "NELU training instance starting — Node ${NODE_ID}"
echo "$(date -u)"

# ── 1. Mount data volume (from EBS snapshot) ──
DATA_DEV=""
for dev in /dev/nvme1n1 /dev/xvdf /dev/sdf; do
    [ -b "$dev" ] && DATA_DEV="$dev" && break
done
if [ -n "$DATA_DEV" ]; then
    mkdir -p /data
    mount "$DATA_DEV" /data 2>/dev/null || true
    echo "Mounted $DATA_DEV on /data"
fi

# ── 2. Activate conda env ──
source /data/env/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || \
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate /data/env/nelu 2>/dev/null || conda activate nelu 2>/dev/null || true

# ── 3. Pull latest code ──
WORKSPACE="/data/repos/NELU"
cd "$WORKSPACE"
git pull origin main --ff-only 2>/dev/null || true

cd "$WORKSPACE"

# ── 4. Download job file from S3 ──
aws s3 cp "${S3_BUCKET}/jobs/jobs_node${NODE_ID}.txt" \
    "${WORKSPACE}/scripts/jobs_node${NODE_ID}.txt" --quiet 2>/dev/null || true

# ── 5. Set environment ──
export S3_BUCKET
# No upstream repos needed — all training via train_imagenet_timm.py
export RESULTS_DIR="/data/results"
export TORCH_EXTENSIONS_DIR="/data/cache/torch_extensions"
mkdir -p "$RESULTS_DIR" /data/cache

if [ -z "${WANDB_API_KEY:-}" ]; then
    export ENABLE_WANDB=0
    echo "WANDB_API_KEY not set — wandb disabled"
else
    export ENABLE_WANDB=1
fi

# wandb rate-limit mitigation
export WANDB__SERVICE_WAIT=3600
export WANDB_INIT_TIMEOUT=600
export WANDB_DISABLE_CODE=true
export WANDB_HTTP_TIMEOUT=120

# ── 6. Start spot interrupt handler ──
bash "${WORKSPACE}/scripts/infra/spot_interrupt_handler.sh" &
echo "Spot handler PID: $!"

# ── 7. Start training ──
echo "Starting training — Node ${NODE_ID}"
cd "$WORKSPACE"
bash scripts/run_all.sh "scripts/jobs_node${NODE_ID}.txt" 2>&1 | \
    tee "/data/logs/train_node${NODE_ID}.log"

# Upload log
aws s3 cp "/data/logs/train_node${NODE_ID}.log" \
    "${S3_BUCKET}/logs/node${NODE_ID}.log" --quiet 2>/dev/null || true

# ── 8. Shutdown ──
echo "All jobs complete. Shutting down."
sudo shutdown -h now
