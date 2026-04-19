#!/bin/bash
# User data for instances booted from the NELU training EBS snapshot.
# The snapshot at /data already contains: conda env, datasets, repos, compiled kernels.
# This script only needs to: pull latest code, set env vars, start training.

set -euo pipefail
exec > >(tee /var/log/nelu-userdata.log) 2>&1

S3_BUCKET="__S3_BUCKET__"
NODE_ID="__NODE_ID__"
export WANDB_API_KEY="__WANDB_API_KEY__"
ORCH_RUN_ID="__ORCH_RUN_ID__"

echo "NELU training instance starting — Node ${NODE_ID}"
echo "$(date -u)"

# ── 1. Mount data volume (from EBS snapshot) ──
# The EBS data volume was attached as /dev/sdf but may appear as
# /dev/nvmeXn1 on NVMe instances.  Prefer EBS over instance-store.
DATA_DEV=""
for dev in /dev/sdf /dev/xvdf; do
    [ -b "$dev" ] && DATA_DEV="$dev" && break
done
if [ -z "$DATA_DEV" ]; then
    for dev in /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1; do
        if [ -b "$dev" ] && nvme id-ctrl "$dev" 2>/dev/null | grep -qi "Amazon Elastic"; then
            DATA_DEV="$dev"
            break
        fi
    done
fi
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

# ── 3. Refresh code ──
WORKSPACE="/data/repos/NELU"
mkdir -p "$WORKSPACE"

if aws s3 cp "${S3_BUCKET}/code/nelu-code.tar.gz" /tmp/nelu-code.tar.gz --quiet 2>/dev/null; then
    TMP_CODE_DIR="$(mktemp -d)"
    tar xzf /tmp/nelu-code.tar.gz -C "$TMP_CODE_DIR"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete --exclude '.git/' "$TMP_CODE_DIR"/ "$WORKSPACE"/
    else
        find "$WORKSPACE" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
        tar xzf /tmp/nelu-code.tar.gz -C "$WORKSPACE"
    fi
    rm -rf "$TMP_CODE_DIR" /tmp/nelu-code.tar.gz
    echo "Workspace updated from ${S3_BUCKET}/code/nelu-code.tar.gz"
elif [ -d "$WORKSPACE/.git" ]; then
    cd "$WORKSPACE"
    git pull origin main --ff-only 2>/dev/null || true
else
    git clone https://github.com/medicalissue/NELU.git "$WORKSPACE"
fi

cd "$WORKSPACE"

# ── 4. Download job file from S3 ──
aws s3 cp "${S3_BUCKET}/jobs/jobs_node${NODE_ID}.txt" \
    "${WORKSPACE}/scripts/jobs_node${NODE_ID}.txt" --quiet 2>/dev/null || true

# ── 5. Set environment ──
export S3_BUCKET
export ORCH_RUN_ID
# No upstream repos needed — all training via train_imagenet_timm.py
export RESULTS_DIR="/data/results"
export TORCH_EXTENSIONS_DIR="/data/cache/torch_extensions"
mkdir -p "$RESULTS_DIR" /data/cache /data/logs

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
set +e
bash scripts/run_all.sh "scripts/jobs_node${NODE_ID}.txt" 2>&1 | \
    tee "/data/logs/train_node${NODE_ID}.log"
RUN_ALL_EXIT=${PIPESTATUS[0]}
set -e

# Upload log
aws s3 cp "/data/logs/train_node${NODE_ID}.log" \
    "${S3_BUCKET}/logs/node${NODE_ID}.log" --quiet 2>/dev/null || true

if [ $RUN_ALL_EXIT -ne 0 ]; then
    echo "run_all.sh exited with code $RUN_ALL_EXIT"
    if [ -n "${ORCH_RUN_ID:-}" ]; then
        FAIL_FILE="/data/logs/node${NODE_ID}.failed.txt"
        {
            echo "node_id=${NODE_ID}"
            echo "run_id=${ORCH_RUN_ID}"
            echo "exit_code=${RUN_ALL_EXIT}"
            echo "timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
            echo "instance_id=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || true)"
        } > "$FAIL_FILE"
        aws s3 cp "$FAIL_FILE" \
            "${S3_BUCKET}/orchestrator/${ORCH_RUN_ID}/node${NODE_ID}.FAILED" \
            --quiet 2>/dev/null || true
    fi
    exit $RUN_ALL_EXIT
fi

# ── 8. Shutdown ──
echo "All jobs complete. Shutting down."
sudo shutdown -h now
