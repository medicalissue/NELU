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
INTERRUPTED_EXIT_CODE="${INTERRUPTED_EXIT_CODE:-90}"
SPOT_INTERRUPT_MARKER="${SPOT_INTERRUPT_MARKER:-/tmp/nelu_spot_interrupted}"
EXPECTED_GPUS="${EXPECTED_GPUS:-8}"
MAX_HEALTH_ATTEMPTS="${MAX_HEALTH_ATTEMPTS:-3}"

echo "NELU training instance starting — Node ${NODE_ID}"
echo "$(date -u)"

# ── 1. Mount data volume (from EBS snapshot) ──
# Prefer explicit EBS identifiers; Nitro instance-store NVMe devices can
# appear ahead of the attached data EBS volume and are not mountable here.
DATA_DEV=""
ROOT_DEV=$(lsblk -no PKNAME "$(findmnt -n -o SOURCE /)" 2>/dev/null || echo "")
declare -A seen_devs=()
candidate_devs=()

for dev in /dev/sdf /dev/xvdf /dev/disk/by-id/nvme-Amazon_Elastic_Block_Store_vol* /dev/nvme*n1; do
    [ -e "$dev" ] || continue
    resolved=$(readlink -f "$dev")
    [ -b "$resolved" ] || continue
    [ -n "${seen_devs[$resolved]:-}" ] && continue
    seen_devs[$resolved]=1
    candidate_devs+=("$resolved")
done

mkdir -p /data
for dev in "${candidate_devs[@]}"; do
    devbase=$(basename "$dev")
    [ "$devbase" = "$ROOT_DEV" ] && continue
    [[ "$devbase" == "${ROOT_DEV}"* ]] && continue
    findmnt -rn -S "$dev" >/dev/null 2>&1 && continue
    fstype=$(lsblk -no FSTYPE "$dev" 2>/dev/null | head -n1 || true)
    [ "$fstype" = "LVM2_member" ] && continue
    case "$fstype" in
        ext4|xfs)
            if mount "$dev" /data 2>/dev/null; then
                DATA_DEV="$dev"
                break
            fi
            ;;
    esac
done

if [ -z "$DATA_DEV" ]; then
    echo "ERROR: Failed to locate and mount the EBS data volume on /data"
    lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT
    exit 1
fi

echo "Mounted $DATA_DEV on /data"

# ── 2. Activate conda env ──
source /data/env/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || \
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate /data/env/nelu 2>/dev/null || conda activate nelu 2>/dev/null || true
if [ -d /data/env/nelu/bin ]; then
    export PATH="/data/env/nelu/bin:${PATH}"
fi

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

# ── 6. Pre-flight GPU health check ──
# p5 spot pool occasionally hands out hosts with a dead NVLink / defective
# GPU (NCCL reports "P2P is disabled between NVLINK connected GPUs …
# probably due to a hardware issue"). Detect before DDP init so we can
# self-terminate and let the orchestrator grab a different host.
#
# We skip the FAILED marker so launch_spot.sh sees `terminated` without
# FAILED and calls launch_node_instance for this slot. A counter at
# s3://…/orchestrator/<run>/node<N>.health_attempts caps retries so a
# truly-broken config doesn't loop forever.

HEALTH_ATTEMPT_KEY="${S3_BUCKET}/orchestrator/${ORCH_RUN_ID}/node${NODE_ID}.health_attempts"

check_gpu_health() {
    local gpu_count
    gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$gpu_count" -ne "$EXPECTED_GPUS" ]; then
        echo "HEALTH: expected $EXPECTED_GPUS GPUs, found $gpu_count"
        nvidia-smi -L || true
        return 1
    fi

    # P2P matrix: legitimate rows start with "GPU<n>" and should contain
    # only X / OK. "NS" anywhere means that GPU's P2P link is dead.
    # grep -w avoids matching CNS/GNS/TNS legend tokens.
    local p2p_out
    p2p_out=$(nvidia-smi topo -p2p r 2>/dev/null || true)
    if echo "$p2p_out" | grep -E '^[[:space:]]*GPU[0-9]+[[:space:]]' | grep -qw 'NS'; then
        echo "HEALTH: P2P matrix has NS entries (broken NVLink/NVSwitch):"
        echo "$p2p_out" | grep -E '^[[:space:]]*GPU[0-9]+[[:space:]]'
        return 1
    fi

    # XID / NVRM errors in kernel log indicate driver-level GPU faults.
    if sudo dmesg -T 2>/dev/null | grep -iE 'Xid|NVRM: .*error' | tail -n 5 | grep -q .; then
        echo "HEALTH: XID / NVRM errors in dmesg:"
        sudo dmesg -T | grep -iE 'Xid|NVRM: .*error' | tail -n 5
        return 1
    fi

    echo "HEALTH: ok (${gpu_count} GPUs, P2P clean, no XID)"
    return 0
}

HEALTH_COUNT=0
if aws s3 cp "$HEALTH_ATTEMPT_KEY" /tmp/health_attempts --quiet 2>/dev/null; then
    HEALTH_COUNT=$(tr -d '\n' </tmp/health_attempts)
fi
HEALTH_COUNT=${HEALTH_COUNT:-0}

set +e
check_gpu_health
HEALTH_RC=$?
set -e

if [ $HEALTH_RC -ne 0 ]; then
    HEALTH_COUNT=$((HEALTH_COUNT + 1))
    echo "HEALTH: failure #${HEALTH_COUNT}/${MAX_HEALTH_ATTEMPTS}"

    INST_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || true)
    if [ -n "$INST_ID" ]; then
        aws ec2 create-tags --resources "$INST_ID" \
            --tags "Key=Health,Value=bad_hardware" 2>/dev/null || true
    fi

    if [ "$HEALTH_COUNT" -ge "$MAX_HEALTH_ATTEMPTS" ]; then
        echo "HEALTH: max retries exhausted — writing FAILED marker"
        FAIL_FILE="/tmp/node${NODE_ID}.failed.txt"
        {
            echo "node_id=${NODE_ID}"
            echo "run_id=${ORCH_RUN_ID}"
            echo "exit_code=bad_hardware"
            echo "timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
            echo "instance_id=${INST_ID}"
            echo "health_attempts=${HEALTH_COUNT}"
        } > "$FAIL_FILE"
        aws s3 cp "$FAIL_FILE" \
            "${S3_BUCKET}/orchestrator/${ORCH_RUN_ID}/node${NODE_ID}.FAILED" \
            --quiet 2>/dev/null || true
    else
        printf '%s' "$HEALTH_COUNT" > /tmp/health_attempts
        aws s3 cp /tmp/health_attempts "$HEALTH_ATTEMPT_KEY" --quiet 2>/dev/null || true
        echo "HEALTH: self-terminating; orchestrator will relaunch this slot on a different host"
    fi

    sudo shutdown -h now
    exit 0
fi

# Healthy: drop any stale attempt counter from prior launches.
aws s3 rm "$HEALTH_ATTEMPT_KEY" --quiet 2>/dev/null || true

# ── 7. Start spot interrupt handler ──
bash "${WORKSPACE}/scripts/infra/spot_interrupt_handler.sh" &
echo "Spot handler PID: $!"

# ── 8. Start training ──
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
    if [ "$RUN_ALL_EXIT" -eq "$INTERRUPTED_EXIT_CODE" ] || [ -f "$SPOT_INTERRUPT_MARKER" ]; then
        echo "Spot interruption detected — skipping FAILED marker so the orchestrator can relaunch."
        exit 0
    fi
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

# ── 9. Shutdown ──
echo "All jobs complete. Shutting down."
sudo shutdown -h now
