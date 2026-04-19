#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Master experiment orchestrator.
#
#  Reads a job queue file and runs each experiment sequentially.
#  Skips jobs that already have results on S3. After all jobs
#  complete, shuts down the instance to avoid idle charges.
#
#  Usage:
#    ./scripts/run_all.sh <job_queue_file>
#    ./scripts/run_all.sh scripts/jobs_node1.txt
#
#  Job queue format (one line per job, # comments ignored):
#    <phase> <model> <act> [extra_args...]
#
#  Example:
#    imagenet convnext_tiny gelu
#    imagenet convnext_tiny nelu
#    cifar100 resnet20 nelu --seed 42
#    ablation convnext_tiny nelu --gamma_init 0.01
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# wandb rate-limit mitigation: when 8 GPU jobs init simultaneously,
# the wandb service port-file poll and init handshake can time out.
export WANDB__SERVICE_WAIT=3600
export WANDB_INIT_TIMEOUT=600
export WANDB_DISABLE_CODE=true
export WANDB_HTTP_TIMEOUT=120

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
S3_BUCKET="${S3_BUCKET:-s3://nelu-datasets/v2}"
SHUTDOWN_WHEN_DONE="${SHUTDOWN_WHEN_DONE:-true}"

# ── GPU slot pool for single-GPU jobs ──────────────────────────
# Runs up to NUM_GPUS single-GPU jobs in parallel using a FIFO semaphore.
# Multi-GPU jobs (imagenet, ablation) drain the pool first, then run alone.

NUM_GPUS=${NUM_GPUS:-8}
SLOT_FIFO="/tmp/nelu_gpu_slots.$$"
SLOT_PIDS=()

slot_init() {
    rm -f "$SLOT_FIFO"
    mkfifo "$SLOT_FIFO"
    exec 9<>"$SLOT_FIFO"
    for g in $(seq 0 $((NUM_GPUS - 1))); do
        echo $g >&9
    done
    SLOT_PIDS=()
}

slot_run() {
    local tag="$1"; shift
    local g
    read -u 9 g
    local logf="logs/${tag}.log"
    local ind_cache="/tmp/inductor_cache_gpu${g}"
    local triton_cache="/tmp/triton_cache_gpu${g}"
    mkdir -p "$ind_cache" "$triton_cache"
    (
        echo "[$(date +%H:%M:%S)] [gpu $g] start $tag"
        TORCHINDUCTOR_CACHE_DIR="$ind_cache" \
        TRITON_CACHE_DIR="$triton_cache" \
        CUDA_VISIBLE_DEVICES=$g eval "$@" > "$logf" 2>&1
        local rc=$?
        if [ $rc -eq 0 ]; then
            echo "[$(date +%H:%M:%S)] [gpu $g] done  $tag"
        else
            echo "[$(date +%H:%M:%S)] [gpu $g] FAIL  $tag (rc=$rc, see $logf)"
        fi
        echo $g >&9
    ) &
    SLOT_PIDS+=($!)
}

slot_drain() {
    if [ ${#SLOT_PIDS[@]} -gt 0 ]; then
        wait "${SLOT_PIDS[@]}" 2>/dev/null || true
    fi
    SLOT_PIDS=()
}

slot_cleanup() {
    exec 9>&- 2>/dev/null || true
    rm -f "$SLOT_FIFO"
}
trap slot_cleanup EXIT

# ── Parse arguments ─────────────────────────────────────────────

if [ $# -lt 1 ]; then
    echo "Usage: $0 <job_queue_file>"
    echo ""
    echo "  Runs all experiments listed in the job queue file."
    echo "  Set SHUTDOWN_WHEN_DONE=false to skip auto-shutdown."
    exit 1
fi

JOB_FILE="$1"

if [ ! -f "$JOB_FILE" ]; then
    echo "ERROR: Job queue file not found: $JOB_FILE"
    exit 1
fi

# ── Count jobs ──────────────────────────────────────────────────

TOTAL=$(grep -v '^\s*#' "$JOB_FILE" | grep -v '^\s*$' | wc -l | tr -d ' ')
echo "═══════════════════════════════════════════════════════════"
echo "  NELU Experiment Orchestrator"
echo "═══════════════════════════════════════════════════════════"
echo "  Job file:    $JOB_FILE"
echo "  Total jobs:  $TOTAL"
echo "  S3 bucket:   $S3_BUCKET"
echo "  Auto-shutdown: $SHUTDOWN_WHEN_DONE"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Run each job ────────────────────────────────────────────────

mkdir -p logs
slot_init

JOB_NUM=0
FAILED=0
SKIPPED=0

while IFS= read -r line; do
    # Skip comments and blank lines
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue

    JOB_NUM=$((JOB_NUM + 1))
    # shellcheck disable=SC2086
    set -- $line
    PHASE="$1"
    MODEL="$2"
    ACT="$3"
    shift 3
    EXTRA_ARGS=("$@")

    echo ""
    echo "───────────────────────────────────────────────────────"
    echo "  Job $JOB_NUM / $TOTAL: $PHASE $MODEL $ACT ${EXTRA_ARGS[*]+${EXTRA_ARGS[*]}}"
    echo "───────────────────────────────────────────────────────"

    # Check if already done on S3
    RUN_NAME="${PHASE}_${MODEL}_${ACT}"
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        for arg in "${EXTRA_ARGS[@]}"; do
            clean=$(echo "$arg" | sed 's/^--//; s/=/_/g')
            RUN_NAME="${RUN_NAME}_${clean}"
        done
    fi

    if aws s3 ls "${S3_BUCKET}/results/${RUN_NAME}/DONE" >/dev/null 2>&1; then
        echo "  Already complete on S3. Skipping."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Decide single-GPU vs multi-GPU based on phase
    if [ "$PHASE" = "cifar100" ] || [ "$PHASE" = "eval" ]; then
        # Single-GPU job — dispatch via slot pool
        slot_run "$RUN_NAME" "bash '${REPO_ROOT}/scripts/run_single.sh' '$PHASE' '$MODEL' '$ACT' ${EXTRA_ARGS[*]+${EXTRA_ARGS[*]}}"
    else
        # Multi-GPU job (imagenet, ablation) — drain single-GPU jobs first
        slot_drain
        if bash "${REPO_ROOT}/scripts/run_single.sh" "$PHASE" "$MODEL" "$ACT" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; then
            echo "  Job $JOB_NUM complete."
        else
            echo "  Job $JOB_NUM FAILED (exit code $?)."
            FAILED=$((FAILED + 1))
        fi
    fi

done < "$JOB_FILE"

# Drain any remaining single-GPU jobs
slot_drain

# ── Summary ─────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All jobs processed."
echo "  Total: $TOTAL  |  Skipped: $SKIPPED  |  Failed: $FAILED"
echo "═══════════════════════════════════════════════════════════"

# ── Shutdown ────────────────────────────────────────────────────

if [ "$SHUTDOWN_WHEN_DONE" = "true" ] && [ $FAILED -eq 0 ]; then
    echo ""
    echo "All jobs succeeded. Shutting down in 60 seconds..."
    echo "(Cancel with: sudo shutdown -c)"
    sudo shutdown -h +1
elif [ $FAILED -gt 0 ]; then
    echo ""
    echo "Some jobs failed. NOT shutting down — investigate manually."
fi
