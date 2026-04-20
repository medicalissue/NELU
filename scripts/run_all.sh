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
S3_BUCKET="${S3_BUCKET:-s3://nelu-datasets}"
SHUTDOWN_WHEN_DONE="${SHUTDOWN_WHEN_DONE:-true}"
INTERRUPTED_EXIT_CODE="${INTERRUPTED_EXIT_CODE:-90}"
SPOT_INTERRUPT_MARKER="${SPOT_INTERRUPT_MARKER:-/tmp/nelu_spot_interrupted}"

# ── GPU slot pool for single-GPU jobs ──────────────────────────
# Runs up to NUM_GPUS single-GPU jobs in parallel using a FIFO semaphore.
# Multi-GPU jobs (imagenet, ablation) drain the pool first, then run alone.

NUM_GPUS=${NUM_GPUS:-8}
SLOT_FIFO="/tmp/nelu_gpu_slots.$$"
SLOT_PIDS=()
SLOT_TAGS=()
SLOT_STATUS_DIR="/tmp/nelu_slot_status.$$"

slot_init() {
    rm -f "$SLOT_FIFO"
    rm -rf "$SLOT_STATUS_DIR"
    mkfifo "$SLOT_FIFO"
    mkdir -p "$SLOT_STATUS_DIR"
    exec 9<>"$SLOT_FIFO"
    for g in $(seq 0 $((NUM_GPUS - 1))); do
        echo $g >&9
    done
    SLOT_PIDS=()
    SLOT_TAGS=()
}

slot_run() {
    local tag="$1"; shift
    local g
    read -u 9 g
    local logf="logs/${tag}.log"
    local ind_cache="/tmp/inductor_cache_gpu${g}"
    local triton_cache="/tmp/triton_cache_gpu${g}"
    local status_file="${SLOT_STATUS_DIR}/${tag}.status"
    mkdir -p "$ind_cache" "$triton_cache"
    (
        local rc=0
        echo "[$(date +%H:%M:%S)] [gpu $g] start $tag"
        TORCHINDUCTOR_CACHE_DIR="$ind_cache" \
        TRITON_CACHE_DIR="$triton_cache" \
        CUDA_VISIBLE_DEVICES=$g \
            bash "${REPO_ROOT}/scripts/run_single.sh" "$@" > "$logf" 2>&1 || rc=$?
        printf '%s\n' "$rc" > "$status_file"
        if [ $rc -eq 0 ]; then
            echo "[$(date +%H:%M:%S)] [gpu $g] done  $tag"
        else
            echo "[$(date +%H:%M:%S)] [gpu $g] FAIL  $tag (rc=$rc, see $logf)"
        fi
        echo $g >&9
    ) &
    SLOT_PIDS+=($!)
    SLOT_TAGS+=("$tag")
}

slot_drain() {
    local i pid tag wait_rc status_rc
    for i in "${!SLOT_PIDS[@]}"; do
        pid="${SLOT_PIDS[$i]}"
        tag="${SLOT_TAGS[$i]}"

        if wait "$pid"; then
            wait_rc=0
        else
            wait_rc=$?
        fi

        status_rc="$wait_rc"
        if [ -f "${SLOT_STATUS_DIR}/${tag}.status" ]; then
            status_rc="$(cat "${SLOT_STATUS_DIR}/${tag}.status")"
        fi

        if [ "$status_rc" -ne 0 ]; then
            if [ "$status_rc" -eq "$INTERRUPTED_EXIT_CODE" ] || [ -f "$SPOT_INTERRUPT_MARKER" ]; then
                INTERRUPTED=1
            else
                FAILED=$((FAILED + 1))
            fi
        fi
    done
    SLOT_PIDS=()
    SLOT_TAGS=()
}

slot_cleanup() {
    exec 9>&- 2>/dev/null || true
    rm -f "$SLOT_FIFO"
    rm -rf "$SLOT_STATUS_DIR"
}

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
rm -f "$SPOT_INTERRUPT_MARKER"

# ── Background S3 sync — upload checkpoints every 10 minutes ──
# This ensures that if spot interruption handler fails or checkpoint
# is large, we still have recent checkpoints on S3.
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
S3_SYNC_INTERVAL="${S3_SYNC_INTERVAL:-600}"  # 10 minutes
SYNC_PID=""
if aws sts get-caller-identity --output text >/dev/null 2>&1; then
    (
        while true; do
            sleep "$S3_SYNC_INTERVAL"
            aws s3 sync "${RESULTS_DIR}" "${S3_BUCKET}/results/" \
                --exclude "*.log" --exclude "wandb/*" --quiet 2>/dev/null || true
        done
    ) &
    SYNC_PID=$!
    echo "[$(date +%H:%M:%S)] [s3-sync] background loop started (PID $SYNC_PID, every ${S3_SYNC_INTERVAL}s)"
fi
# Kill the sync loop when run_all exits
trap "[ -n \"$SYNC_PID\" ] && kill $SYNC_PID 2>/dev/null; slot_cleanup" EXIT

JOB_NUM=0
FAILED=0
SKIPPED=0
INTERRUPTED=0

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
        slot_run "$RUN_NAME" "$PHASE" "$MODEL" "$ACT" "${EXTRA_ARGS[@]}"
    else
        # Multi-GPU job (imagenet, ablation) — drain single-GPU jobs first
        slot_drain
        if [ "$INTERRUPTED" -eq 1 ]; then
            echo "  Spot interruption detected while draining queued jobs. Exiting for replacement."
            exit "$INTERRUPTED_EXIT_CODE"
        fi
        if bash "${REPO_ROOT}/scripts/run_single.sh" "$PHASE" "$MODEL" "$ACT" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; then
            echo "  Job $JOB_NUM complete."
        else
            rc=$?
            if [ "$rc" -eq "$INTERRUPTED_EXIT_CODE" ] || [ -f "$SPOT_INTERRUPT_MARKER" ]; then
                echo "  Spot interruption detected. Exiting for replacement."
                exit "$INTERRUPTED_EXIT_CODE"
            fi
            echo "  Job $JOB_NUM FAILED (exit code $rc)."
            FAILED=$((FAILED + 1))
        fi
    fi

done < "$JOB_FILE"

# Drain any remaining single-GPU jobs
slot_drain
if [ "$INTERRUPTED" -eq 1 ]; then
    echo ""
    echo "Spot interruption detected. Exiting for replacement."
    exit "$INTERRUPTED_EXIT_CODE"
fi

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
    exit 1
fi
