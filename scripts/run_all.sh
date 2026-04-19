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

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
S3_BUCKET="${S3_BUCKET:-s3://nelu-experiments}"
SHUTDOWN_WHEN_DONE="${SHUTDOWN_WHEN_DONE:-true}"

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
    echo "  Job $JOB_NUM / $TOTAL: $PHASE $MODEL $ACT ${EXTRA_ARGS[*]:-}"
    echo "───────────────────────────────────────────────────────"

    # Check if already done on S3
    RUN_NAME="${PHASE}_${MODEL}_${ACT}"
    for arg in "${EXTRA_ARGS[@]}"; do
        clean=$(echo "$arg" | sed 's/^--//; s/=/_/g')
        RUN_NAME="${RUN_NAME}_${clean}"
    done

    if aws s3 ls "${S3_BUCKET}/results/${RUN_NAME}/DONE" >/dev/null 2>&1; then
        echo "  Already complete on S3. Skipping."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Run the experiment
    if bash "${REPO_ROOT}/scripts/run_single.sh" "$PHASE" "$MODEL" "$ACT" "${EXTRA_ARGS[@]}"; then
        echo "  Job $JOB_NUM complete."
    else
        echo "  Job $JOB_NUM FAILED (exit code $?)."
        FAILED=$((FAILED + 1))
    fi

done < "$JOB_FILE"

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
