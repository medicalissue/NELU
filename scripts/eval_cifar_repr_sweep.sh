#!/bin/bash
# Fan out per-checkpoint repr-quality eval across N GPU slots on one box.
#
# Usage:
#   eval_cifar_repr_sweep.sh CKPT_DIR OUTDIR [DATA_ROOT] [SLOTS]
#
# CKPT_DIR holds <model>-<act>-s<seed>/checkpoint.pt subdirs (matches
# scripts/orchestrate_cifar.sh layout). For each ckpt found, schedules
# scripts/eval_cifar_repr.sh into one of SLOTS GPUs (default: number of
# CUDA devices). Results land in OUTDIR/<run-name>/<probe>.json.

set -euo pipefail

CKPT_DIR=${1:?usage: $0 CKPT_DIR OUTDIR [DATA_ROOT] [SLOTS]}
OUTDIR=${2:?}
DATA_ROOT=${3:-/data}
SLOTS=${4:-$(python -c 'import torch; print(torch.cuda.device_count() or 1)')}

mkdir -p "$OUTDIR"
HERE="$(cd "$(dirname "$0")" && pwd)"

# Build job list: one line per ckpt directory.
mapfile -t JOBS < <(
    find "$CKPT_DIR" -mindepth 1 -maxdepth 1 -type d \
        | grep -E '/[a-z0-9_]+-(relu|gelu|silu|nelu|nilu)-s[0-9]+$' \
        | sort
)
echo "[sweep] $((${#JOBS[@]})) checkpoints across $SLOTS slots"

slot_pids=()
slot_log=()
for ((i = 0; i < SLOTS; i++)); do
    slot_pids[i]=0
    slot_log[i]="$OUTDIR/.slot${i}.log"
done

dispatch_to_slot() {
    local slot=$1 job=$2
    local name; name=$(basename "$job")
    local model=${name%%-*}; local rest=${name#*-}
    local act=${rest%%-*}; local seed=${rest##*-}
    CUDA_VISIBLE_DEVICES=$slot \
        bash "$HERE/eval_cifar_repr.sh" \
        "$model" "$act" "$job/checkpoint.pt" "$OUTDIR/$name" "$DATA_ROOT" \
        >> "${slot_log[slot]}" 2>&1 &
    slot_pids[slot]=$!
    echo "[slot $slot] $name (pid ${slot_pids[slot]})"
}

for job in "${JOBS[@]}"; do
    placed=0
    while (( placed == 0 )); do
        for ((i = 0; i < SLOTS; i++)); do
            if (( slot_pids[i] == 0 )) || ! kill -0 "${slot_pids[i]}" 2>/dev/null; then
                dispatch_to_slot "$i" "$job"
                placed=1
                break
            fi
        done
        (( placed == 0 )) && sleep 5
    done
done

for pid in "${slot_pids[@]}"; do
    (( pid > 0 )) && wait "$pid" || true
done

# Final mass sync as a safety net for any results the per-job upload
# missed (e.g. a probe that errored mid-checkpoint).
if [[ -n "${S3_RESULTS_PREFIX:-}" ]]; then
    aws s3 sync "$OUTDIR" "$S3_RESULTS_PREFIX/" --no-progress \
        --exclude '.shared_logits/*' --exclude '.slot*.log' \
        && echo "[sweep] final S3 sync done."
fi
echo "[sweep] done."
