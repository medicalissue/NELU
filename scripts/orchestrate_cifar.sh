#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# CIFAR-100 worker fanout.
#
# Each CIFAR job uses a single GPU, so a multi-GPU VM should pull
# ``ngpus`` independent jobs at once from the shared S3 queue. This
# script simply spawns one ``orchestrate_cifar_slot.sh`` process per
# visible GPU, each pinned to its own ``CUDA_VISIBLE_DEVICES``, and
# waits for all of them. The slot runners race against the same lease
# queue: whichever GPU finishes first pops the next job. Order is
# naturally "first-idle-first-serve".
#
# Shared-queue coordination lives entirely in the slot script; see
# scripts/orchestrate_cifar_slot.sh for the lease-claim / heartbeat /
# preempt logic. Environment variables propagate to every slot.
#
# If no GPU is visible (CPU debugging), fall back to a single slot.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

log() { printf '[orchestrate-cifar %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLOT_RUNNER="$SCRIPT_DIR/orchestrate_cifar_slot.sh"

# Detect GPUs. Honor an explicit override for debugging / constrained
# testing: ``NUM_CIFAR_SLOTS=2 bash orchestrate_cifar.sh`` forces two
# slots regardless of nvidia-smi.
if [[ -n "${NUM_CIFAR_SLOTS:-}" ]]; then
    ngpus="$NUM_CIFAR_SLOTS"
elif command -v nvidia-smi >/dev/null; then
    ngpus=$(nvidia-smi -L | wc -l | tr -d ' ')
else
    ngpus=1
fi
if (( ngpus < 1 )); then ngpus=1; fi

log "spawning $ngpus slot runner(s)"

pids=()
for (( gpu=0; gpu<ngpus; gpu++ )); do
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export GATE_NORM_GPU_SLOT="$gpu"
        exec bash "$SLOT_RUNNER"
    ) &
    pids+=("$!")
    log "  slot $gpu (CUDA_VISIBLE_DEVICES=$gpu) pid=${pids[-1]}"
done

# Propagate SIGTERM/SIGINT to every child. AWS spot preempt → the
# bootstrap wrapper SIGTERMs us, we forward to each slot which then
# forwards to its trainer via the per-slot preempt watcher.
cleanup() {
    log "received signal — forwarding to all slots"
    for pid in "${pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup TERM INT

# Wait for every slot. Exit code = OR of slot rcs (non-zero if any failed).
rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=$?
done

log "all slots exited (rc=$rc)"
exit "$rc"
