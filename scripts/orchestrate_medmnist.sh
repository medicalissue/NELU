#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# MedMNIST v2 worker fanout.
#
# Mirrors scripts/orchestrate_cifar.sh: spawn one slot runner per visible
# GPU, each pinned to its own CUDA_VISIBLE_DEVICES, racing the same S3
# lease queue. MedMNIST jobs are single-GPU, so a 4-GPU VM pulls 4 jobs
# at once.
#
# Difference from the CIFAR fanout: NO torch.hub pre-warm. The CIFAR
# orchestrator pre-clones chenyaofo/pytorch-cifar-models; MedMNIST uses
# torchvision ResNet-18/50 (no hub clone) so that race does not exist.
# The only shared state is the medmnist dataset download dir, which the
# slot script creates and the package writes atomically.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

log() { printf '[orchestrate-medmnist %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLOT_RUNNER="$SCRIPT_DIR/orchestrate_medmnist_slot.sh"

# Slot count. NUM_MEDMNIST_SLOTS (set by launch_medmnist.sh) is
# authoritative — explicit beats inferred. Only fall back to GPU
# detection when it is unset. ``nvidia-smi -L`` emits one line PER GPU
# but can also emit warning/log lines on some driver builds (this is
# what produced 2 slots on a 1-GPU g5.2xlarge), so count only lines
# that actually start with "GPU <n>:". Default to 1 on any ambiguity.
if [[ -n "${NUM_MEDMNIST_SLOTS:-}" ]]; then
    ngpus="$NUM_MEDMNIST_SLOTS"
elif command -v nvidia-smi >/dev/null; then
    ngpus=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU [0-9]' || echo 1)
    ngpus=$(echo "$ngpus" | tr -d ' ')
else
    ngpus=1
fi
case "$ngpus" in ''|*[!0-9]*) ngpus=1 ;; esac
if (( ngpus < 1 )); then ngpus=1; fi

log "spawning $ngpus slot runner(s)"

pids=()
for (( gpu=0; gpu<ngpus; gpu++ )); do
    # Stagger slot starts by ``gpu`` seconds so concurrent S3 lease
    # claims settle (same race mitigation as the CIFAR fanout).
    (
        sleep "$gpu"
        export CUDA_VISIBLE_DEVICES="$gpu"
        export GATE_NORM_GPU_SLOT="$gpu"
        exec bash "$SLOT_RUNNER"
    ) &
    pids+=("$!")
    log "  slot $gpu (CUDA_VISIBLE_DEVICES=$gpu) pid=${pids[-1]}"
done

cleanup() {
    log "received signal — forwarding to all slots"
    for pid in "${pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup TERM INT

rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=$?
done

log "all slots exited (rc=$rc)"
exit "$rc"
