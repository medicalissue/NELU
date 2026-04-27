#!/usr/bin/env bash
# β-pipeline fanout — same as orchestrate_cifar.sh but spawns
# orchestrate_beta_slot.sh instead. Each slot reads <cfg>:<act>:<seed>:<mode>
# entries from the shared S3 queue.
set -euo pipefail

log() { printf '[orchestrate-beta %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLOT_RUNNER="$SCRIPT_DIR/orchestrate_beta_slot.sh"

if [[ -n "${NUM_CIFAR_SLOTS:-}" ]]; then
    ngpus="$NUM_CIFAR_SLOTS"
elif command -v nvidia-smi >/dev/null; then
    ngpus=$(nvidia-smi -L | wc -l | tr -d ' ')
else
    ngpus=1
fi
(( ngpus < 1 )) && ngpus=1

log "spawning $ngpus β-slot runner(s)"

# Pre-warm chenyaofo hub cache once before slots fan out
log "pre-warming torch.hub cache"
python3 -c "
import torch
torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar100_resnet56',
               pretrained=False, trust_repo=True, force_reload=False)
torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar100_vgg16_bn',
               pretrained=False, trust_repo=True, force_reload=False)
print('hub cache ready')
" >/dev/null 2>&1 || log "  hub pre-warm skipped"

pids=()
for (( gpu=0; gpu<ngpus; gpu++ )); do
    (
        sleep "$gpu"
        export CUDA_VISIBLE_DEVICES="$gpu"
        export GATE_NORM_GPU_SLOT="$gpu"
        exec bash "$SLOT_RUNNER"
    ) &
    pids+=("$!")
    log "  slot $gpu pid=${pids[-1]}"
done

cleanup() {
    log "received signal — forwarding to slots"
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
}
trap cleanup TERM INT

rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
log "all slots exited (rc=$rc)"
exit "$rc"
