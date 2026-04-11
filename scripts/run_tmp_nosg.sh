#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Temporary ablation: NELU_NoSG on WRN-28-10 + ResNet-110
#
#  Dispatches 2 archs × 3 seeds = 6 single-GPU jobs in parallel
#  across GPUs 0-5. Logs each to logs/nosg_<arch>_s<seed>.log.
#  Results go to results/nosg/ (separate from Phase 1a's nelu).
#
#  Usage (from ResAct root):
#      bash scripts/run_tmp_nosg.sh
#
#  Note: does NOT interfere with run_h100.sh. If run_h100.sh is
#  currently training, wait for its current phase to free GPUs
#  or run this on spare GPUs by adjusting GPU_IDS below.
# ═══════════════════════════════════════════════════════════════

set -u
cd "$(dirname "$0")/.."
mkdir -p logs results/nosg

ARCHS=(wrn28_10 resnet110)
SEEDS=(42 123 456)
GPU_IDS=(0 1 2 3 4 5)    # 6 GPUs for 6 jobs

PY="python experiments/tmp_nosg_ablation.py --wandb --amp --compile"

i=0
for arch in "${ARCHS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        g="${GPU_IDS[$i]}"
        tag="nosg_${arch}_s${seed}"
        logf="logs/${tag}.log"
        echo "[$(date +%H:%M:%S)] launching on GPU $g: $tag"
        CUDA_VISIBLE_DEVICES=$g \
            TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_cache_gpu${g}" \
            TRITON_CACHE_DIR="/tmp/triton_cache_gpu${g}" \
            $PY --arch "$arch" --seed "$seed" > "$logf" 2>&1 &
        PIDS[$i]=$!
        i=$((i + 1))
    done
done

echo ""
echo "6 jobs launched. PIDs: ${PIDS[*]}"
echo "Tail a specific log:"
echo "  tail -f logs/nosg_wrn28_10_s42.log"
echo ""
echo "Waiting for all to finish..."
wait
echo ""
echo "[$(date +%H:%M:%S)] ALL DONE"
ls -la results/nosg/ 2>/dev/null
