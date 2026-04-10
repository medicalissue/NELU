set -e
cd "$(dirname "$0")"

CIFAR="python experiments/main_cifar_tinyimagenet.py"
LM="torchrun --nproc_per_node=8 experiments/train_lm.py"
IMNET="torchrun --nproc_per_node=8 experiments/train_imagenet.py"
C="--amp --compile --wandb"
IMNET_DATA="/data/imagenet"

wait_all() { echo "[$(date +%H:%M)] Waiting..."; wait; echo "[$(date +%H:%M)] Done."; }

# Skip if result JSON already exists
skip_if_done() {
    local f="$1"
    if [ -f "$f" ]; then
        echo "  SKIP (already done): $f"
        return 0  # true = skip
    fi
    return 1  # false = run
}

# CIFAR: result file pattern
cifar_result() {
    local arch=$1 dataset=$2 act=$3 noise=$4
    local tag=""
    [ -n "$noise" ] && tag="_noise${noise}"
    echo "results/main_${arch}_${dataset}_${act}${tag}.json"
}

SEEDS=(42)
CNN_ARCHS=(resnet20)

echo "═══════════════════════════════════════════════════════════"
echo "  NELU — $(date)"
echo "═══════════════════════════════════════════════════════════"


# ─────────────────────────────────────────────────────────────
# Phase 1b: CIFAR-10 CNN (same archs, 3 seeds)
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 1b: CIFAR-10 ═══"
gpu=0
for seed in "${SEEDS[@]}"; do
    for arch in "${CNN_ARCHS[@]}"; do
        for act in relu gelu nelu; do
            skip_if_done "$(cifar_result $arch cifar10 $act)" && continue
            $CIFAR --arch $arch --dataset cifar10 --act $act --seed $seed $C
            wait_all
        done
    done
done
wait_all