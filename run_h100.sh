#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  NELU full experiment suite — H100×8
#
#  Estimated total: ~3-4 days
#
#  Phase 1: CIFAR CNN + ViT scaling        ~6h   (8 GPU parallel)
#  Phase 2: LR sensitivity + Ablation      ~2h   (8 GPU parallel)
#  Phase 3: GPT-2 LM (Small+Medium+Large)  ~24h  (8 GPU DDP)
#  Phase 4: ImageNet ViT-B                 ~10h  (8 GPU DDP)
#  Phase 5: ImageNet ViT-L                 ~36h  (8 GPU DDP)
#  Phase 6: Eval (OOD, PTQ)                ~1h   (inference)
# ═══════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

CIFAR="python experiments/main_cifar_tinyimagenet.py"
LM="torchrun --nproc_per_node=8 experiments/train_lm.py"
IMNET="torchrun --nproc_per_node=8 experiments/train_imagenet.py"
C="--amp --compile --wandb"
IMNET_DATA="/data/imagenet"

run_gpu() { local g=$1; shift; CUDA_VISIBLE_DEVICES=$g "$@" & }
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

SEEDS=(42 123 456)
CNN_ARCHS=(resnet20 resnet56 wrn28_10 densenet100 mobilenetv2 shufflenetv1)

echo "═══════════════════════════════════════════════════════════"
echo "  NELU — $(date)"
echo "═══════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────
# Phase 1a: CIFAR-100 CNN (6 arch × 3 act × 3 seeds)
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 1a: CIFAR-100 CNN ═══"
gpu=0
for seed in "${SEEDS[@]}"; do
    for arch in "${CNN_ARCHS[@]}"; do
        for act in relu gelu nelu; do
            skip_if_done "$(cifar_result $arch cifar100 $act)" && continue
            echo "[$(date +%H:%M)] GPU $gpu: $arch cifar100 $act s$seed"
            run_gpu $gpu $CIFAR --arch $arch --dataset cifar100 --act $act --seed $seed $C
            gpu=$(( (gpu+1) % 8 )); [ $gpu -eq 0 ] && wait_all
        done
    done
done
wait_all

# ─────────────────────────────────────────────────────────────
# Phase 1b: CIFAR-10 CNN (same archs, 3 seeds)
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 1b: CIFAR-10 ═══"
gpu=0
for seed in "${SEEDS[@]}"; do
    for arch in "${CNN_ARCHS[@]}"; do
        for act in relu gelu nelu; do
            skip_if_done "$(cifar_result $arch cifar10 $act)" && continue
            run_gpu $gpu $CIFAR --arch $arch --dataset cifar10 --act $act --seed $seed $C
            gpu=$(( (gpu+1) % 8 )); [ $gpu -eq 0 ] && wait_all
        done
    done
done
wait_all

# ─────────────────────────────────────────────────────────────
# Phase 2a: LR sensitivity (ResNet-20 CIFAR-100)
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 2a: LR sensitivity ═══"
gpu=0
for lr in 0.01 0.05 0.1 0.2 0.5; do
    for act in relu gelu nelu; do
        run_gpu $gpu $CIFAR --arch resnet20 --dataset cifar100 --act $act --lr $lr --seed 42 $C
        gpu=$(( (gpu+1) % 8 )); [ $gpu -eq 0 ] && wait_all
    done
done
wait_all

# ─────────────────────────────────────────────────────────────
# Phase 2b: Full ablation (ResNet-20 CIFAR-100, 3 seeds)
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 2b: Ablation ═══"
gpu=0
run_gpu $gpu python experiments/ablation_full.py --all --amp --compile
wait_all

# ─────────────────────────────────────────────────────────────
# Phase 2c: OOD + PTQ eval (inference only)
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 2c: OOD eval ═══"
python experiments/eval_ood.py

# ─────────────────────────────────────────────────────────────
# Phase 3: GPT-2 LM on FineWeb-Edu 10B (DDP, ~24h)
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 3: GPT-2 LM ═══"
for size in small medium large; do
    for act in gelu nelu; do
        RESULT="results/lm/${size}_${act}/result.json"
        skip_if_done "$RESULT" && continue
        RESUME=""
        [ -f "results/lm/${size}_${act}/last.pt" ] && RESUME="--resume results/lm/${size}_${act}/last.pt"
        echo "[$(date +%H:%M)] GPT-2 $size $act"
        $LM --size $size --act $act --wandb --compile $RESUME
    done
done

# ─────────────────────────────────────────────────────────────
# Phase 4: ImageNet DeiT-III ViT-B (DDP)
#   GELU: pretrained eval. NELU: from scratch, same recipe.
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 4: ImageNet DeiT-III ViT-B ═══"
# GELU eval (pretrained, instant)
python experiments/train_imagenet.py --model deit3_base --act gelu \
    --data $IMNET_DATA --eval-only

# NELU from scratch
if ! skip_if_done "results/imagenet/deit3_base_nelu/result.json"; then
    RESUME=""
    [ -f "results/imagenet/deit3_base_nelu/last.pt" ] && RESUME="--resume results/imagenet/deit3_base_nelu/last.pt"
    $IMNET --model deit3_base --act nelu --data $IMNET_DATA $C $RESUME
fi

# ─────────────────────────────────────────────────────────────
# Phase 5: ImageNet DeiT-III ViT-L (DDP)
#   GELU: pretrained eval. NELU: from scratch, same recipe.
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 5: ImageNet DeiT-III ViT-L ═══"
python experiments/train_imagenet.py --model deit3_large --act gelu \
    --data $IMNET_DATA --eval-only

if ! skip_if_done "results/imagenet/deit3_large_nelu/result.json"; then
    RESUME=""
    [ -f "results/imagenet/deit3_large_nelu/last.pt" ] && RESUME="--resume results/imagenet/deit3_large_nelu/last.pt"
    $IMNET --model deit3_large --act nelu --data $IMNET_DATA $C $RESUME
fi

# ─────────────────────────────────────────────────────────────
# Phase 6: ImageNet-C eval
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 6: ImageNet-C eval ═══"
python experiments/eval_ood.py \
    --gelu-ckpt results/imagenet/deit3_base_gelu/best.pt \
    --nelu-ckpt results/imagenet/deit3_base_nelu/best.pt

echo -e "\n═══════════════════════════════════════════════════════════"
echo "  ALL DONE — $(date)"
echo "═══════════════════════════════════════════════════════════"
