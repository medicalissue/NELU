#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  NELU full experiment suite — H100×8 / A100×8
#
#  Estimated total: ~3-4 days
#
#  Phase 1: CIFAR CNN (6 archs × 3 acts × 3 seeds × 2 datasets) ~8-15h
#  Phase 2: LR sensitivity + Ablation (parallel) + OOD eval     ~3h
#  Phase 3: ImageNet DeiT-III ViT-B (NELU from scratch)         ~10h
#  Phase 4: ImageNet DeiT-III ViT-L (NELU from scratch)         ~36h
#  Phase 5: GPT-2 LM (Small+Medium+Large, GELU+NELU)            ~24h
#
#  Pre-flight: `wandb login` (required, --wandb is on for all runs)
# ═══════════════════════════════════════════════════════════════

# Don't `set -e` globally — a single failure should not kill 3 days
# of training. Each phase wraps its own runs in `|| true`.
cd "$(dirname "$0")"
mkdir -p logs results

CIFAR="python experiments/main_cifar_tinyimagenet.py"
LM="torchrun --nproc_per_node=8 experiments/train_lm.py"
IMNET="torchrun --nproc_per_node=8 experiments/train_imagenet.py"
C="--amp --compile --wandb"
IMNET_DATA="/data/imagenet"

# ── Pre-flight: wandb login ─────────────────────────────────────
if ! python -c "import wandb; wandb.api.api_key" 2>/dev/null; then
    echo "ERROR: wandb is not logged in. Run 'wandb login' first."
    echo "       Or remove --wandb from \$C if you don't want logging."
    exit 1
fi

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

# CIFAR: result file pattern (must match main_cifar_tinyimagenet.py)
#   main_<arch>_<dataset>_<act>[_noise<x>][_lr<x>]_s<seed>.json
cifar_result() {
    local arch=$1 dataset=$2 act=$3 seed=$4 lr=$5 noise=$6
    local tag=""
    [ -n "$noise" ] && tag="${tag}_noise${noise}"
    [ -n "$lr" ] && tag="${tag}_lr${lr}"
    echo "results/main_${arch}_${dataset}_${act}${tag}_s${seed}.json"
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
            skip_if_done "$(cifar_result $arch cifar100 $act $seed)" && continue
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
            skip_if_done "$(cifar_result $arch cifar10 $act $seed)" && continue
            echo "[$(date +%H:%M)] GPU $gpu: $arch cifar10 $act s$seed"
            run_gpu $gpu $CIFAR --arch $arch --dataset cifar10 --act $act --seed $seed $C
            gpu=$(( (gpu+1) % 8 )); [ $gpu -eq 0 ] && wait_all
        done
    done
done
wait_all

# ─────────────────────────────────────────────────────────────
# Phase 2a: LR sensitivity (ResNet-20 CIFAR-100, single seed)
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 2a: LR sensitivity ═══"
gpu=0
for lr in 0.01 0.05 0.1 0.2 0.5; do
    for act in relu gelu nelu; do
        skip_if_done "$(cifar_result resnet20 cifar100 $act 42 $lr)" && continue
        echo "[$(date +%H:%M)] GPU $gpu: resnet20 cifar100 $act lr=$lr"
        run_gpu $gpu $CIFAR --arch resnet20 --dataset cifar100 --act $act --lr $lr --seed 42 $C
        gpu=$(( (gpu+1) % 8 )); [ $gpu -eq 0 ] && wait_all
    done
done
wait_all

# ─────────────────────────────────────────────────────────────
# Phase 2b: Full ablation (ResNet-20 CIFAR-100, 9 variants × 3 seeds)
#   Dispatched across 8 GPUs in parallel — ~1h instead of ~9h.
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 2b: Ablation (parallel) ═══"
ABLATION_VARIANTS=(gelu nelu nelu_no_sg nelu_dim_w nelu_dim_c nelu_dim_hw nelu_dim_chw learnable_tau gelu_wd2)
gpu=0
for variant in "${ABLATION_VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        RESULT="results/ablation_${variant}_s${seed}.json"
        skip_if_done "$RESULT" && continue
        echo "[$(date +%H:%M)] GPU $gpu: ablation $variant s$seed"
        run_gpu $gpu python experiments/ablation_full.py \
            --variant $variant --seeds $seed --amp --compile
        gpu=$(( (gpu+1) % 8 )); [ $gpu -eq 0 ] && wait_all
    done
done
wait_all

# Aggregate per-run JSONs into the combined ablation_full.json
python - <<'PY' || echo "[WARN] aggregation failed"
import json, glob, os, numpy as np
results = {}
for f in glob.glob("results/ablation_*_s*.json"):
    name = os.path.basename(f).replace("ablation_", "").replace(".json", "")
    variant, seed = name.rsplit("_s", 1)
    with open(f) as fp:
        d = json.load(fp)
    results.setdefault(variant, []).append(d.get("best_acc"))
agg = {v: {"mean": float(np.mean(a)), "std": float(np.std(a)), "runs": a}
       for v, a in results.items()}
with open("results/ablation_full.json", "w") as fp:
    json.dump(agg, fp, indent=2)
print(f"  aggregated {len(results)} variants → results/ablation_full.json")
for v, r in sorted(agg.items(), key=lambda x: -x[1]["mean"]):
    print(f"    {v:<16} {r['mean']:>6.2f} ± {r['std']:.2f}")
PY

# ─────────────────────────────────────────────────────────────
# Phase 2c: OOD eval on CIFAR-100-C
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 2c: OOD eval (CIFAR-100-C) ═══"
python experiments/eval_ood.py 2>&1 | tee logs/phase2c_ood.log || \
    echo "[WARN] Phase 2c failed — continuing"

# ─────────────────────────────────────────────────────────────
# Phase 3: ImageNet DeiT-III ViT-B (DDP)
#   GELU: pretrained eval. NELU: from scratch, same recipe.
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 3: ImageNet DeiT-III ViT-B ═══"
python experiments/train_imagenet.py --model deit3_base --act gelu \
    --data $IMNET_DATA --eval-only \
    2>&1 | tee logs/imnet_b_gelu_eval.log || \
    echo "[WARN] ViT-B GELU eval failed — continuing"

if ! skip_if_done "results/imagenet/deit3_base_nelu/result.json"; then
    echo "[$(date +%H:%M)] DeiT-III ViT-B NELU from scratch"
    $IMNET --model deit3_base --act nelu --data $IMNET_DATA $C \
        2>&1 | tee logs/imnet_b_nelu.log || \
        echo "[WARN] ViT-B NELU train failed — continuing"
fi

# ─────────────────────────────────────────────────────────────
# Phase 4: ImageNet DeiT-III ViT-L (DDP)
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 4: ImageNet DeiT-III ViT-L ═══"
python experiments/train_imagenet.py --model deit3_large --act gelu \
    --data $IMNET_DATA --eval-only \
    2>&1 | tee logs/imnet_l_gelu_eval.log || \
    echo "[WARN] ViT-L GELU eval failed — continuing"

if ! skip_if_done "results/imagenet/deit3_large_nelu/result.json"; then
    echo "[$(date +%H:%M)] DeiT-III ViT-L NELU from scratch"
    $IMNET --model deit3_large --act nelu --data $IMNET_DATA $C \
        2>&1 | tee logs/imnet_l_nelu.log || \
        echo "[WARN] ViT-L NELU train failed — continuing"
fi

# ─────────────────────────────────────────────────────────────
# Phase 5: GPT-2 LM on FineWeb-Edu 10B (DDP, ~24h)
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 5: GPT-2 LM ═══"
for size in small medium large; do
    for act in gelu nelu; do
        RESULT="results/lm/${size}_${act}/result.json"
        skip_if_done "$RESULT" && continue
        echo "[$(date +%H:%M)] GPT-2 $size $act (auto-resume from last.pt if present)"
        $LM --size $size --act $act --wandb --compile \
            2>&1 | tee logs/lm_${size}_${act}.log || \
            echo "[WARN] LM $size $act failed — continuing"
    done
done

# ─────────────────────────────────────────────────────────────
# Phase 6: (placeholder) ImageNet-C eval
# eval_ood.py only supports CIFAR-100-C; ImageNet-C eval needs a
# separate script that wraps timm models. Skipping for now.
# ─────────────────────────────────────────────────────────────

echo -e "\n═══════════════════════════════════════════════════════════"
echo "  ALL DONE — $(date)"
echo "═══════════════════════════════════════════════════════════"
