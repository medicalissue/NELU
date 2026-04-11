#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  NELU full experiment suite — H100×8 / A100×8
#
#  Estimated total: ~3-4 days
#
#  Phase 1: CIFAR CNN (7 archs × 3 acts × 3 seeds × 2 datasets) ~8-15h
#  Phase 2: LR sensitivity + Ablation (parallel) + OOD eval     ~3h
#  Phase 3: ImageNet DeiT-III ViT-B (NELU from scratch)         ~10h
#  Phase 4: ImageNet DeiT-III ViT-L (NELU from scratch)         ~36h
#  Phase 5: GPT-2 LM (Small+Medium+Large, GELU+NELU)            ~24h
#  Phase 6: ImageNet-C eval (mCE)                                ~2h
#
#  Pre-flight:
#    - `wandb login`  (required, --wandb is on for all runs)
#    - AWS creds      (optional but recommended: enables S3 backup)
#    - GPU dispatch  → FIFO slot pool (no idle GPUs between jobs)
#    - Backups       → s3://nelu-datasets/ckpt-backup every 10 min
# ═══════════════════════════════════════════════════════════════

# Don't `set -e` globally — a single failure should not kill 3 days
# of training. Each phase wraps its own runs in `|| true`.
cd "$(dirname "$0")"
# Pre-create all output directories ONCE before dispatching jobs.
# Prevents races where N processes call mkdir simultaneously.
mkdir -p logs results results/checkpoints results/lm results/imagenet

CIFAR="python experiments/main_cifar_tinyimagenet.py"
LM="torchrun --nproc_per_node=8 experiments/train_lm.py"
IMNET="torchrun --nproc_per_node=8 experiments/train_imagenet.py"
C="--amp --compile --wandb"
IMNET_DATA="/data/imagenet"

# wandb service can't keep up when 8 jobs init simultaneously.
# Raise the port-file poll timeout from 30 s → 3600 s (effectively
# "wait until ready"). Fallback in the Python scripts kicks in if
# wandb truly breaks.
export WANDB__SERVICE_WAIT=3600
# Also raise the init timeout (server-side handshake).
export WANDB_INIT_TIMEOUT=600
# Disable code capture to speed up init.
export WANDB_DISABLE_CODE=true
# Keep heartbeat/start timeouts generous
export WANDB_HTTP_TIMEOUT=120

# ── S3 backup destination (override with env var) ──────────────
S3_BACKUP_BUCKET="${S3_BACKUP_BUCKET:-s3://nelu-datasets/ckpt-backup}"
S3_BACKUP_INTERVAL="${S3_BACKUP_INTERVAL:-600}"   # seconds
S3_SYNC_TOOL="${S3_SYNC_TOOL:-auto}"              # auto | s5cmd | aws

# Pick s5cmd if present, else aws
if [ "$S3_SYNC_TOOL" = "auto" ]; then
    if command -v s5cmd >/dev/null 2>&1; then
        S3_SYNC_TOOL=s5cmd
    else
        S3_SYNC_TOOL=aws
    fi
fi

backup_once() {
    # Sync key directories. Excludes tmp files (atomic write artifacts).
    if [ "$S3_SYNC_TOOL" = "s5cmd" ]; then
        s5cmd --numworkers 32 sync \
            --exclude "*.tmp" --exclude "*.tmp.tmp" \
            "results/" "$S3_BACKUP_BUCKET/results/" 2>&1 | tail -3
        s5cmd --numworkers 32 sync \
            "logs/" "$S3_BACKUP_BUCKET/logs/" 2>&1 | tail -3
    else
        aws s3 sync results/ "$S3_BACKUP_BUCKET/results/" \
            --exclude "*.tmp" --exclude "*.tmp.tmp" \
            --only-show-errors 2>&1 | tail -3
        aws s3 sync logs/ "$S3_BACKUP_BUCKET/logs/" \
            --only-show-errors 2>&1 | tail -3
    fi
}

backup_loop() {
    # Wait once before first sync so initial state has something to back up
    sleep "$S3_BACKUP_INTERVAL"
    while true; do
        echo "[$(date +%H:%M)] [backup] syncing → $S3_BACKUP_BUCKET"
        backup_once || echo "[$(date +%H:%M)] [backup] WARN: sync failed"
        sleep "$S3_BACKUP_INTERVAL"
    done
}

# Pre-flight: confirm S3 access (read OR write — sts works for either)
if aws sts get-caller-identity --output text >/dev/null 2>&1; then
    backup_loop &
    BACKUP_PID=$!
    echo "[$(date +%H:%M)] [backup] background loop started (PID $BACKUP_PID, "
    echo "                 every ${S3_BACKUP_INTERVAL}s → $S3_BACKUP_BUCKET)"
else
    echo "[$(date +%H:%M)] [backup] WARN: no AWS creds — checkpoint backup DISABLED"
    BACKUP_PID=""
fi
cleanup_backup() {
    [ -n "$BACKUP_PID" ] && kill "$BACKUP_PID" 2>/dev/null
    # Final flush before script exits
    echo "[$(date +%H:%M)] [backup] final sync"
    backup_once || true
}

# ── Pre-flight: wandb login ─────────────────────────────────────
if ! python -c "import wandb; wandb.api.api_key" 2>/dev/null; then
    echo "ERROR: wandb is not logged in. Run 'wandb login' first."
    echo "       Or remove --wandb from \$C if you don't want logging."
    exit 1
fi

wait_all() { echo "[$(date +%H:%M)] Waiting..."; wait; echo "[$(date +%H:%M)] Done."; }

# ── GPU job slot pool ──────────────────────────────────────────
# Uses a named FIFO as a semaphore: each "slot" is a GPU id.
# slot_init            — fill pool with GPU ids 0..NUM_GPUS-1
# slot_run TAG CMD...  — wait for a free slot, run command on that
#                        GPU in background with output redirected
#                        to logs/TAG.log; release slot when done
# slot_drain           — wait for all background jobs
#
# stdout only gets "[start]/[done]/[FAIL]" one-liners so parallel
# tqdm bars don't garble the tmux pane. Per-job output is in logs/.
NUM_GPUS=${NUM_GPUS:-8}
SLOT_FIFO="/tmp/nelu_gpu_slots.$$"
SLOT_PIDS=()   # track slot job PIDs so slot_drain doesn't wait on backup_loop

slot_init() {
    rm -f "$SLOT_FIFO"
    mkfifo "$SLOT_FIFO"
    exec 9<>"$SLOT_FIFO"
    for g in $(seq 0 $((NUM_GPUS - 1))); do
        echo $g >&9
    done
    SLOT_PIDS=()
}

slot_run() {
    # Usage: slot_run <tag> <command string>
    #
    # Each GPU gets its own torch.compile / Triton cache directory to
    # avoid JSON race conditions when 8 processes share ~/.cache/torch.
    local tag="$1"; shift
    local g
    read -u 9 g           # block until a GPU id is available
    local logf="logs/${tag}.log"
    local ind_cache="/tmp/inductor_cache_gpu${g}"
    local triton_cache="/tmp/triton_cache_gpu${g}"
    mkdir -p "$ind_cache" "$triton_cache"
    (
        echo "[$(date +%H:%M:%S)] [gpu $g] start $tag"
        TORCHINDUCTOR_CACHE_DIR="$ind_cache" \
        TORCH_INDUCTOR_AUTOTUNE_LOCAL_CACHE=1 \
        TORCH_INDUCTOR_AUTOTUNE_REMOTE_CACHE=0 \
        TRITON_CACHE_DIR="$triton_cache" \
        CUDA_VISIBLE_DEVICES=$g eval "$@" > "$logf" 2>&1
        local rc=$?
        if [ $rc -eq 0 ]; then
            echo "[$(date +%H:%M:%S)] [gpu $g] done  $tag"
        else
            echo "[$(date +%H:%M:%S)] [gpu $g] FAIL  $tag (rc=$rc, see $logf)"
        fi
        echo $g >&9       # release the slot
    ) &
    SLOT_PIDS+=($!)
}

slot_drain() {
    # Wait ONLY for slot jobs, not for the backup_loop (which is infinite).
    if [ ${#SLOT_PIDS[@]} -gt 0 ]; then
        wait "${SLOT_PIDS[@]}" 2>/dev/null || true
    fi
    SLOT_PIDS=()
}

slot_cleanup() {
    exec 9>&- 2>/dev/null || true
    rm -f "$SLOT_FIFO"
}
trap 'slot_cleanup; cleanup_backup' EXIT

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
CNN_ARCHS=(resnet20 resnet56 resnet110 wrn28_10 densenet100 mobilenetv2 shufflenetv1)

echo "═══════════════════════════════════════════════════════════"
echo "  NELU — $(date)"
echo "═══════════════════════════════════════════════════════════"

# Initialize the GPU slot pool for phases that use single-GPU jobs
slot_init

# ─────────────────────────────────────────────────────────────
# Phase 1a: CIFAR-100 CNN (7 arch × 3 act × 3 seeds = 63 jobs)
# ─────────────────────────────────────────────────────────────

# echo -e "\n═══ Phase 1a: CIFAR-100 CNN ═══"
# for seed in "${SEEDS[@]}"; do
#     for arch in "${CNN_ARCHS[@]}"; do
#         for act in relu gelu nelu; do
#             skip_if_done "$(cifar_result $arch cifar100 $act $seed)" && continue
#             slot_run "cifar100_${arch}_${act}_s${seed}" \
#                 "$CIFAR --arch $arch --dataset cifar100 --act $act --seed $seed $C"
#         done
#     done
# done
# slot_drain

# ─────────────────────────────────────────────────────────────
# Phase 1b: CIFAR-10 CNN (7 arch × 3 act × 3 seeds = 63 jobs)
# ─────────────────────────────────────────────────────────────

# echo -e "\n═══ Phase 1b: CIFAR-10 ═══"
# for seed in "${SEEDS[@]}"; do
#     for arch in "${CNN_ARCHS[@]}"; do
#         for act in relu gelu nelu; do
#             skip_if_done "$(cifar_result $arch cifar10 $act $seed)" && continue
#             slot_run "cifar10_${arch}_${act}_s${seed}" \
#                 "$CIFAR --arch $arch --dataset cifar10 --act $act --seed $seed $C"
#         done
#     done
# done
# slot_drain

# ─────────────────────────────────────────────────────────────
# Phase 2a: LR sensitivity (ResNet-20 CIFAR-100, single seed)
# ─────────────────────────────────────────────────────────────

# echo -e "\n═══ Phase 2a: LR sensitivity ═══"
# for lr in 0.01 0.05 0.1 0.2 0.5; do
#     for act in relu gelu nelu; do
#         skip_if_done "$(cifar_result resnet20 cifar100 $act 42 $lr)" && continue
#         slot_run "lrsweep_resnet20_${act}_lr${lr}" \
#             "$CIFAR --arch resnet20 --dataset cifar100 --act $act --lr $lr --seed 42 $C"
#     done
# done
# slot_drain

# ─────────────────────────────────────────────────────────────
# Phase 2b: Full ablation (ResNet-20 CIFAR-100, 9 variants × 3 seeds = 27 jobs)
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 2b: Ablation (parallel) ═══"
ABLATION_VARIANTS=(gelu nelu nelu_no_sg nelu_dim_w nelu_dim_c nelu_dim_hw nelu_dim_chw learnable_tau gelu_wd2)
for variant in "${ABLATION_VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        RESULT="results/ablation_${variant}_s${seed}.json"
        skip_if_done "$RESULT" && continue
        slot_run "ablation_${variant}_s${seed}" \
            "python experiments/ablation_full.py --variant $variant --seeds $seed --amp --compile --wandb"
    done
done
slot_drain

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
python experiments/eval_ood.py --wandb 2>&1 | tee logs/phase2c_ood.log || \
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
# Phase 6: ImageNet-C eval (timm GELU pretrained vs trained NELU)
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 6: ImageNet-C eval ═══"
for model in deit3_base deit3_large; do
    NELU_CKPT="results/imagenet/${model}_nelu/best.pt"
    if [ ! -f "$NELU_CKPT" ]; then
        echo "  SKIP $model: NELU checkpoint not found ($NELU_CKPT)"
        continue
    fi
    echo "[$(date +%H:%M)] ImageNet-C eval: $model"
    python experiments/eval_imagenet_c.py \
        --model $model \
        --data /data/ImageNet-C \
        --gelu-pretrained \
        --nelu-ckpt "$NELU_CKPT" \
        --wandb \
        2>&1 | tee logs/imnet_c_${model}.log || \
        echo "[WARN] ImageNet-C eval $model failed — continuing"
done

echo -e "\n═══════════════════════════════════════════════════════════"
echo "  ALL DONE — $(date)"
echo "═══════════════════════════════════════════════════════════"
