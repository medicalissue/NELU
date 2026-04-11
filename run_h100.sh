#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  NELU full experiment suite — H100×8 / A100×8
#
#  Estimated total: ~3-4 days
#
#  Activations under test: relu / gelu / silu  (baselines)
#                          nelu (=GELU+RMS)  nilu (=SiLU+RMS)  (ours)
#
#  Phase 1: CIFAR training — main + LR sweep + ablation
#           (combined LPT queue, 7 archs × 5 acts × 3 seeds = 105 main
#            + 25 LR sweep + 15 ablation = 145 single-GPU jobs)     ~10-14h
#  Phase 2: OOD eval on CIFAR-100-C (per-ckpt parallel, 105+ jobs)   ~15min
#  Phase 3: ImageNet ConvNeXt-T  (GELU pretrained, NELU from scratch) ~30h
#  Phase 4: ImageNet EfficientNet-B2 (SiLU pretrained, NiLU scratch)  ~22-26h
#  Phase 5: ImageNet DeiT-III ViT-B  (GELU pretrained, NELU scratch)  ~40h
#  Phase 6: ImageNet-C eval (mCE, timm baseline vs trained ours)      ~1h
#  Phase 7: COCO Det/Seg — Mask R-CNN + ConvNeXt-T (GELU vs NELU)     ~20h
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
mkdir -p logs results results/checkpoints results/imagenet results/coco

CIFAR="python experiments/main_cifar_tinyimagenet.py"
C="--amp --compile --wandb"
IMNET_DATA="/data/imagenet"
COCO_DATA="/data/coco"
# Phase 3-5 ImageNet runs use bit-exact upstream stacks; see those phases
# below for the exact torchrun invocations (no $IMNET shorthand).

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

# Longest-Processing-Time (LPT) order: largest archs first so that
# they don't end up at the tail of the schedule. With FIFO slot pool
# this gives near-optimal makespan. Order roughly tracks per-job
# wall-clock from earlier runs.
CNN_ARCHS_LPT=(densenet100 wrn28_10 resnet110 mobilenetv2 shufflenetv1 resnet56 resnet20)

ABLATION_VARIANTS=(nelu_dim_w nelu_dim_c nelu_dim_hw learnable_tau gelu_wd2)

echo "═══════════════════════════════════════════════════════════"
echo "  NELU — $(date)"
echo "═══════════════════════════════════════════════════════════"

# Initialize the GPU slot pool for all single-GPU phases
slot_init

# ─────────────────────────────────────────────────────────────
# Phase 1: CIFAR training — combined queue (no per-sub-phase drain)
#
# Three groups of single-GPU jobs are queued into ONE FIFO pool:
#   1a. Main CIFAR-100  (7 archs × 3 acts × 3 seeds = 63 jobs)
#   1b. LR sensitivity  (resnet20 × 5 LRs × 3 acts = 15 jobs)
#   1c. Ablation        (resnet20 × 5 variants × 3 seeds = 15 jobs)
#
# Single drain at the end: idle gaps between sub-phases vanish, and
# the smaller LR/ablation jobs naturally fill the tail of the larger
# CIFAR jobs. Largest archs queued first (LPT) to minimize makespan.
# ─────────────────────────────────────────────────────────────

# 5 activations:
#   relu, gelu, silu           — baselines
#   nelu  (= GELU + RMS)       — our GELU variant
#   nilu  (= SiLU + RMS)       — our SiLU variant
# Total main grid: 7 archs × 5 acts × 3 seeds = 105 jobs
MAIN_ACTS=(relu gelu silu nelu nilu)

# LR sweep keeps only the 3 baselines (plus both of ours) at seed 42:
#   5 lrs × 5 acts = 25 jobs
LR_ACTS=(relu gelu silu nelu nilu)

echo -e "\n═══ Phase 1: CIFAR training (combined queue) ═══"

# 1a. Main CIFAR-100, longest archs first
for arch in "${CNN_ARCHS_LPT[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for act in "${MAIN_ACTS[@]}"; do
            skip_if_done "$(cifar_result $arch cifar100 $act $seed)" && continue
            slot_run "cifar100_${arch}_${act}_s${seed}" \
                "$CIFAR --arch $arch --dataset cifar100 --act $act --seed $seed $C"
        done
    done
done

# 1b. LR sensitivity sweep (resnet20 only)
for lr in 0.01 0.05 0.1 0.2 0.5; do
    for act in "${LR_ACTS[@]}"; do
        skip_if_done "$(cifar_result resnet20 cifar100 $act 42 $lr)" && continue
        slot_run "lrsweep_resnet20_${act}_lr${lr}" \
            "$CIFAR --arch resnet20 --dataset cifar100 --act $act --lr $lr --seed 42 $C"
    done
done

# 1c. Ablation (resnet20 variants)
for variant in "${ABLATION_VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        RESULT="results/ablation_${variant}_s${seed}.json"
        skip_if_done "$RESULT" && continue
        slot_run "ablation_${variant}_s${seed}" \
            "python experiments/ablation_full.py --variant $variant --seeds $seed --amp --compile --wandb"
    done
done

# 1d. RMS-axis ablation on MobileNetV2 (DW conv, single seed 42).
# CHW = default `nelu` and is already in Phase 1a's main grid as
#   results/main_mobilenetv2_cifar100_nelu_s42.json
# Only HW and C variants need to be run here.
for variant in nelu_hw nelu_c; do
    RESULT="results/rms_axis/main_mobilenetv2_cifar100_${variant}_s42.json"
    skip_if_done "$RESULT" && continue
    slot_run "rmsaxis_mobilenetv2_${variant}_s42" \
        "python experiments/ablation_mobilenetv2_rms_axis.py --variant $variant --seed 42 --amp --compile --wandb"
done

echo -e "\n[$(date +%H:%M)] Phase 1 fully queued — draining..."
slot_drain
echo "[$(date +%H:%M)] Phase 1 complete."

# Aggregate ablation runs (post-processing only)
python - <<'PY' || echo "[WARN] ablation aggregation failed"
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
# Phase 2: OOD eval on CIFAR-100-C (per-ckpt parallel via slot pool)
# Tiny inference jobs (~5-15 s each) — totally fills the pool
# briefly and finishes fast. Cached per-ckpt under results/ood/.
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 2: OOD eval (CIFAR-100-C, parallel) ═══"
for f in results/checkpoints/*_cifar100_*_best.pt; do
    [ -f "$f" ] || continue
    name=$(basename "$f" _best.pt)
    cache="results/ood/${name}.json"
    if [ -f "$cache" ]; then
        echo "  SKIP (cached): $name"
        continue
    fi
    slot_run "ood_${name}" \
        "python experiments/eval_ood.py --checkpoint \"$f\""
done
slot_drain

python experiments/eval_ood.py --aggregate-only 2>&1 | tee logs/phase2_ood_agg.log || \
    echo "[WARN] OOD aggregation failed — continuing"

# ─────────────────────────────────────────────────────────────
# §4.3 ImageNet trio — bit-exact reproduction of timm/FB recipes.
#
# Each phase uses the ORIGINAL training stack that produced the
# pretrained checkpoint, with a single-line activation swap. The
# baseline is the timm pretrained ckpt (loaded via timm + evaluated
# under the same eval transform) — only OUR variant is trained.
#
#   Model            Trained by      Stack we use
#   ----------------------------------------------------------------
#   ConvNeXt-T       FB ConvNeXt     /home/ubuntu/convnext-train/main.py
#   EffNet-B2        timm/train.py   /home/ubuntu/NELU/timm-train/train.py
#                                    via experiments/train_imagenet_timm.py
#   DeiT-III B       FB deit         /home/ubuntu/deit-train/main.py
#
# Each clone has been patched with NELU/NiLU activation swap. The
# original optimizer / scheduler / dataloader / EMA / AMP / DDP /
# scaler / sampler are 100 % preserved.
#
# Drained sequentially — each occupies all 8 GPUs via DDP.
# ─────────────────────────────────────────────────────────────

CONVNEXT_DIR=/home/ubuntu/convnext-train
DEIT_DIR=/home/ubuntu/deit-train
TIMM_TRAIN_PY=/home/ubuntu/NELU/timm-train/train.py
export TIMM_TRAIN_PY
RESACT_DIR=/home/ubuntu/ResAct

# Baseline pretrained ImageNet eval — just `python -c` against timm.
# We don't need a full DDP launch for inference; one GPU is enough.
imnet_eval_baseline() {
    local timm_id="$1" eval_log="$2"
    if [ -f "logs/${eval_log}.done" ]; then
        echo "  SKIP baseline eval: $timm_id (already done)"
        return 0
    fi
    echo "[$(date +%H:%M)] baseline eval: $timm_id"
    python experiments/eval_timm_pretrained.py \
        --model "$timm_id" --data "$IMNET_DATA/val" \
        2>&1 | tee "logs/${eval_log}.log" && \
        touch "logs/${eval_log}.done" || \
        echo "[WARN] baseline eval $timm_id failed — continuing"
}

# ─────────── Phase 3: ConvNeXt-T (FB ConvNeXt main.py) ───────────
echo -e "\n═══ Phase 3: ImageNet ConvNeXt-T (GELU vs NELU) ═══"
imnet_eval_baseline convnext_tiny.fb_in1k imnet_convnext_t_gelu_eval

if ! skip_if_done "results/imagenet/convnext_tiny_nelu/result.json"; then
    echo "[$(date +%H:%M)] ConvNeXt-T NELU from scratch (FB ConvNeXt main.py)"
    (cd "$CONVNEXT_DIR" && \
     torchrun --nproc_per_node=8 main.py \
        --model convnext_tiny --drop_path 0.1 \
        --batch_size 128 --lr 4e-3 --update_freq 4 \
        --model_ema true --model_ema_eval true \
        --data_path "$IMNET_DATA" \
        --output_dir "$RESACT_DIR/results/imagenet/convnext_tiny_nelu" \
        --use_amp true --auto_resume true \
        --enable_wandb true --project nelu \
        --torch_compile true \
        --act nelu) \
        2>&1 | tee logs/imnet_convnext_t_nelu.log || \
        echo "[WARN] ConvNeXt-T NELU train failed — continuing"
fi

# ─────────── Phase 4: EfficientNet-B2 (timm/train.py) ────────────
echo -e "\n═══ Phase 4: ImageNet EfficientNet-B2 (SiLU vs NiLU) ═══"
imnet_eval_baseline efficientnet_b2.ra_in1k imnet_effnet_b2_silu_eval

if ! skip_if_done "results/imagenet/efficientnet_b2_nilu/result.json"; then
    echo "[$(date +%H:%M)] EfficientNet-B2 NiLU from scratch (timm/train.py)"
    # Wightman B2 RA recipe (from hfdocs/source/training_script.mdx).
    # Original: -b 128 × 2 GPU = 256 effective at lr 0.016.
    # We use 8 GPU × -b 128 = 1024 effective with linearly scaled lr 0.064
    # (matches Wightman's B0 scaling: 3× effective batch ⇒ 3× lr).
    torchrun --nproc_per_node=8 experiments/train_imagenet_timm.py \
        --our-act nilu \
        "$IMNET_DATA" \
        --model efficientnet_b2 -b 128 \
        --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 \
        --opt rmsproptf --opt-eps .001 -j 8 \
        --warmup-lr 1e-6 --warmup-epochs 5 --weight-decay 1e-5 \
        --drop 0.3 --drop-path 0.2 \
        --model-ema --model-ema-decay 0.9999 \
        --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 \
        --amp --amp-dtype float16 --lr .064 \
        --torchcompile inductor \
        --output "$RESACT_DIR/results/imagenet/efficientnet_b2_nilu" \
        --experiment efficientnet_b2_nilu \
        --log-wandb --wandb-project nelu --wandb-tags imagenet effnet \
        2>&1 | tee logs/imnet_effnet_b2_nilu.log || \
        echo "[WARN] EfficientNet-B2 NiLU train failed — continuing"
fi

# ─────────── Phase 5: DeiT-III ViT-B (FB deit main.py) ───────────
echo -e "\n═══ Phase 5: ImageNet DeiT-III ViT-B (GELU vs NELU) ═══"
imnet_eval_baseline deit3_base_patch16_224.fb_in1k imnet_deit3_b_gelu_eval

if ! skip_if_done "results/imagenet/deit3_base_nelu/result.json"; then
    echo "[$(date +%H:%M)] DeiT-III ViT-B NELU from scratch (FB deit main.py)"
    # README_revenge ImageNet-1k pretraining cmd (line 412 of README_revenge.md).
    (cd "$DEIT_DIR" && \
     torchrun --nproc_per_node=8 main.py \
        --model deit_base_patch16_LS \
        --data-path "$IMNET_DATA" \
        --output_dir "$RESACT_DIR/results/imagenet/deit3_base_nelu" \
        --batch 256 --lr 3e-3 --epochs 800 --weight-decay 0.05 \
        --sched cosine --input-size 192 --eval-crop-ratio 1.0 \
        --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 \
        --drop 0.0 --nb-classes 1000 --seed 0 \
        --opt fusedlamb --warmup-lr 1e-6 \
        --mixup .8 --drop-path 0.2 --cutmix 1.0 \
        --unscale-lr --repeated-aug --bce-loss \
        --color-jitter 0.3 --ThreeAugment \
        --torch-compile \
        --act nelu) \
        2>&1 | tee logs/imnet_deit3_b_nelu.log || \
        echo "[WARN] DeiT-III ViT-B NELU train failed — continuing"
fi

# ─────────────────────────────────────────────────────────────
# Phase 6: ImageNet-C eval (timm baseline pretrained vs trained ours)
# All 3 models evaluated with DataParallel across 8 GPUs.
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 6: ImageNet-C eval ═══"
declare -A IMNET_C_OURS=(
    [convnext_tiny]=nelu
    [efficientnet_b2]=nilu
    [deit3_base]=nelu
)
for model in convnext_tiny efficientnet_b2 deit3_base; do
    our="${IMNET_C_OURS[$model]}"
    OUR_CKPT="results/imagenet/${model}_${our}/best.pt"
    if [ ! -f "$OUR_CKPT" ]; then
        echo "  SKIP $model: ${our} checkpoint not found ($OUR_CKPT)"
        continue
    fi
    echo "[$(date +%H:%M)] ImageNet-C eval: $model (baseline vs $our)"
    python experiments/eval_imagenet_c.py \
        --model "$model" \
        --data /data/ImageNet-C \
        --baseline-pretrained \
        --our-act "$our" \
        --our-ckpt "$OUR_CKPT" \
        --wandb \
        2>&1 | tee "logs/imnet_c_${model}.log" || \
        echo "[WARN] ImageNet-C eval $model failed — continuing"
done

# ─────────────────────────────────────────────────────────────
# Phase 7: COCO Det/Seg — Mask R-CNN + ConvNeXt-T  (1× × 2)
#   Backbone init: GELU = timm pretrained, NELU = our Phase 3 ckpt
# ─────────────────────────────────────────────────────────────

echo -e "\n═══ Phase 7: COCO Mask R-CNN + ConvNeXt-T ═══"
for act in gelu nelu; do
    RESULT="results/coco/maskrcnn_convnext_tiny_${act}/result.json"
    skip_if_done "$RESULT" && continue
    if [ "$act" = "nelu" ]; then
        BACKBONE_CKPT="results/imagenet/convnext_tiny_nelu/best.pt"
        if [ ! -f "$BACKBONE_CKPT" ]; then
            echo "  SKIP coco nelu: missing backbone ($BACKBONE_CKPT)"
            continue
        fi
        BACKBONE_ARG="--backbone-ckpt $BACKBONE_CKPT"
    else
        BACKBONE_ARG=""
    fi
    echo "[$(date +%H:%M)] COCO Mask R-CNN ConvNeXt-T $act"
    torchrun --nproc_per_node=8 experiments/train_coco_maskrcnn.py \
        --backbone convnext_tiny --act "$act" \
        --data $COCO_DATA --schedule 1x --wandb \
        $BACKBONE_ARG \
        2>&1 | tee "logs/coco_maskrcnn_convnext_tiny_${act}.log" || \
        echo "[WARN] COCO $act failed — continuing"
done

echo -e "\n═══════════════════════════════════════════════════════════"
echo "  ALL DONE — $(date)"
echo "═══════════════════════════════════════════════════════════"
