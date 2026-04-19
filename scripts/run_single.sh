#!/bin/bash
# ===================================================================
#  Run a single NELU/NiLU experiment.
#
#  Handles config loading, S3 checkpoint resume, training, and
#  result upload. Designed to be called by run_all.sh or manually.
#
#  Usage:
#    ./scripts/run_single.sh <phase> <model> <act> [extra_args...]
#
#  Examples:
#    ./scripts/run_single.sh imagenet convnext_tiny gelu
#    ./scripts/run_single.sh imagenet convnext_tiny nelu --gamma_init 0.01
#    ./scripts/run_single.sh cifar100 resnet20 nelu --seed 42
#    ./scripts/run_single.sh ablation convnext_tiny nelu --gamma_init 1.0
#
#  Environment variables:
#    S3_BUCKET       S3 prefix for results (default: s3://nelu-datasets/v2)
#    RESULTS_DIR     Local results directory (default: $REPO_ROOT/results)
#    UPSTREAM_DIR    Location of upstream training repos (default: $HOME)
# ===================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
S3_BUCKET="${S3_BUCKET:-s3://nelu-datasets/v2}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
UPSTREAM_DIR="${UPSTREAM_DIR:-${HOME}}"

# -- Parse arguments -------------------------------------------------

if [ $# -lt 3 ]; then
    echo "Usage: $0 <phase> <model> <act> [extra_args...]"
    echo ""
    echo "  phase:  imagenet | cifar100 | ablation"
    echo "  model:  convnext_tiny | efficientnet_b0 | vit_base_patch16_224 | resnet20 | ..."
    echo "  act:    gelu | nelu | silu | nilu"
    echo ""
    echo "Extra args are passed directly to the training script."
    exit 1
fi

PHASE="$1"
MODEL="$2"
ACT="$3"
shift 3
EXTRA_ARGS=("$@")

# -- Derive config and output paths ----------------------------------

# Map model names to config files (short names -> config basenames)
case "$MODEL" in
    convnext_tiny)    CONFIG_FILE="configs/imagenet/convnext_t.yaml" ;;
    convnext_small)   CONFIG_FILE="configs/imagenet/convnext_s.yaml" ;;
    convnext_base)    CONFIG_FILE="configs/imagenet/convnext_b.yaml" ;;
    efficientnet_b0)  CONFIG_FILE="configs/imagenet/efficientnet_b0.yaml" ;;
    efficientnet_b2)  CONFIG_FILE="configs/imagenet/efficientnet_b2.yaml" ;;
    efficientnet_b4)  CONFIG_FILE="configs/imagenet/efficientnet_b4.yaml" ;;
    vit_base*)        CONFIG_FILE="configs/imagenet/vit_b16.yaml" ;;
    vit_large*)       CONFIG_FILE="configs/imagenet/vit_l16.yaml" ;;
    resnet*)          CONFIG_FILE="configs/cifar100/default.yaml" ;;
    mobilenet*)       CONFIG_FILE="configs/cifar100/default.yaml" ;;
    *)                CONFIG_FILE="configs/${PHASE}/default.yaml" ;;
esac

# Override config for ablation phase
if [ "$PHASE" = "ablation" ]; then
    CONFIG_FILE="configs/ablation/gamma_init.yaml"
fi

# Build a unique run name
RUN_NAME="${PHASE}_${MODEL}_${ACT}"
for arg in "${EXTRA_ARGS[@]}"; do
    # Strip leading dashes and convert to underscores for the name
    clean=$(echo "$arg" | sed 's/^--//; s/=/_/g')
    RUN_NAME="${RUN_NAME}_${clean}"
done

OUTPUT_DIR="${RESULTS_DIR}/${RUN_NAME}"
S3_OUTPUT="${S3_BUCKET}/results/${RUN_NAME}"

echo "==================================================================="
echo "  NELU Experiment Runner"
echo "==================================================================="
echo "  Phase:     $PHASE"
echo "  Model:     $MODEL"
echo "  Act:       $ACT"
echo "  Config:    $CONFIG_FILE"
echo "  Output:    $OUTPUT_DIR"
echo "  S3:        $S3_OUTPUT"
echo "  Upstream:  $UPSTREAM_DIR"
echo "  Extra:     ${EXTRA_ARGS[*]:-<none>}"
echo "==================================================================="

# -- Check if already completed --------------------------------------

DONE_MARKER="${OUTPUT_DIR}/DONE"
if [ -f "$DONE_MARKER" ]; then
    echo "Already completed (found $DONE_MARKER). Skipping."
    exit 0
fi

# Also check S3
if aws s3 ls "${S3_OUTPUT}/DONE" >/dev/null 2>&1; then
    echo "Already completed on S3. Skipping."
    mkdir -p "$OUTPUT_DIR"
    touch "$DONE_MARKER"
    exit 0
fi

# -- Resume from S3 if checkpoint exists -----------------------------

mkdir -p "$OUTPUT_DIR"
RESUME_FLAG=""

if aws s3 ls "${S3_OUTPUT}/checkpoint.pt" >/dev/null 2>&1; then
    echo "Found checkpoint on S3 -- downloading for resume..."
    aws s3 cp "${S3_OUTPUT}/checkpoint.pt" "${OUTPUT_DIR}/checkpoint.pt" --quiet
    RESUME_FLAG="--resume ${OUTPUT_DIR}/checkpoint.pt"
    echo "  Resume from: ${OUTPUT_DIR}/checkpoint.pt"
elif [ -f "${OUTPUT_DIR}/checkpoint.pt" ]; then
    RESUME_FLAG="--resume ${OUTPUT_DIR}/checkpoint.pt"
    echo "  Resume from local: ${OUTPUT_DIR}/checkpoint.pt"
fi

# -- Validate config exists ------------------------------------------

if [ ! -f "${REPO_ROOT}/${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# -- Determine the training command ----------------------------------

# PYTHONPATH includes the NELU repo so that `import nelu` and
# `import train.act_swap` work from within upstream scripts.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

case "$PHASE" in
    imagenet)
        case "$MODEL" in
            convnext_*)
                # ConvNeXt training script (patched to support --act)
                TRAIN_CMD=(
                    torchrun
                    --nproc_per_node=8
                    "${UPSTREAM_DIR}/convnext-train/main.py"
                    --model "$MODEL"
                    --act "$ACT"
                    --data_path /data/imagenet
                    --output_dir "$OUTPUT_DIR"
                    --config "${REPO_ROOT}/${CONFIG_FILE}"
                    $RESUME_FLAG
                    "${EXTRA_ARGS[@]}"
                )
                ;;
            efficientnet_*)
                # Thin wrapper that creates a timm model + activation swap
                TRAIN_CMD=(
                    torchrun
                    --nproc_per_node=8
                    "${REPO_ROOT}/train/train_imagenet_timm.py"
                    --model "$MODEL"
                    --activation "$ACT"
                    --data-dir /data/imagenet
                    --output "$OUTPUT_DIR"
                    --config "${REPO_ROOT}/${CONFIG_FILE}"
                    $RESUME_FLAG
                    "${EXTRA_ARGS[@]}"
                )
                ;;
            vit_*)
                # DeiT III training script (patched to support --act)
                TRAIN_CMD=(
                    torchrun
                    --nproc_per_node=8
                    "${UPSTREAM_DIR}/deit-train/main.py"
                    --model "$MODEL"
                    --act "$ACT"
                    --data-path /data/imagenet
                    --output_dir "$OUTPUT_DIR"
                    --config "${REPO_ROOT}/${CONFIG_FILE}"
                    $RESUME_FLAG
                    "${EXTRA_ARGS[@]}"
                )
                ;;
            *)
                echo "ERROR: Unknown ImageNet model: $MODEL"
                exit 1
                ;;
        esac
        ;;
    cifar100)
        TRAIN_CMD=(
            python "${REPO_ROOT}/train/train_cifar.py"
            --model "$MODEL"
            --activation "$ACT"
            --config "${REPO_ROOT}/${CONFIG_FILE}"
            --output_dir "$OUTPUT_DIR"
            $RESUME_FLAG
            "${EXTRA_ARGS[@]}"
        )
        ;;
    ablation)
        TRAIN_CMD=(
            torchrun
            --nproc_per_node=8
            "${UPSTREAM_DIR}/convnext-train/main.py"
            --model "$MODEL"
            --act "$ACT"
            --data_path /data/imagenet
            --output_dir "$OUTPUT_DIR"
            --config "${REPO_ROOT}/${CONFIG_FILE}"
            $RESUME_FLAG
            "${EXTRA_ARGS[@]}"
        )
        ;;
    *)
        echo "ERROR: Unknown phase: $PHASE"
        exit 1
        ;;
esac

# -- Run training ----------------------------------------------------

echo ""
echo "Command: ${TRAIN_CMD[*]}"
echo ""

"${TRAIN_CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
TRAIN_EXIT=${PIPESTATUS[0]}

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training exited with code $TRAIN_EXIT"
    # Still sync partial results to S3
    aws s3 sync "$OUTPUT_DIR" "$S3_OUTPUT" --quiet 2>/dev/null || true
    exit $TRAIN_EXIT
fi

# -- Mark complete and sync ------------------------------------------

touch "$DONE_MARKER"
echo ""
echo "Training complete. Syncing to S3..."
aws s3 sync "$OUTPUT_DIR" "$S3_OUTPUT" --quiet 2>/dev/null || true
echo "Done: $RUN_NAME"
