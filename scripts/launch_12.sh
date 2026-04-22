#!/usr/bin/env bash
# Launch the 12-run ImageNet-1k campaign on SkyPilot managed jobs.
#
# Covers six architectures × {baseline, Gate-Normalization}:
#   ConvNeXt-T/S, DeiT-S/B, Swin-T/S  (EfficientNet-B0/B2 are run separately.)
#
# All jobs share sky/train.yaml, which pins us-west-2 and seeks across AZs
# a/b/c/d until spot capacity opens. The managed-jobs controller handles
# initial-capacity retries and preemption recovery; we just enqueue.
#
# Usage:
#   set -a; source .env; set +a
#   bash scripts/launch_12.sh
#
# Required env (loaded from .env):
#   DATA_BUCKET, CKPT_BUCKET, WANDB_API_KEY, WANDB_ENTITY
# Optional:
#   TORCHCOMPILE, TORCHCOMPILE_MODE  — forwarded if set.

set -euo pipefail

: "${DATA_BUCKET:?DATA_BUCKET not set — source .env first}"
: "${CKPT_BUCKET:?CKPT_BUCKET not set — source .env first}"
: "${WANDB_API_KEY:?WANDB_API_KEY not set — source .env first}"
: "${WANDB_ENTITY:=}"
: "${TORCHCOMPILE:=}"
: "${TORCHCOMPILE_MODE:=}"

# model → "baseline alt" pairs. ConvNeXt/DeiT/Swin all pair GELU↔NELU.
MODELS=(convnext_tiny convnext_small deit_small deit_base swin_tiny swin_small)

for model in "${MODELS[@]}"; do
  for act in gelu nelu; do
    name="${model//_/-}-${act}"        # e.g. convnext-tiny-nelu
    config="configs/imagenet/${model}.yaml"

    echo "▶ Enqueuing ${name} (${config})"
    sky jobs launch -y -n "${name}" sky/train.yaml \
      --env CONFIG="${config}" \
      --env ACTIVATION="${act}" \
      --env EXP_NAME="${name}" \
      --env DATA_BUCKET="${DATA_BUCKET}" \
      --env CKPT_BUCKET="${CKPT_BUCKET}" \
      --env WANDB_API_KEY="${WANDB_API_KEY}" \
      --env WANDB_ENTITY="${WANDB_ENTITY}" \
      --env TORCHCOMPILE="${TORCHCOMPILE}" \
      --env TORCHCOMPILE_MODE="${TORCHCOMPILE_MODE}"
  done
done

echo
echo "All 12 jobs submitted. Follow with:"
echo "  sky jobs queue                   # status"
echo "  sky jobs logs --name <job-name>  # stream logs"
