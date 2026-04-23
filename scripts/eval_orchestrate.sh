#!/usr/bin/env bash
# Eval-instance entrypoint. Pulls the two Swin-S checkpoints at epoch
# EVAL_EPOCH, runs eval/imagenet_robustness.py for each (clean val +
# ImageNet-C + A + R + O if present on the snapshot), uploads the JSON
# results to S3, then self-terminates the VM.
#
# The dataset volume (attached from the standard DATA_SNAPSHOT in
# run_eval.sh) is mounted at /data and already contains:
#   /data/imagenet/val
#   /data/ImageNet-C/<corruption>/<severity>/<wnid>/...
#   /data/imagenet-a (may or may not be present)
#   /data/imagenet-r
#   /data/imagenet-o
#
# Required env (injected by the launcher via user-data):
#   CKPT_BUCKET         s3://nelu-checkpoints
#   EVAL_EPOCH          integer, e.g. 250
#   EVAL_MODELS         space-separated "config:activation:exp" triples,
#                       e.g. "configs/imagenet/swin_small.yaml:gelu:swin_small-gelu
#                             configs/imagenet/swin_small.yaml:nilu:swin_small-nelu"
#   AWS_DEFAULT_REGION  us-west-2
#   EVAL_RESULT_PREFIX  subprefix under $CKPT_BUCKET for results,
#                       default "eval-results"

set -euo pipefail

log() { printf '[eval %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

: "${CKPT_BUCKET:?CKPT_BUCKET required}"
: "${EVAL_EPOCH:?EVAL_EPOCH required}"
: "${EVAL_MODELS:?EVAL_MODELS required}"
: "${AWS_DEFAULT_REGION:=us-west-2}"
: "${EVAL_RESULT_PREFIX:=eval-results}"
export AWS_DEFAULT_REGION

# ── Model-name lookup ─────────────────────────────────────────────
# eval/imagenet_robustness.py expects `--model <timm_name>`. Map our
# config file paths to timm model ids so the launcher can keep speaking
# in terms of config:activation:exp triples.
model_name_from_cfg() {
    case "$1" in
        *swin_tiny*)        echo "swin_tiny_patch4_window7_224" ;;
        *swin_small*)       echo "swin_small_patch4_window7_224" ;;
        *deit_small*)       echo "deit_small_patch16_224" ;;
        *deit_base*)        echo "deit_base_patch16_224" ;;
        *convnext_tiny*)    echo "convnext_tiny" ;;
        *convnext_small*)   echo "convnext_small" ;;
        *efficientnet_b0*)  echo "efficientnet_b0" ;;
        *efficientnet_b2*)  echo "efficientnet_b2" ;;
        *) echo "unknown-model-for-$1" ;;
    esac
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_MOUNT=/data
if [[ ! -d "$DATA_MOUNT/imagenet/val" ]]; then
    log "FATAL: $DATA_MOUNT/imagenet/val missing"
    ls "$DATA_MOUNT" >&2 || true
    exit 1
fi
log "data volume OK: $(df -h $DATA_MOUNT | tail -1)"

# Timestamp for this eval run, used inside the S3 result path.
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
WORKDIR="/tmp/eval"
mkdir -p "$WORKDIR"

ngpus=$(nvidia-smi -L | wc -l)
log "GPUs: $ngpus"

# ── Run each checkpoint ────────────────────────────────────────────
for triple in $EVAL_MODELS; do
    IFS=: read -r cfg act exp <<< "$triple"
    log "================================================================"
    log "▶ eval: cfg=$cfg act=$act exp=$exp ep=$EVAL_EPOCH"

    # The trainer stores checkpoints nested under s3://.../<exp>/<exp>/...
    src="${CKPT_BUCKET}/${exp}/${exp}/checkpoint-${EVAL_EPOCH}.pth.tar"
    local_ckpt="$WORKDIR/${exp}-ep${EVAL_EPOCH}.pth.tar"
    log "  fetching $src"
    aws s3 cp "$src" "$local_ckpt" --only-show-errors

    out_json="$WORKDIR/${exp}-ep${EVAL_EPOCH}-result.json"
    model_name=$(model_name_from_cfg "$cfg")
    log "  running imagenet_robustness.py (model=$model_name, act=$act)"
    # DataParallel across all visible GPUs is built-in; batch-size is
    # per-GPU. 256×4 on g5.12xlarge is comfortable for Swin-S.
    python -m eval.imagenet_robustness \
        --model "$model_name" \
        --activation "$act" \
        --checkpoint "$local_ckpt" \
        --data-root "$DATA_MOUNT" \
        --output "$out_json" \
        --batch-size 256 \
        --workers 8

    # Upload result
    dst="${CKPT_BUCKET}/${EVAL_RESULT_PREFIX}/${RUN_TS}/${exp}-ep${EVAL_EPOCH}.json"
    log "  uploading $dst"
    aws s3 cp "$out_json" "$dst" --only-show-errors

    # Keep the local json around in case we want to grep from the console;
    # the checkpoint can be deleted to free disk for the next one.
    rm -f "$local_ckpt"
done

log "================================================================"
log "all evals done. results under ${CKPT_BUCKET}/${EVAL_RESULT_PREFIX}/${RUN_TS}/"
