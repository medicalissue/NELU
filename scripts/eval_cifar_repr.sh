#!/bin/bash
# Run every CIFAR-100 representation-quality probe for one checkpoint.
#
# Usage:
#   eval_cifar_repr.sh MODEL ACTIVATION CHECKPOINT OUTDIR [DATA_ROOT]
#
# Example:
#   eval_cifar_repr.sh resnet56 nelu \
#       /tmp/ckpt/resnet56-nelu-s42/checkpoint.pt \
#       results/resnet56-nelu-s42
#
# Each probe writes ``<OUTDIR>/<probe>.json``. The script is idempotent:
# if an output already exists it skips that probe. To rerun, ``rm`` the
# JSON first. CKA needs *two* checkpoints, so it is *not* run from this
# launcher — see ``scripts/eval_cifar_cka.sh`` for the pairwise sweep.

set -euo pipefail

MODEL=${1:?usage: $0 MODEL ACTIVATION CHECKPOINT OUTDIR [DATA_ROOT]}
ACT=${2:?}
CKPT=${3:?}
OUTDIR=${4:?}
DATA_ROOT=${5:-/data}

mkdir -p "$OUTDIR"

run_probe() {
    local probe=$1; shift
    local out="$OUTDIR/${probe}.json"
    if [[ -s "$out" ]]; then
        echo "[skip] $probe (exists)"
        return 0
    fi
    echo "[run]  $probe → $out"
    python -m eval.cifar."$probe" \
        --model "$MODEL" --activation "$ACT" \
        --checkpoint "$CKPT" --data-root "$DATA_ROOT" \
        --output "$out" "$@"
}

# Order matters: calibration runs first, dumping per-corruption logits to
# a shared cache; cifar_c then reuses those logits instead of re-forwarding.
# The eval-only probes get a large batch and num_workers=0 (CIFAR-100-C is
# eager-loaded into RAM by CIFAR100CFull, no worker forking needed).
SHARE_LOGITS="$OUTDIR/.shared_logits"

run_probe knn          --batch-size 1024 --workers 2
run_probe geometry     --batch-size 1024 --workers 2
run_probe linear_probe --batch-size 1024 --workers 2 --epochs 50
run_probe calibration  --batch-size 1024 --workers 0 --share-logits-dir "$SHARE_LOGITS"
run_probe cifar_c      --batch-size 1024 --workers 0 --share-logits-dir "$SHARE_LOGITS"
run_probe adversarial  --batch-size 256 --workers 2 --max-batches 10

# Clean up shared logits cache once both probes consumed it.
rm -rf "$SHARE_LOGITS" 2>/dev/null || true

# Per-checkpoint S3 upload so partial sweep results survive spot reclaim.
# Set ``S3_RESULTS_PREFIX=s3://bucket/path`` to enable; unset = no upload.
if [[ -n "${S3_RESULTS_PREFIX:-}" ]]; then
    name=$(basename "$OUTDIR")
    aws s3 sync "$OUTDIR" "$S3_RESULTS_PREFIX/$name/" --no-progress \
        --exclude '.shared_logits/*' >/dev/null 2>&1 \
        && echo "[s3]   uploaded $name" \
        || echo "[s3]   upload failed for $name"
fi

echo "[done] $MODEL $ACT → $OUTDIR"
