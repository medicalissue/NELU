#!/bin/bash
# Sweep CKA between every (NELU, baseline) pair for one model + seed.
#
# Usage:
#   eval_cifar_cka.sh MODEL CKPT_DIR OUTDIR [DATA_ROOT]
#
# CKPT_DIR holds <model>-<act>-s<seed>/checkpoint.pt files. We pair NELU
# against each baseline (relu, gelu, silu) at the same seed, and NiLU
# against the same set; that's six matrices per (model, seed) pair.

set -euo pipefail

MODEL=${1:?usage: $0 MODEL CKPT_DIR OUTDIR [DATA_ROOT]}
CKPT_DIR=${2:?}
OUTDIR=${3:?}
DATA_ROOT=${4:-/data}

mkdir -p "$OUTDIR"

PAIRS=(
    "nelu relu" "nelu gelu" "nelu silu"
    "nilu relu" "nilu gelu" "nilu silu"
    "nelu nilu"
)

for SEED in 42 43 44; do
    for pair in "${PAIRS[@]}"; do
        A=${pair% *}; B=${pair#* }
        ckpt_a="$CKPT_DIR/${MODEL}-${A}-s${SEED}/checkpoint.pt"
        ckpt_b="$CKPT_DIR/${MODEL}-${B}-s${SEED}/checkpoint.pt"
        if [[ ! -s "$ckpt_a" || ! -s "$ckpt_b" ]]; then
            echo "[skip] missing ${A}/${B} for ${MODEL} s${SEED}"
            continue
        fi
        out="$OUTDIR/cka_${MODEL}_${A}_vs_${B}_s${SEED}.json"
        if [[ -s "$out" ]]; then
            echo "[skip] $out"
            continue
        fi
        echo "[cka]  ${MODEL} ${A}↔${B} s${SEED}"
        python -m eval.cifar.cka \
            --model "$MODEL" \
            --activation-a "$A" --activation-b "$B" \
            --checkpoint-a "$ckpt_a" --checkpoint-b "$ckpt_b" \
            --data-root "$DATA_ROOT" \
            --output "$out"
    done
done
