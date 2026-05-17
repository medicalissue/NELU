#!/usr/bin/env bash
# Clone the official GEM benchmark and overlay our two patch files so a
# worker can run the NELU/NiLU small-data campaign with GEM's exact HPO
# / training / evaluation machinery.
#
# Reviewer-defensibility: GEM is used verbatim; we ONLY add two files
# (a pipeline that swaps the activation, and a CIFAR-100 N-grid dataset
# that mirrors GEM's own ciFAIR-10 split pattern). Nothing in GEM is
# modified.
#
# Usage:
#   bash third_party/gem_patches/setup_gem.sh <gem_dir> <nelu_repo_dir>
# After this, run GEM with:
#   PYTHONPATH=<nelu_repo_dir>:<gem_dir> python <gem_dir>/scripts/full_train.py ...
set -euo pipefail

GEM_DIR="${1:?gem_dir required}"
NELU_REPO="${2:?nelu_repo_dir required}"
GEM_REF="${GEM_REF:-master}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$GEM_DIR/.git" ]]; then
    echo "[setup_gem] cloning GEM into $GEM_DIR"
    git clone --depth 1 --branch "$GEM_REF" \
        https://github.com/lorenzobrigato/gem.git "$GEM_DIR"
else
    echo "[setup_gem] GEM already present at $GEM_DIR"
fi

# Overlay our two additive files (never touch existing GEM files).
cp "$PATCH_DIR/pipelines_nelu.py"  "$GEM_DIR/gem/pipelines/nelu.py"
cp "$PATCH_DIR/datasets_cifar100.py" "$GEM_DIR/gem/datasets/cifar100.py"
echo "[setup_gem] overlaid gem/pipelines/nelu.py + gem/datasets/cifar100.py"

# Sanity: GEM's loader is py<3.12 (uses importlib find_module). Warn if
# the interpreter is too new (workers run py3.10 venv — fine).
PYV=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')
case "$PYV" in
    3.7|3.8|3.9|3.10|3.11) echo "[setup_gem] python $PYV OK for GEM" ;;
    *) echo "[setup_gem] WARNING: python $PYV — GEM loader needs <3.12" >&2 ;;
esac

echo "[setup_gem] done. NELU repo (gate_norm) = $NELU_REPO"
echo "[setup_gem] run with: PYTHONPATH=$NELU_REPO:$GEM_DIR python $GEM_DIR/scripts/full_train.py ..."
