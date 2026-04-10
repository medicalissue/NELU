#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  Pull datasets from s3://nelu-datasets/ into /data/ and symlink into
#  the repo's ./data directory.
#
#  Idempotent: skips already-present directories. Safe to re-run.
#
#  Usage:
#      bash scripts/sync_from_s3.sh                 # default: cifar + c100c
#      bash scripts/sync_from_s3.sh imagenet        # one extra
#      bash scripts/sync_from_s3.sh fineweb imagenet imagenet-c
#      bash scripts/sync_from_s3.sh all             # everything
#
#  Valid names: cifar10 cifar100 cifar100c imagenet imagenet-c fineweb all
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

S3_BUCKET="${S3_BUCKET:-s3://nelu-datasets}"
DATA_DIR="${DATA_DIR:-/data}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$DATA_DIR" "$REPO_ROOT/data"

declare -A PATHS=(
    [cifar10]="cifar-10-batches-py"
    [cifar100]="cifar-100-python"
    [cifar100c]="CIFAR-100-C"
    [imagenet]="imagenet"
    [imagenet-c]="ImageNet-C"
    [fineweb]="fineweb-edu"
)

ALL_KEYS=(cifar10 cifar100 cifar100c imagenet imagenet-c fineweb)

# Default when no args: just cifar + c100c
if [ $# -eq 0 ]; then
    set -- cifar10 cifar100 cifar100c
fi

# Expand "all"
TARGETS=()
for arg in "$@"; do
    if [ "$arg" = "all" ]; then
        TARGETS=("${ALL_KEYS[@]}")
        break
    fi
    TARGETS+=("$arg")
done

pull() {
    local key="$1"
    local sub="${PATHS[$key]:-}"
    if [ -z "$sub" ]; then
        echo "  UNKNOWN: $key" >&2
        return 1
    fi
    local dest="$DATA_DIR/$sub"
    if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "  OK    $sub  (present at $dest)"
    else
        echo "  SYNC  $sub  →  $dest"
        mkdir -p "$dest"
        if command -v s5cmd >/dev/null 2>&1; then
            s5cmd --numworkers 64 cp --concurrency 8 \
                "$S3_BUCKET/$sub/*" "$dest/" 2>&1 | tail -3
        else
            aws s3 sync "$S3_BUCKET/$sub" "$dest" --only-show-errors --no-progress
        fi
        echo "        size: $(du -sh "$dest" | cut -f1)"
    fi
    ln -snf "$dest" "$REPO_ROOT/data/$sub"
}

echo "S3 : $S3_BUCKET"
echo "dst: $DATA_DIR"
echo "targets: ${TARGETS[*]}"
echo ""

for k in "${TARGETS[@]}"; do
    pull "$k"
done

echo ""
df -h "$DATA_DIR"
