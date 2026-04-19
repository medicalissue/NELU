#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Prepare datasets required by the NELU experiments.
#
#  Supports two modes:
#    1. Local target path: hydrate /data (or any local directory)
#       from S3 when possible, otherwise download the public
#       robustness benchmarks directly.
#    2. S3 target prefix: copy datasets into another bucket/prefix.
#
#  Datasets covered:
#    - ImageNet-1k
#    - CIFAR-100
#    - CIFAR-100-C
#    - ImageNet-C
#    - ImageNet-A / ImageNet-R / ImageNet-O
#
#  Usage:
#      bash scripts/download_data.sh [--env-file .env] /data
#      bash scripts/download_data.sh [--env-file .env] s3://my-bucket/nelu
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/infra/aws_common.sh"

usage() {
    echo "Usage: $0 [--env-file FILE] [TARGET]"
    echo ""
    echo "  TARGET defaults to ./data"
    echo "  TARGET can be a local path or an s3:// prefix"
    echo ""
    echo "  If a local TARGET is used, the script prefers DATA_SOURCE_S3"
    echo "  (or S3_BUCKET) as the source of truth when available."
}

ENV_FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --env-file)
            if [ $# -lt 2 ]; then
                echo "ERROR: --env-file requires a path" >&2
                exit 1
            fi
            ENV_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

load_env_file "$ENV_FILE" "$REPO_ROOT"

TARGET="${1:-./data}"
IS_S3=false
[[ "$TARGET" == s3://* ]] && IS_S3=true

DATA_SOURCE_S3="${DATA_SOURCE_S3:-${S3_BUCKET:-s3://nelu-datasets}}"
AWS_OK=false
if command -v aws >/dev/null 2>&1; then
    AWS_OK=true
fi
S5CMD_OK=false
if command -v s5cmd >/dev/null 2>&1; then
    S5CMD_OK=true
fi

if $IS_S3; then
    LOCAL="/tmp/nelu_dataset_staging"
    S3_TARGET="${TARGET%/}"
    echo "Mode: S3 target"
    echo "  Target: $S3_TARGET"
    echo "  Source: $DATA_SOURCE_S3"
    echo "  Staging: $LOCAL"
else
    LOCAL="$TARGET"
    S3_TARGET=""
    echo "Mode: local target"
    echo "  Target: $LOCAL"
    echo "  Source: $DATA_SOURCE_S3"
fi

mkdir -p "$LOCAL"

local_dir_populated() {
    local dir="$1"
    [ -d "$dir" ] && find "$dir" -mindepth 1 -maxdepth 1 | read -r _
}

s3_prefix_exists() {
    local uri="$1"
    $AWS_OK || return 1
    aws s3 ls "$uri" >/dev/null 2>&1
}

s3_sync() {
    local src="$1"
    local dest="$2"
    if $S5CMD_OK; then
        s5cmd sync --show-progress "${src%/}/*" "${dest%/}/"
    else
        aws s3 sync "$src" "$dest" --no-progress
    fi
}

sync_dir_to_s3() {
    local src_dir="$1"
    local rel="$2"
    if ! $IS_S3; then
        return 0
    fi
    echo "  Uploading ${rel} -> ${S3_TARGET}/${rel}/"
    s3_sync "$src_dir/" "${S3_TARGET}/${rel}/"
    rm -rf "$src_dir"
}

sync_dir_from_source_s3() {
    local rel="$1"
    local dest="$2"
    local src_uri="${DATA_SOURCE_S3%/}/${rel}/"

    if ! s3_prefix_exists "$src_uri"; then
        return 1
    fi

    echo "  Syncing ${rel} from ${src_uri}"
    mkdir -p "$dest"
    s3_sync "$src_uri" "$dest/"
}

copy_dir_s3_to_s3() {
    local rel="$1"
    local src_uri="${DATA_SOURCE_S3%/}/${rel}/"
    local dest_uri="${S3_TARGET%/}/${rel}/"

    if ! $IS_S3; then
        return 1
    fi
    if [ "${DATA_SOURCE_S3%/}" = "${S3_TARGET%/}" ]; then
        return 1
    fi
    if ! s3_prefix_exists "$src_uri"; then
        return 1
    fi

    echo "  Copying ${rel} from ${src_uri} to ${dest_uri}"
    s3_sync "$src_uri" "$dest_uri/"
}

download_and_extract_tar() {
    local url="$1"
    local archive="$2"
    local extract_dir="$3"
    wget -q --show-progress -O "$archive" "$url"
    tar xf "$archive" -C "$extract_dir"
    rm -f "$archive"
}

ensure_cifar100() {
    local rel="cifar-100-python"
    local dest="${LOCAL}/${rel}"

    echo ""
    echo "── CIFAR-100 ──"
    if $IS_S3 && s3_prefix_exists "${S3_TARGET}/${rel}/"; then
        echo "  Already present on target S3."
        return 0
    fi
    if ! $IS_S3 && local_dir_populated "$dest"; then
        echo "  Already present locally."
        return 0
    fi
    if copy_dir_s3_to_s3 "$rel"; then
        return 0
    fi
    if sync_dir_from_source_s3 "$rel" "$dest"; then
        sync_dir_to_s3 "$dest" "$rel"
        return 0
    fi

    echo "  Downloading from torchvision..."
    LOCAL_PATH="$LOCAL" python - <<'PY'
import os
from torchvision import datasets
root = os.environ["LOCAL_PATH"]
datasets.CIFAR100(root, train=True, download=True)
datasets.CIFAR100(root, train=False, download=True)
PY
    sync_dir_to_s3 "$dest" "$rel"
}

ensure_cifar100c() {
    local rel="CIFAR-100-C"
    local dest="${LOCAL}/${rel}"

    echo ""
    echo "── CIFAR-100-C ──"
    if $IS_S3 && s3_prefix_exists "${S3_TARGET}/${rel}/"; then
        echo "  Already present on target S3."
        return 0
    fi
    if ! $IS_S3 && [ -f "${dest}/labels.npy" ]; then
        echo "  Already present locally."
        return 0
    fi
    if copy_dir_s3_to_s3 "$rel"; then
        return 0
    fi
    if sync_dir_from_source_s3 "$rel" "$dest"; then
        sync_dir_to_s3 "$dest" "$rel"
        return 0
    fi

    echo "  Downloading from Zenodo..."
    download_and_extract_tar \
        "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar" \
        "${LOCAL}/CIFAR-100-C.tar" \
        "$LOCAL"
    sync_dir_to_s3 "$dest" "$rel"
}

ensure_imagenet() {
    local rel="imagenet"
    local dest="${LOCAL}/${rel}"

    echo ""
    echo "── ImageNet-1k ──"
    if $IS_S3 && s3_prefix_exists "${S3_TARGET}/${rel}/train/"; then
        echo "  Already present on target S3."
        return 0
    fi
    if ! $IS_S3 && [ -d "${dest}/train" ] && [ -d "${dest}/val" ]; then
        echo "  Already present locally."
        return 0
    fi
    if copy_dir_s3_to_s3 "$rel"; then
        return 0
    fi
    if sync_dir_from_source_s3 "$rel" "$dest"; then
        sync_dir_to_s3 "$dest" "$rel"
        return 0
    fi

    echo "  MISSING: ImageNet-1k was not found in ${DATA_SOURCE_S3}."
    echo "  Upload it manually:"
    if $IS_S3; then
        echo "    aws s3 sync /path/to/imagenet ${S3_TARGET}/imagenet/"
    else
        echo "    aws s3 sync ${DATA_SOURCE_S3}/imagenet/ ${dest}/"
    fi
    return 1
}

ensure_imagenet_c() {
    local rel="ImageNet-C"
    local dest="${LOCAL}/${rel}"

    echo ""
    echo "── ImageNet-C ──"
    if $IS_S3 && s3_prefix_exists "${S3_TARGET}/${rel}/"; then
        echo "  Already present on target S3."
        return 0
    fi
    if ! $IS_S3 && local_dir_populated "$dest"; then
        echo "  Already present locally."
        return 0
    fi
    if copy_dir_s3_to_s3 "$rel"; then
        return 0
    fi
    if sync_dir_from_source_s3 "$rel" "$dest"; then
        sync_dir_to_s3 "$dest" "$rel"
        return 0
    fi

    echo "  Downloading from Zenodo..."
    mkdir -p "$dest"
    for corruption in blur digital extra noise weather; do
        download_and_extract_tar \
            "https://zenodo.org/records/2235448/files/${corruption}.tar" \
            "${LOCAL}/${corruption}.tar" \
            "$dest"
    done
    sync_dir_to_s3 "$dest" "$rel"
}

ensure_public_imagenet_variant() {
    local rel="$1"
    local url="$2"
    local dest="${LOCAL}/${rel}"

    echo ""
    echo "── ${rel} ──"
    if $IS_S3 && s3_prefix_exists "${S3_TARGET}/${rel}/"; then
        echo "  Already present on target S3."
        return 0
    fi
    if ! $IS_S3 && local_dir_populated "$dest"; then
        echo "  Already present locally."
        return 0
    fi
    if copy_dir_s3_to_s3 "$rel"; then
        return 0
    fi
    if sync_dir_from_source_s3 "$rel" "$dest"; then
        sync_dir_to_s3 "$dest" "$rel"
        return 0
    fi

    echo "  Downloading ${rel}..."
    download_and_extract_tar "$url" "${LOCAL}/${rel}.tar" "$LOCAL"
    sync_dir_to_s3 "$dest" "$rel"
}

dataset_status_local() {
    local rel="$1"
    if local_dir_populated "${LOCAL}/${rel}"; then
        printf 'OK'
    else
        printf 'MISSING'
    fi
}

dataset_status_s3() {
    local rel="$1"
    if s3_prefix_exists "${S3_TARGET}/${rel}/"; then
        printf 'OK'
    else
        printf 'MISSING'
    fi
}

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Preparing NELU experiment datasets"
echo "═══════════════════════════════════════════════════════════"

MISSING_REQUIRED=0

ensure_cifar100
ensure_cifar100c
if ! ensure_imagenet; then
    MISSING_REQUIRED=1
fi
ensure_imagenet_c
ensure_public_imagenet_variant "imagenet-a" "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar" "n01498041"
ensure_public_imagenet_variant "imagenet-r" "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar" "n01443537"
ensure_public_imagenet_variant "imagenet-o" "https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar" "n01773157"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Summary"
echo "═══════════════════════════════════════════════════════════"

if $IS_S3; then
    echo "  Target: $S3_TARGET"
    for ds in imagenet cifar-100-python CIFAR-100-C ImageNet-C imagenet-a imagenet-r imagenet-o; do
        echo "  $(dataset_status_s3 "$ds")  $ds"
    done
else
    echo "  Target: $LOCAL"
    echo "  $(dataset_status_local "imagenet")  imagenet"
    echo "  $(dataset_status_local "cifar-100-python")  cifar-100-python"
    echo "  $(dataset_status_local "CIFAR-100-C")  CIFAR-100-C"
    echo "  $(dataset_status_local "ImageNet-C")  ImageNet-C"
    echo "  $(dataset_status_local "imagenet-a")  imagenet-a"
    echo "  $(dataset_status_local "imagenet-r")  imagenet-r"
    echo "  $(dataset_status_local "imagenet-o")  imagenet-o"
fi

echo "═══════════════════════════════════════════════════════════"

if $IS_S3; then
    rm -rf "$LOCAL" 2>/dev/null || true
fi

if [ "$MISSING_REQUIRED" -ne 0 ]; then
    echo "ERROR: required ImageNet-1k data is still missing." >&2
    exit 1
fi
