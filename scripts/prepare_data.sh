#!/usr/bin/env bash
# Download and extract every dataset needed by the paper's experiments.
#
# Usage:
#     bash scripts/prepare_data.sh [TARGET]
#
# TARGET defaults to /data. The final layout is::
#
#     TARGET/
#         imagenet/{train,val}/
#         cifar-100-python/
#         CIFAR-100-C/
#         ImageNet-C/
#         imagenet-a/
#         imagenet-r/
#         imagenet-o/
#
# ImageNet-1k itself is *not* redistributable; only the train/val tarballs
# from the ILSVRC-2012 release are supported and the user must download
# them manually from https://image-net.org/ before running this script.
# Every other dataset is fetched from the authors' public download URL.

set -euo pipefail

TARGET="${1:-/data}"
mkdir -p "$TARGET"

echo "[prepare_data] target = ${TARGET}"

download_and_extract() {
    local url="$1"
    local archive="$2"
    local dest="$3"
    mkdir -p "$dest"
    if [ ! -f "$archive" ]; then
        echo "  downloading $(basename "$archive")"
        curl -L -o "$archive" "$url"
    fi
    echo "  extracting $(basename "$archive")"
    tar xf "$archive" -C "$dest"
    rm -f "$archive"
}

populated() {
    [ -d "$1" ] && [ "$(ls -A "$1" 2>/dev/null)" ]
}

# ── CIFAR-100 (torchvision picks up this layout automatically) ───────
CIFAR100_DIR="${TARGET}/cifar-100-python"
if ! populated "$CIFAR100_DIR"; then
    echo
    echo "── CIFAR-100 ──"
    download_and_extract \
        "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz" \
        "${TARGET}/cifar-100-python.tar.gz" \
        "${TARGET}"
fi

# ── CIFAR-100-C (Hendrycks) ────────────────────────────────────────
CIFAR100C_DIR="${TARGET}/CIFAR-100-C"
if ! populated "$CIFAR100C_DIR"; then
    echo
    echo "── CIFAR-100-C ──"
    download_and_extract \
        "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar" \
        "${TARGET}/CIFAR-100-C.tar" \
        "${TARGET}"
fi

# ── ImageNet-C (Hendrycks, 5 tarballs) ─────────────────────────────
IMAGENETC_DIR="${TARGET}/ImageNet-C"
if ! populated "$IMAGENETC_DIR"; then
    echo
    echo "── ImageNet-C ──"
    mkdir -p "$IMAGENETC_DIR"
    for group in blur digital extra noise weather; do
        download_and_extract \
            "https://zenodo.org/records/2235448/files/${group}.tar" \
            "${TARGET}/${group}.tar" \
            "$IMAGENETC_DIR"
    done
fi

# ── ImageNet-A / R / O (Hendrycks) ─────────────────────────────────
for variant in imagenet-a imagenet-r imagenet-o; do
    variant_dir="${TARGET}/${variant}"
    if populated "$variant_dir"; then
        continue
    fi
    echo
    echo "── ${variant} ──"
    download_and_extract \
        "https://people.eecs.berkeley.edu/~hendrycks/${variant}.tar" \
        "${TARGET}/${variant}.tar" \
        "${TARGET}"
done

echo
echo "[prepare_data] summary:"
for d in imagenet cifar-100-python CIFAR-100-C ImageNet-C imagenet-a imagenet-r imagenet-o; do
    if populated "${TARGET}/${d}"; then
        printf "  OK       %s\n" "$d"
    else
        printf "  MISSING  %s\n" "$d"
    fi
done
