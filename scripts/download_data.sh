#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Download all datasets needed for NELU experiments.
#  Run once before training.
#
#  Usage:
#      bash scripts/download_data.sh [DATA_DIR]
#      bash scripts/download_data.sh /data
# ═══════════════════════════════════════════════════════════════

set -e

DATA_DIR="${1:-./data}"
mkdir -p "$DATA_DIR"

echo "═══════════════════════════════════════════════════════════"
echo "  Downloading datasets to: $DATA_DIR"
echo "═══════════════════════════════════════════════════════════"

# ── CIFAR-10/100 ────────────────────────────────────────────────
# Downloaded automatically by torchvision on first use.
# Just trigger the download.

echo ""
echo "── CIFAR-10/100 ──"
python -c "
from torchvision import datasets
print('  CIFAR-10...')
datasets.CIFAR10('$DATA_DIR', train=True, download=True)
datasets.CIFAR10('$DATA_DIR', train=False, download=True)
print('  CIFAR-100...')
datasets.CIFAR100('$DATA_DIR', train=True, download=True)
datasets.CIFAR100('$DATA_DIR', train=False, download=True)
print('  Done.')
"

# ── CIFAR-100-C (corruption robustness) ─────────────────────────
# ~700MB from Zenodo

echo ""
echo "── CIFAR-100-C ──"
CIFAR100C_DIR="$DATA_DIR/CIFAR-100-C"
if [ -d "$CIFAR100C_DIR" ] && [ $(ls "$CIFAR100C_DIR"/*.npy 2>/dev/null | wc -l) -ge 19 ]; then
    echo "  Already exists, skipping."
else
    echo "  Downloading from Zenodo (~700MB)..."
    wget -q --show-progress -O "$DATA_DIR/CIFAR-100-C.tar" \
        "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar"
    echo "  Extracting..."
    tar xf "$DATA_DIR/CIFAR-100-C.tar" -C "$DATA_DIR"
    rm "$DATA_DIR/CIFAR-100-C.tar"
    echo "  Done."
fi

# ── ImageNet-1k ─────────────────────────────────────────────────
# Must be downloaded manually (requires academic access).
# Expected layout:
#   $DATA_DIR/imagenet/train/n01440764/*.JPEG
#   $DATA_DIR/imagenet/val/n01440764/*.JPEG

echo ""
echo "── ImageNet-1k ──"
IMNET_DIR="$DATA_DIR/imagenet"
if [ -d "$IMNET_DIR/train" ] && [ -d "$IMNET_DIR/val" ]; then
    N_TRAIN=$(find "$IMNET_DIR/train" -name "*.JPEG" 2>/dev/null | wc -l)
    N_VAL=$(find "$IMNET_DIR/val" -name "*.JPEG" 2>/dev/null | wc -l)
    echo "  Found: train=$N_TRAIN val=$N_VAL images"
else
    echo "  NOT FOUND. Download manually from https://image-net.org"
    echo "  Expected path: $IMNET_DIR/{train,val}/nXXXXXXXX/*.JPEG"
fi

# ── ImageNet-C (corruption robustness) ──────────────────────────
# ~18GB from Zenodo. Download only if ImageNet exists.

echo ""
echo "── ImageNet-C ──"
IMNETC_DIR="$DATA_DIR/ImageNet-C"
if [ -d "$IMNETC_DIR" ]; then
    echo "  Already exists, skipping."
elif [ -d "$IMNET_DIR/val" ]; then
    echo "  Downloading from Zenodo (~18GB). This will take a while..."
    echo "  If you want to skip, press Ctrl+C."
    mkdir -p "$IMNETC_DIR"
    # Download each corruption type separately
    for corruption in blur digital extra noise weather; do
        echo "  Downloading $corruption..."
        wget -q --show-progress -O "$DATA_DIR/${corruption}.tar" \
            "https://zenodo.org/records/2235448/files/${corruption}.tar"
        tar xf "$DATA_DIR/${corruption}.tar" -C "$IMNETC_DIR"
        rm "$DATA_DIR/${corruption}.tar"
    done
    echo "  Done."
else
    echo "  Skipping (ImageNet not found)."
fi

# ── FineWeb-Edu 10B (for GPT-2 LM experiments) ──────────────────
# ~20GB pre-tokenized, or stream from HuggingFace.

echo ""
echo "── FineWeb-Edu 10B ──"
FINEWEB_DIR="$DATA_DIR/fineweb-edu"
if [ -d "$FINEWEB_DIR" ] || [ -f "$FINEWEB_DIR/train.bin" ]; then
    echo "  Found at $FINEWEB_DIR"
else
    echo "  Option 1: Pre-tokenize (recommended for speed)"
    echo "    python scripts/prepare_fineweb.py --output $FINEWEB_DIR"
    echo ""
    echo "  Option 2: HuggingFace streaming (no download needed)"
    echo "    train_lm.py will stream if --data is not provided."
    echo "    Slower but requires no disk space."
fi

# ── Summary ─────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Summary"
echo "═══════════════════════════════════════════════════════════"
echo "  CIFAR-10/100:    $(du -sh $DATA_DIR/cifar-* 2>/dev/null | tail -1 | awk '{print $1}' || echo 'ready')"
echo "  CIFAR-100-C:     $(du -sh $CIFAR100C_DIR 2>/dev/null | awk '{print $1}' || echo 'missing')"
echo "  ImageNet:        $([ -d "$IMNET_DIR/train" ] && echo 'ready' || echo 'missing')"
echo "  ImageNet-C:      $([ -d "$IMNETC_DIR" ] && echo 'ready' || echo 'missing')"
echo "  FineWeb-Edu:     $([ -f "$FINEWEB_DIR/train.bin" ] && echo 'ready' || echo 'missing (use HF streaming)')"
echo "═══════════════════════════════════════════════════════════"
