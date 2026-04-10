#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Download all datasets — to local dir or S3.
#
#  Usage:
#      bash scripts/download_data.sh                      # local ./data
#      bash scripts/download_data.sh /data                 # local /data
#      bash scripts/download_data.sh s3://my-bucket/data   # direct to S3
# ═══════════════════════════════════════════════════════════════

set -e

TARGET="${1:-./data}"
IS_S3=false
[[ "$TARGET" == s3://* ]] && IS_S3=true

if $IS_S3; then
    LOCAL="/tmp/nelu_staging"
    S3_TARGET="$TARGET"
    echo "Mode: stream to S3 (minimal local disk)"
    echo "  S3: $S3_TARGET"
    echo "  Staging: $LOCAL (temp, cleaned per-dataset)"
else
    LOCAL="$TARGET"
    S3_TARGET=""
    echo "Mode: local only"
    echo "  Path: $LOCAL"
fi

mkdir -p "$LOCAL"

# Upload to S3 then delete local to save disk
sync_to_s3() {
    if ! $IS_S3; then return; fi
    if [ -d "$1" ]; then
        echo "  → Uploading to ${S3_TARGET}/$2/"
        aws s3 sync "$1" "${S3_TARGET}/$2/" --quiet
        echo "  → Cleaning local staging..."
        rm -rf "$1"
    elif [ -f "$1" ]; then
        echo "  → Uploading to ${S3_TARGET}/$2"
        aws s3 cp "$1" "${S3_TARGET}/$2" --quiet
        rm -f "$1"
    fi
}

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Downloading datasets"
echo "═══════════════════════════════════════════════════════════"

# ── CIFAR-10/100 ────────────────────────────────────────────────

echo -e "\n── CIFAR-10/100 ──"
if $IS_S3 && aws s3 ls "${S3_TARGET}/cifar-100-python/" &>/dev/null; then
    echo "  Already on S3, skipping."
else
    python -c "
from torchvision import datasets
print('  CIFAR-10...'); datasets.CIFAR10('$LOCAL', train=True, download=True)
print('  CIFAR-100...'); datasets.CIFAR100('$LOCAL', train=True, download=True)
print('  Done.')
"
    sync_to_s3 "$LOCAL/cifar-10-batches-py" "cifar-10-batches-py"
    sync_to_s3 "$LOCAL/cifar-100-python" "cifar-100-python"
fi

# ── CIFAR-100-C ─────────────────────────────────────────────────

echo -e "\n── CIFAR-100-C ──"
C100C="$LOCAL/CIFAR-100-C"
if $IS_S3 && aws s3 ls "${S3_TARGET}/CIFAR-100-C/labels.npy" &>/dev/null; then
    echo "  Already on S3, skipping."
elif [ -d "$C100C" ] && [ $(ls "$C100C"/*.npy 2>/dev/null | wc -l) -ge 19 ]; then
    echo "  Already exists locally."
    sync_to_s3 "$C100C" "CIFAR-100-C"
else
    echo "  Downloading from Zenodo (~700MB)..."
    wget -q --show-progress -O "$LOCAL/CIFAR-100-C.tar" \
        "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar"
    tar xf "$LOCAL/CIFAR-100-C.tar" -C "$LOCAL"
    rm "$LOCAL/CIFAR-100-C.tar"
    sync_to_s3 "$C100C" "CIFAR-100-C"
    echo "  Done."
fi

# ── ImageNet-1k ─────────────────────────────────────────────────

echo -e "\n── ImageNet-1k ──"
IMNET="$LOCAL/imagenet"
if $IS_S3; then
    N=$(aws s3 ls "${S3_TARGET}/imagenet/train/" 2>/dev/null | wc -l)
    if [ "$N" -gt 100 ]; then
        echo "  Found on S3 ($N class dirs)."
    else
        echo "  NOT on S3."
        echo "  Upload manually: aws s3 sync /path/to/imagenet ${S3_TARGET}/imagenet/"
    fi
elif [ -d "$IMNET/train" ]; then
    echo "  Found locally."
else
    echo "  NOT FOUND. Download from https://image-net.org"
    echo "  Then: aws s3 sync /path/to/imagenet ${S3_TARGET}/imagenet/"
fi

# ── ImageNet-C ──────────────────────────────────────────────────

echo -e "\n── ImageNet-C ──"
IMNETC="$LOCAL/ImageNet-C"
if $IS_S3 && aws s3 ls "${S3_TARGET}/ImageNet-C/" &>/dev/null; then
    echo "  Already on S3, skipping."
elif [ -d "$IMNETC" ]; then
    echo "  Already exists locally."
    sync_to_s3 "$IMNETC" "ImageNet-C"
else
    echo "  Downloading from Zenodo (~18GB)..."
    echo "  Press Ctrl+C to skip."
    mkdir -p "$IMNETC"
    for corruption in blur digital extra noise weather; do
        echo "  $corruption..."
        wget -q --show-progress -O "$LOCAL/${corruption}.tar" \
            "https://zenodo.org/records/2235448/files/${corruption}.tar"
        tar xf "$LOCAL/${corruption}.tar" -C "$IMNETC"
        rm -f "$LOCAL/${corruption}.tar"
        # S3: upload this chunk then free disk
        if $IS_S3; then
            sync_to_s3 "$IMNETC" "ImageNet-C"
            mkdir -p "$IMNETC"  # recreate for next chunk
        fi
    done
    echo "  Done."
fi

# ── FineWeb-Edu 10B ─────────────────────────────────────────────

echo -e "\n── FineWeb-Edu 10B ──"
FW="$LOCAL/fineweb-edu"
if $IS_S3 && aws s3 ls "${S3_TARGET}/fineweb-edu/train.bin" &>/dev/null; then
    echo "  Already on S3."
elif [ -f "$FW/train.bin" ]; then
    echo "  Found locally."
    sync_to_s3 "$FW/train.bin" "fineweb-edu/train.bin"
else
    echo "  Option 1: Pre-tokenize then upload"
    echo "    python scripts/prepare_fineweb.py --output $FW"
    if $IS_S3; then
        echo "    aws s3 cp $FW/train.bin ${S3_TARGET}/fineweb-edu/train.bin"
    fi
    echo ""
    echo "  Option 2: HuggingFace streaming (no download)"
    echo "    train_lm.py streams from HF if --data is not set."
fi

# ── Summary ─────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Summary"
echo "═══════════════════════════════════════════════════════════"

if $IS_S3; then
    echo "  Target: $S3_TARGET"
    for ds in cifar-10-batches-py cifar-100-python CIFAR-100-C imagenet ImageNet-C fineweb-edu; do
        EXISTS=$(aws s3 ls "${S3_TARGET}/${ds}/" 2>/dev/null | head -1)
        STATUS=$([ -n "$EXISTS" ] && echo "✓" || echo "✗")
        echo "  $STATUS  $ds"
    done
    echo ""
    echo "  On training instance, download from S3:"
    echo "    aws s3 sync ${S3_TARGET}/ /data/ --exclude 'ImageNet-C/*'"
else
    echo "  Target: $LOCAL"
    for ds in cifar-10-batches-py cifar-100-python CIFAR-100-C imagenet ImageNet-C fineweb-edu; do
        STATUS=$([ -e "$LOCAL/$ds" ] && echo "✓" || echo "✗")
        echo "  $STATUS  $ds"
    done
fi
echo "═══════════════════════════════════════════════════════════"

# Clean staging if S3 mode (already cleaned per-dataset, just remove dir)
if $IS_S3; then
    rm -rf "$LOCAL" 2>/dev/null
    echo "  Local staging cleaned."
fi
