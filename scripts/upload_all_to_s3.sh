#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Upload ALL datasets to S3 — including FineWeb tokenization
#  and ImageNet-C download.
#
#  Handles limited local disk by processing one dataset at a time:
#    download/process → upload to S3 → delete local
#
#  Usage:
#      bash scripts/upload_all_to_s3.sh
# ═══════════════════════════════════════════════════════════════

set -e

S3="s3://nelu-datasets"
CONDA="conda run --no-capture-output -n resact"
DATA="./data"

echo "═══════════════════════════════════════════════════════════"
echo "  Uploading all datasets to $S3"
echo "  Local disk: $(df -h / | tail -1 | awk '{print $4}') free"
echo "═══════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────
# 1. CIFAR-10/100 (already downloaded, ~350MB)
# ─────────────────────────────────────────────────────────────

echo -e "\n[1/5] CIFAR-10/100"
if aws s3 ls "${S3}/cifar-100-python/meta" &>/dev/null; then
    echo "  Already on S3, skipping."
else
    # Download if not present
    if [ ! -d "$DATA/cifar-100-python" ]; then
        $CONDA python -c "
from torchvision import datasets
datasets.CIFAR10('$DATA', train=True, download=True)
datasets.CIFAR100('$DATA', train=True, download=True)
"
    fi
    echo "  Uploading CIFAR-10..."
    aws s3 sync "$DATA/cifar-10-batches-py" "${S3}/cifar-10-batches-py/" --quiet
    echo "  Uploading CIFAR-100..."
    aws s3 sync "$DATA/cifar-100-python" "${S3}/cifar-100-python/" --quiet
    # Keep local (small)
    echo "  Done."
fi

# ─────────────────────────────────────────────────────────────
# 2. CIFAR-100-C (~2.8GB, already downloaded)
# ─────────────────────────────────────────────────────────────

echo -e "\n[2/5] CIFAR-100-C"
if aws s3 ls "${S3}/CIFAR-100-C/labels.npy" &>/dev/null; then
    echo "  Already on S3, skipping."
else
    if [ ! -d "$DATA/CIFAR-100-C" ]; then
        echo "  Downloading from Zenodo (~700MB)..."
        wget -q --show-progress -O "$DATA/CIFAR-100-C.tar" \
            "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar"
        tar xf "$DATA/CIFAR-100-C.tar" -C "$DATA"
        rm -f "$DATA/CIFAR-100-C.tar"
    fi
    echo "  Uploading to S3..."
    aws s3 sync "$DATA/CIFAR-100-C" "${S3}/CIFAR-100-C/" --quiet
    echo "  Done."
fi

# ─────────────────────────────────────────────────────────────
# 3. FineWeb-Edu 10B tokenized (~20GB)
#    Downloads from HF, tokenizes with GPT-2 tokenizer,
#    saves as .bin, uploads to S3, then deletes local.
# ─────────────────────────────────────────────────────────────

echo -e "\n[3/5] FineWeb-Edu 10B (tokenize + upload)"
if aws s3 ls "${S3}/fineweb-edu/train.bin" &>/dev/null; then
    echo "  Already on S3, skipping."
else
    FW_DIR="$DATA/fineweb-edu"
    if [ -f "$FW_DIR/train.bin" ]; then
        echo "  Already tokenized locally."
    else
        echo "  Tokenizing FineWeb-Edu 10B (this will take 2-3 hours)..."
        echo "  Disk needed: ~20GB ($(df -h / | tail -1 | awk '{print $4}') available)"
        $CONDA python scripts/prepare_fineweb.py --output "$FW_DIR"
    fi
    echo "  Uploading to S3 (~20GB, may take a while)..."
    aws s3 cp "$FW_DIR/train.bin" "${S3}/fineweb-edu/train.bin"
    echo "  Deleting local copy to free disk..."
    rm -rf "$FW_DIR"
    echo "  Done. Freed ~20GB."
fi

# ─────────────────────────────────────────────────────────────
# 4. ImageNet-C (~18GB total, downloaded in chunks)
#    Each corruption type is ~3-4GB.
#    Download one → extract → upload to S3 → delete local.
# ─────────────────────────────────────────────────────────────

echo -e "\n[4/5] ImageNet-C (chunked download+upload)"
if aws s3 ls "${S3}/ImageNet-C/blur/" &>/dev/null; then
    echo "  Already on S3, skipping."
else
    echo "  Downloading from Zenodo in chunks (~18GB total)..."
    echo "  Peak local disk per chunk: ~4GB"
    IMNETC_DIR="$DATA/ImageNet-C"

    for corruption in blur digital extra noise weather; do
        # Check if this chunk already uploaded
        if aws s3 ls "${S3}/ImageNet-C/${corruption}/" &>/dev/null; then
            echo "  $corruption: already on S3, skipping."
            continue
        fi

        echo "  Downloading $corruption..."
        wget -q --show-progress -O "$DATA/${corruption}.tar" \
            "https://zenodo.org/records/2235448/files/${corruption}.tar"

        echo "  Extracting..."
        mkdir -p "$IMNETC_DIR"
        tar xf "$DATA/${corruption}.tar" -C "$IMNETC_DIR"
        rm -f "$DATA/${corruption}.tar"

        echo "  Uploading $corruption to S3..."
        aws s3 sync "$IMNETC_DIR/" "${S3}/ImageNet-C/" --quiet

        echo "  Cleaning local..."
        rm -rf "$IMNETC_DIR"

        echo "  $corruption done."
    done
    echo "  All ImageNet-C uploaded."
fi

# ─────────────────────────────────────────────────────────────
# 5. ImageNet-1k (user provides separately)
# ─────────────────────────────────────────────────────────────

echo -e "\n[5/5] ImageNet-1k"
N=$(aws s3 ls "${S3}/imagenet/train/" 2>/dev/null | wc -l)
if [ "$N" -gt 100 ]; then
    echo "  Found on S3 ($N class dirs)."
else
    echo "  NOT on S3."
    echo "  Upload from your local machine:"
    echo "    aws s3 sync /path/to/imagenet/train ${S3}/imagenet/train/"
    echo "    aws s3 sync /path/to/imagenet/val ${S3}/imagenet/val/"
fi

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  S3 Summary: $S3"
echo "═══════════════════════════════════════════════════════════"

for ds in cifar-10-batches-py cifar-100-python CIFAR-100-C fineweb-edu ImageNet-C imagenet; do
    EXISTS=$(aws s3 ls "${S3}/${ds}/" 2>/dev/null | head -1)
    STATUS=$( [ -n "$EXISTS" ] && echo "✓" || echo "✗" )
    SIZE=""
    if [ -n "$EXISTS" ]; then
        SIZE=$(aws s3 ls "${S3}/${ds}/" --recursive --summarize 2>/dev/null | grep "Total Size" | awk '{print $3, $4}')
    fi
    echo "  $STATUS  $ds  $SIZE"
done

echo ""
echo "  On training instance:"
echo "    # CIFAR experiments (small, fast)"
echo "    aws s3 sync ${S3}/ /data/ --exclude 'imagenet/*' --exclude 'ImageNet-C/*' --exclude 'fineweb-edu/*'"
echo ""
echo "    # LM experiments"
echo "    aws s3 cp ${S3}/fineweb-edu/train.bin /data/fineweb-edu/train.bin"
echo ""
echo "    # ImageNet experiments"
echo "    aws s3 sync ${S3}/imagenet/ /data/imagenet/"
echo ""
echo "    # ImageNet-C eval (after training)"
echo "    aws s3 sync ${S3}/ImageNet-C/ /data/ImageNet-C/"
echo ""
echo "  Disk free after cleanup: $(df -h / | tail -1 | awk '{print $4}')"
echo "═══════════════════════════════════════════════════════════"
