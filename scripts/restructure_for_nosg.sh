#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  Restructure local H100 results/ to match the new NELU=NoSG default.
#
#  Mirrors the S3 restructure:
#    1. Move all existing SG nelu files (results + checkpoints) to
#       results/sg_backup/
#    2. Move the NoSG WRN/ResNet-110 files from results/nosg/ to
#       the main results/ directory
#    3. Remove stale ablation_*.json files (they used the old code)
#    4. Delete CIFAR-10 result + checkpoint files (phase removed)
#
#  Run from the repo root:
#      bash scripts/restructure_for_nosg.sh
# ═══════════════════════════════════════════════════════════════════

set -u
cd "$(dirname "$0")/.."

SG_BACKUP_DIR="results/sg_backup"
mkdir -p "$SG_BACKUP_DIR/checkpoints"

have_files() {
    [ -n "$(ls -A "$1" 2>/dev/null)" ]
}

# ── 1. Backup all existing SG nelu main results ────────────────
echo "═══ 1. Backup SG nelu result.json to $SG_BACKUP_DIR ═══"
for f in results/main_*_cifar100_nelu_*.json; do
    [ -f "$f" ] || continue
    mv -v "$f" "$SG_BACKUP_DIR/" 2>/dev/null || true
done

# ── 2. Move NoSG (WRN + ResNet-110) results to main location ──
echo ""
echo "═══ 2. Promote NoSG WRN/ResNet-110 to main results/ ═══"
if [ -d "results/nosg" ]; then
    for f in results/nosg/main_*.json; do
        [ -f "$f" ] || continue
        base=$(basename "$f")
        mv -v "$f" "results/$base"
    done
    # nosg/ might still contain a checkpoints/ subdir (empty or not)
    rmdir "results/nosg/checkpoints" 2>/dev/null || true
    rmdir "results/nosg" 2>/dev/null || \
        echo "  note: results/nosg/ not empty; inspect manually"
fi

# ── 3. Backup stale ablation results ───────────────────────────
echo ""
echo "═══ 3. Backup stale ablation_*.json ═══"
for f in results/ablation_*.json; do
    [ -f "$f" ] || continue
    mv -v "$f" "$SG_BACKUP_DIR/" 2>/dev/null || true
done

# ── 4. Backup old SG nelu checkpoints ──────────────────────────
echo ""
echo "═══ 4. Backup SG nelu checkpoints to $SG_BACKUP_DIR/checkpoints ═══"
for f in results/checkpoints/*_cifar100_nelu_*.pt; do
    [ -f "$f" ] || continue
    mv "$f" "$SG_BACKUP_DIR/checkpoints/"
done
echo "  moved: $(ls "$SG_BACKUP_DIR/checkpoints/" 2>/dev/null | wc -l) files"

# ── 5. Delete leftover CIFAR-10 artefacts ──────────────────────
echo ""
echo "═══ 5. Remove CIFAR-10 (phase removed from run_h100.sh) ═══"
rm -vf results/main_*_cifar10_*.json 2>/dev/null
rm -vf results/checkpoints/*_cifar10_*.pt 2>/dev/null
rm -vf logs/cifar10_*.log 2>/dev/null

# ── 6. Sanity summary ─────────────────────────────────────────
echo ""
echo "═══ Final state ═══"
echo "results/ json files by activation:"
for a in relu gelu nelu; do
    n=$(ls results/main_*_cifar100_${a}_*.json 2>/dev/null | wc -l)
    echo "  $a: $n"
done
echo ""
echo "results/ nelu files (should be wrn28_10 + resnet110 only):"
ls results/main_*_cifar100_nelu_*.json 2>/dev/null
echo ""
echo "Backup:"
echo "  $(ls $SG_BACKUP_DIR/*.json 2>/dev/null | wc -l) json files"
echo "  $(ls $SG_BACKUP_DIR/checkpoints/*.pt 2>/dev/null | wc -l) checkpoint files"
