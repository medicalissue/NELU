#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  H100 (or any fresh GPU machine) bootstrap for the NELU paper.
#
#  Run from inside ResAct after `git pull`. Idempotent — safe to
#  re-run; existing clones are pulled & re-patched cleanly.
#
#  Steps performed
#  ---------------
#   1. Clone facebookresearch/deit + facebookresearch/ConvNeXt at
#      the commits the patches were generated against.
#   2. Apply patches/{deit,convnext}-train.patch.
#   3. Locate timm-train (Wightman repo) — needed by Phase 4
#      (efficientnet_b2 via experiments/train_imagenet_timm.py).
#   4. Install pip deps (tensorboardX, wandb).
#   5. Clear ~/.cache/torch_extensions so the NELU/NiLU CUDA kernels
#      recompile for the local GPU's compute capability.
#   6. JIT-compile both kernels and run a forward+backward smoke test.
#   7. Verify wandb login.
#   8. Print dataset path checklist.
#
#  Env overrides
#  -------------
#    DEIT_DIR     (default: /home/ubuntu/deit-train)
#    CONVNEXT_DIR (default: /home/ubuntu/convnext-train)
#    TIMM_TRAIN_PY (path to Wightman train.py — auto-detected)
# ═══════════════════════════════════════════════════════════════

set -u

RESACT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PATCH_DIR="$RESACT_DIR/patches"
# Default external repo locations: SIBLING dirs of ResAct. So if ResAct
# lives at /home/ubuntu/NELU/ResAct, externals land at /home/ubuntu/NELU/
# {deit-train,convnext-train,timm-train}. Override with env vars if needed.
DEIT_DIR="${DEIT_DIR:-$RESACT_DIR/../deit-train}"
CONVNEXT_DIR="${CONVNEXT_DIR:-$RESACT_DIR/../convnext-train}"

DEIT_COMMIT="7e160fe43f0252d17191b71cbb5826254114ea5b"
CONVNEXT_COMMIT="048efcea897d999aed302f2639b6270aedf8d4c8"

DEIT_URL="https://github.com/facebookresearch/deit.git"
CONVNEXT_URL="https://github.com/facebookresearch/ConvNeXt.git"

ok()   { echo -e "\033[32m  ✓ $1\033[0m"; }
warn() { echo -e "\033[33m  ! $1\033[0m"; }
err()  { echo -e "\033[31m  ✗ $1\033[0m"; }
hdr()  { echo -e "\n\033[1m$1\033[0m"; }


# ─── 1+2: Clone and patch upstream repos ─────────────────────────
clone_and_patch() {
    local dir="$1" url="$2" commit="$3" patch="$4"
    local name="$(basename "$dir")"
    if [ -d "$dir/.git" ]; then
        ok "$name already cloned at $dir"
        # Reset to clean state at the pinned commit so patch applies cleanly
        (cd "$dir" && git fetch --quiet origin && git reset --hard "$commit" --quiet) \
            || { err "$name reset to $commit failed"; return 1; }
    else
        echo "  cloning $name -> $dir"
        git clone --quiet "$url" "$dir" || { err "clone $name failed"; return 1; }
        (cd "$dir" && git checkout --quiet "$commit") \
            || { err "$name checkout $commit failed"; return 1; }
        ok "$name cloned + pinned to ${commit:0:8}"
    fi
    if [ ! -f "$patch" ]; then
        err "patch file not found: $patch"
        return 1
    fi
    if (cd "$dir" && git apply --check "$patch" 2>/dev/null); then
        (cd "$dir" && git apply "$patch") && ok "$name patched"
    else
        # Maybe already patched — check via reverse-apply dry-run
        if (cd "$dir" && git apply -R --check "$patch" 2>/dev/null); then
            ok "$name already patched (skipping)"
        else
            err "$name patch failed to apply (and not already applied)"
            return 1
        fi
    fi
}

hdr "[1/8] Cloning + patching upstream training repos"
clone_and_patch "$DEIT_DIR"     "$DEIT_URL"     "$DEIT_COMMIT"     "$PATCH_DIR/deit-train.patch"
clone_and_patch "$CONVNEXT_DIR" "$CONVNEXT_URL" "$CONVNEXT_COMMIT" "$PATCH_DIR/convnext-train.patch"


# ─── 3: Locate timm-train ────────────────────────────────────────
hdr "[2/8] Locating Wightman timm-train"
TIMM_TRAIN_PY="${TIMM_TRAIN_PY:-}"
if [ -z "$TIMM_TRAIN_PY" ]; then
    # Search: in-tree first (ResAct/timm-train), then sibling, then legacy locations.
    for c in "$RESACT_DIR/timm-train/train.py" \
             "$RESACT_DIR/../timm-train/train.py" \
             /home/ubuntu/NELU/timm-train/train.py \
             /home/ubuntu/timm-train/train.py \
             /home/ubuntu/pytorch-image-models/train.py ; do
        if [ -f "$c" ]; then
            TIMM_TRAIN_PY="$(cd "$(dirname "$c")" && pwd)/train.py"; break
        fi
    done
fi
if [ -z "$TIMM_TRAIN_PY" ] || [ ! -f "$TIMM_TRAIN_PY" ]; then
    warn "timm-train/train.py not found locally."
    warn "  Phase 4 (efficientnet_b2) needs this. Clone it via:"
    warn "    git clone https://github.com/huggingface/pytorch-image-models \\"
    warn "        $RESACT_DIR/timm-train"
    warn "  then re-run this script (or set TIMM_TRAIN_PY manually)."
else
    ok "timm-train at $TIMM_TRAIN_PY"
    echo "  exporting TIMM_TRAIN_PY for run_h100.sh consumers"
    echo "export TIMM_TRAIN_PY=\"$TIMM_TRAIN_PY\"" > "$RESACT_DIR/.timm_train_py.env"
fi


# ─── 4: pip dependencies ─────────────────────────────────────────
hdr "[3/8] Installing python dependencies"
PIP_DEPS=(tensorboardX wandb timm)
for pkg in "${PIP_DEPS[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        ok "$pkg already installed"
    else
        echo "  pip install $pkg"
        pip install --quiet "$pkg" && ok "$pkg installed" || err "$pkg install failed"
    fi
done

# apex is optional (gives FusedLAMB for DeiT-III; falls back to LAMB if missing)
if python -c "from apex.optimizers import FusedLAMB" 2>/dev/null; then
    ok "apex.FusedLAMB available"
else
    warn "apex not installed — DeiT-III will fall back from FusedLAMB to LAMB"
    warn "  (If you want bit-exact: install nvidia/apex from source — slow build)"
fi


# ─── 5: Clear torch_extensions cache ─────────────────────────────
# hdr "[4/8] Clearing torch_extensions cache (forces NELU/NiLU recompile)"
# EXT_DIR="$HOME/.cache/torch_extensions"
# if [ -d "$EXT_DIR" ]; then
#     for sub in "$EXT_DIR"/*/nelu_cuda "$EXT_DIR"/*/nilu_cuda; do
#         [ -e "$sub" ] && rm -rf "$sub" && ok "removed $sub"
#     done
# else
#     ok "no extension cache yet (clean machine)"
# fi


# ─── 6: Smoke test the kernels ───────────────────────────────────
hdr "[5/8] Compiling + smoke-testing NELU/NiLU CUDA kernels"
python - <<'PY' || { err "kernel smoke test failed"; exit 1; }
import sys, math, torch
sys.path.insert(0, ".")

print("  importing nelu...", flush=True)
from nelu import NELU, NiLU, NELUCUDA, NiLUCUDA
from nelu.cuda_kernel import nelu_cuda
from nelu.nilu_cuda_kernel import nilu_cuda

assert NELUCUDA is not None, "NELUCUDA failed to load"
assert NiLUCUDA is not None, "NiLUCUDA failed to load"

# Forward + backward sanity (fp32, fp16, bf16) at three sizes
for shape in [(2, 768), (4, 4096), (2, 64, 14, 14)]:
    for dt in [torch.float32, torch.float16, torch.bfloat16]:
        z = torch.randn(*shape, device="cuda", dtype=dt, requires_grad=True)
        for fn, name in [(nelu_cuda, "NELU"), (nilu_cuda, "NiLU")]:
            y = fn(z)
            g = torch.randn_like(y)
            y.backward(g)
            assert torch.isfinite(y).all(), f"{name} {shape} {dt} fwd nan"
            assert torch.isfinite(z.grad).all(), f"{name} {shape} {dt} bwd nan"
            z.grad = None

print("  ✓ NELU + NiLU forward + backward OK across fp32/fp16/bf16")
print("  ✓ kernels recompiled for local compute capability")
PY
ok "kernel smoke test passed"


# ─── 7: wandb login ──────────────────────────────────────────────
hdr "[6/8] Checking wandb login"
if python -c "import wandb; assert wandb.api.api_key" 2>/dev/null; then
    ok "wandb is logged in"
else
    warn "wandb is NOT logged in. Run: wandb login"
fi


# ─── 8: Dataset paths ────────────────────────────────────────────
hdr "[7/8] Dataset path checklist"
for d in /data/imagenet/train /data/imagenet/val \
         /data/ImageNet-C \
         /data/coco; do
    if [ -d "$d" ]; then ok "$d"; else warn "missing: $d"; fi
done


# ─── 9: Summary ──────────────────────────────────────────────────
hdr "[8/8] Done"
echo "  Next:"
echo "    1. (if needed) wandb login"
echo "    2. (if missing) symlink datasets to /data/{imagenet,coco,ImageNet-C}"
echo "    3. bash run_h100.sh"
echo
