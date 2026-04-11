"""
OOD / distribution shift evaluation on CIFAR-100-C.

Loads a trained checkpoint and evaluates on all 19 corruption types
at 5 severity levels. Reports per-corruption and mean corruption error (mCE).

CIFAR-100-C: 19 corruptions × 5 severities × 10000 images each.
Downloaded from zenodo automatically.

Usage:
    python experiments/eval_ood.py --checkpoint results/checkpoints/resnet20_cifar100_nelu_s42_best.pt
    python experiments/eval_ood.py --checkpoint results/checkpoints/resnet20_cifar100_gelu_s42_best.pt

    # Compare both at once
    python experiments/eval_ood.py \
        --gelu-ckpt results/checkpoints/resnet20_cifar100_gelu_s42_best.pt \
        --nelu-ckpt results/checkpoints/resnet20_cifar100_nelu_s42_best.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
# DataLoader removed: we batch directly off pre-loaded GPU tensors.
from torchvision import transforms
from tqdm import tqdm

# Path setup — relative to this file, not hardcoded.
_THIS_DIR = Path(__file__).resolve().parent          # .../experiments
_REPO_ROOT = _THIS_DIR.parent                        # .../<repo root>
sys.path.insert(0, str(_REPO_ROOT))                  # for `import nelu`
sys.path.insert(0, str(_THIS_DIR))                   # for `import main_cifar_tinyimagenet`

from nelu import NELU
from nelu.cuda_kernel import NELUCUDA
from main_cifar_tinyimagenet import build_model, CIFAR100_MEAN, CIFAR100_STD

RESULTS_DIR = _REPO_ROOT / "results"
DATA_DIR = _REPO_ROOT / "data"

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    "speckle_noise", "gaussian_blur", "spatter", "saturate",
]


def download_cifar100c(data_dir):
    """Download CIFAR-100-C from zenodo if not present."""
    cifar100c_dir = data_dir / "CIFAR-100-C"
    if cifar100c_dir.exists() and len(list(cifar100c_dir.glob("*.npy"))) >= 19:
        return cifar100c_dir

    cifar100c_dir.mkdir(parents=True, exist_ok=True)
    url = "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar"

    import tarfile
    import urllib.request

    tar_path = data_dir / "CIFAR-100-C.tar"
    if not tar_path.exists():
        print(f"Downloading CIFAR-100-C from zenodo (~700MB)...")
        urllib.request.urlretrieve(url, tar_path)
        print("Done.")

    print("Extracting...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(data_dir)
    print("Done.")

    return cifar100c_dir


def load_corruption(cifar100c_dir, corruption, severity):
    """Load one corruption at one severity. Returns (images, labels)."""
    images = np.load(cifar100c_dir / f"{corruption}.npy")
    labels = np.load(cifar100c_dir / "labels.npy")

    # Each file has 50000 images (10000 per severity, concatenated)
    start = (severity - 1) * 10000
    end = severity * 10000
    images = images[start:end]  # (10000, 32, 32, 3), uint8
    labels = labels[start:end]

    return images, labels.astype(np.int64)


def to_normalized_tensor(images, labels, device="cuda"):
    """Pre-normalize an entire (images, labels) pair on GPU once.
    CIFAR-100-C is small (10k × 32×32×3 ≈ 30 MB), fits trivially in VRAM.
    Returns: (images_gpu_NCHW, labels_gpu)."""
    mean = torch.tensor(CIFAR100_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(CIFAR100_STD,  device=device).view(1, 3, 1, 1)
    x = torch.from_numpy(images).to(device, non_blocking=True).float() / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()    # NHWC → NCHW
    x = (x - mean) / std
    y = torch.from_numpy(labels).to(device, non_blocking=True).long()
    return x, y


def load_checkpoint(ckpt_path, device="cuda"):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    act = ckpt["act"]
    dataset = ckpt.get("dataset", "cifar100")

    num_classes = 100 if "100" in dataset else 10
    img_size = 32

    model = build_model(arch, num_classes, img_size, act)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    return model, arch, act


@torch.no_grad()
def evaluate_tensor(model, x, y, batch_size=4096, device="cuda"):
    """Evaluate on a pre-loaded GPU tensor. Slices into chunks of batch_size
    so we never run out of memory even on the largest CNN (WRN-28-10)."""
    correct, total = 0, 0
    for i in range(0, x.size(0), batch_size):
        outputs = model(x[i:i+batch_size])
        correct += (outputs.argmax(1) == y[i:i+batch_size]).sum().item()
        total += min(batch_size, x.size(0) - i)
    return 100.0 * correct / total


def eval_all_corruptions(model, cifar100c_dir, device="cuda", severities=(1, 2, 3, 4, 5)):
    """Evaluate on all corruptions and severities. Returns dict."""
    results = {}
    for corruption in tqdm(CORRUPTIONS, desc="Corruptions"):
        accs = []
        for severity in severities:
            images, labels = load_corruption(cifar100c_dir, corruption, severity)
            x, y = to_normalized_tensor(images, labels, device)
            acc = evaluate_tensor(model, x, y, batch_size=4096, device=device)
            accs.append(acc)
        results[corruption] = {
            "per_severity": accs,
            "mean": float(np.mean(accs)),
        }
    results["mCE_accuracy"] = np.mean([r["mean"] for r in results.values()])
    return results


def _parse_ckpt_name(name):
    """Parse <arch>_<dataset>_<act>_s<seed> → (arch, dataset, act, seed).
    Returns None if unparsable."""
    import re
    m = re.match(r"^(.+)_(cifar10|cifar100)_(relu|gelu|nelu)_s(\d+)$", name)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), int(m.group(4))


def _aggregate_by_arch_act(raw_results):
    """Group per-checkpoint results by (arch, act), compute mean/std
    across seeds for mCE and per-corruption."""
    import math
    groups = {}   # (arch, act) → list of per-seed result dicts
    for name, r in raw_results.items():
        parsed = _parse_ckpt_name(name)
        if parsed is None:
            continue
        arch, dataset, act, seed = parsed
        key = (arch, act)
        groups.setdefault(key, []).append((seed, r))

    agg = {}
    for (arch, act), seeds in groups.items():
        mces = [r["mCE_accuracy"] for _, r in seeds]
        n = len(mces)
        mu = sum(mces) / n
        sd = math.sqrt(sum((x - mu) ** 2 for x in mces) / (n - 1)) if n > 1 else 0.0
        per_corruption = {}
        for c in CORRUPTIONS:
            vs = [r[c]["mean"] for _, r in seeds if c in r]
            if vs:
                mu_c = sum(vs) / len(vs)
                sd_c = math.sqrt(sum((x - mu_c)**2 for x in vs) / (len(vs)-1)) \
                    if len(vs) > 1 else 0.0
                per_corruption[c] = {"mean": mu_c, "std": sd_c, "n": len(vs)}
        agg[f"{arch}_{act}"] = {
            "arch": arch, "act": act,
            "mCE_accuracy_mean": mu,
            "mCE_accuracy_std": sd,
            "n_seeds": n,
            "seeds": sorted([s for s, _ in seeds]),
            "per_corruption": per_corruption,
        }
    return agg


def main():
    parser = argparse.ArgumentParser(description="OOD evaluation on CIFAR-100-C")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint to evaluate (just this one)")
    parser.add_argument("--relu-ckpt", type=str, default=None,
                        help="ReLU checkpoint for comparison")
    parser.add_argument("--gelu-ckpt", type=str, default=None,
                        help="GELU checkpoint for comparison")
    parser.add_argument("--nelu-ckpt", type=str, default=None,
                        help="NELU checkpoint for comparison")
    parser.add_argument("--arch", type=str, default=None,
                        help="Restrict auto-discovery to a single arch")
    parser.add_argument("--wandb", action="store_true",
                        help="Log per-corruption results to wandb")
    parser.add_argument("--skip-done", action="store_true", default=True,
                        help="Reuse per-ckpt cache under results/ood/")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip eval entirely; just aggregate existing "
                             "per-ckpt caches under results/ood/")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Per-ckpt cache directory for resumable eval
    ood_dir = RESULTS_DIR / "ood"
    ood_dir.mkdir(parents=True, exist_ok=True)

    # ── Aggregate-only mode: skip eval, just load caches and summarize ──
    if args.aggregate_only:
        all_results = {}
        for f in sorted(ood_dir.glob("*.json")):
            name = f.stem
            with open(f) as fp:
                all_results[name] = json.load(fp)
        if not all_results:
            print(f"No cached results in {ood_dir}")
            return
        print(f"Loaded {len(all_results)} cached per-ckpt results")
        # fall through to aggregation block (wandb disabled in agg-only)
        wandb_run = None
        goto_agg = True
    else:
        goto_agg = False

    if not goto_agg:
        # Optional wandb run
        wandb_run = None
        if args.wandb:
            try:
                import wandb
                wandb_run = wandb.init(project="nelu", group="ood_cifar100c",
                                       name=f"ood_eval_{int(__import__('time').time())}",
                                       reinit=True)
            except Exception as e:
                print(f"  WARNING: wandb init failed ({e}); continuing without wandb")
                wandb_run = None

        # Download CIFAR-100-C
        cifar100c_dir = download_cifar100c(DATA_DIR)

        checkpoints = {}
        if args.checkpoint:
            # Derive name from path so aggregation can parse arch/act/seed
            name = Path(args.checkpoint).stem.replace("_best", "")
            checkpoints[name] = args.checkpoint
        if args.relu_ckpt:
            checkpoints[Path(args.relu_ckpt).stem.replace("_best", "")] = args.relu_ckpt
        if args.gelu_ckpt:
            checkpoints[Path(args.gelu_ckpt).stem.replace("_best", "")] = args.gelu_ckpt
        if args.nelu_ckpt:
            checkpoints[Path(args.nelu_ckpt).stem.replace("_best", "")] = args.nelu_ckpt

        if not checkpoints:
            # Auto-find checkpoints
            ckpt_dir = RESULTS_DIR / "checkpoints"
            if ckpt_dir.exists():
                for f in sorted(ckpt_dir.glob("*_best.pt")):
                    name = f.stem.replace("_best", "")
                    # Only CIFAR-100 runs
                    if "_cifar100_" not in name:
                        continue
                    if args.arch and not name.startswith(args.arch + "_"):
                        continue
                    checkpoints[name] = str(f)

        if not checkpoints:
            print("No checkpoints found. Train models first with main_cifar_tinyimagenet.py")
            return

        print(f"Found {len(checkpoints)} checkpoint(s) to evaluate")
        print()

        all_results = {}
        for name, ckpt_path in checkpoints.items():
            cache_path = ood_dir / f"{name}.json"
            if args.skip_done and cache_path.exists():
                with open(cache_path) as f:
                    all_results[name] = json.load(f)
                print(f"  SKIP (cached): {name}  "
                      f"mCE_acc={all_results[name]['mCE_accuracy']:.2f}%")
                continue

            print(f"\n{'='*60}")
            print(f"  Evaluating: {name}")
            print(f"{'='*60}")

            model, arch, act = load_checkpoint(ckpt_path, device)
            results = eval_all_corruptions(model, cifar100c_dir, device)
            all_results[name] = results

            # Save per-ckpt cache immediately (resumable)
            with open(cache_path, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\n  Mean accuracy across all corruptions: "
                  f"{results['mCE_accuracy']:.2f}%")

            # Log per-corruption metrics to wandb (per checkpoint)
            if wandb_run is not None:
                log_dict = {f"ood/{name}/mCE_accuracy": results["mCE_accuracy"]}
                for corruption in CORRUPTIONS:
                    if corruption in results:
                        log_dict[f"ood/{name}/{corruption}"] = results[corruption]["mean"]
                wandb_run.log(log_dict)

            del model
            torch.cuda.empty_cache()

        # If we ran a single checkpoint (via --checkpoint), skip the
        # "aggregation across seeds" printout (meaningless for one).
        if args.checkpoint and not (args.relu_ckpt or args.gelu_ckpt or args.nelu_ckpt):
            print(f"\n  Single-checkpoint mode: result cached at {ood_dir}/{name}.json")
            if wandb_run is not None:
                wandb_run.finish()
            return

    # ── Aggregation across seeds ──────────────────────────────────
    print(f"\n{'='*76}")
    print(f"  Aggregating across seeds (grouped by arch, act)")
    print(f"{'='*76}")
    agg = _aggregate_by_arch_act(all_results)

    if agg:
        # Sort by arch, then act
        ACT_ORDER = ["relu", "gelu", "nelu"]
        keys_sorted = sorted(
            agg.keys(),
            key=lambda k: (agg[k]["arch"],
                           ACT_ORDER.index(agg[k]["act"]) if agg[k]["act"] in ACT_ORDER else 99),
        )

        print(f"\n  {'arch':<14} {'act':<6} {'n':>3}  {'mCE_acc (mean ± std)':>22}")
        print("  " + "-" * 54)
        for k in keys_sorted:
            a = agg[k]
            print(f"  {a['arch']:<14} {a['act']:<6} {a['n_seeds']:>3}  "
                  f"{a['mCE_accuracy_mean']:>16.2f} ± {a['mCE_accuracy_std']:.2f}")

        # Per-arch comparison: NELU - GELU delta
        print(f"\n  Per-arch NELU vs GELU (mCE_acc, higher = more robust):")
        archs = sorted(set(a["arch"] for a in agg.values()))
        print(f"  {'arch':<14} {'ReLU':>14} {'GELU':>14} {'NELU':>14} {'Δ(N-G)':>9}")
        print("  " + "-" * 68)
        for arch in archs:
            row = f"  {arch:<14}"
            vals = {}
            for act in ACT_ORDER:
                key = f"{arch}_{act}"
                if key in agg:
                    mu = agg[key]["mCE_accuracy_mean"]
                    sd = agg[key]["mCE_accuracy_std"]
                    row += f"  {mu:>6.2f} ± {sd:.2f}"
                    vals[act] = mu
                else:
                    row += f"  {'—':>14}"
            if "nelu" in vals and "gelu" in vals:
                row += f"  {vals['nelu'] - vals['gelu']:>+8.2f}"
            print(row)

    # ── Save raw + aggregated ─────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / "ood_cifar100c.json"
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Raw per-ckpt results: {raw_path}")

    agg_path = RESULTS_DIR / "ood_cifar100c_agg.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"  Aggregated (mean ± std): {agg_path}")

    # Final wandb summary (aggregated)
    if wandb_run is not None:
        for k, a in agg.items():
            wandb_run.log({
                f"ood_agg/{k}/mCE_mean": a["mCE_accuracy_mean"],
                f"ood_agg/{k}/mCE_std":  a["mCE_accuracy_std"],
            })
        try:
            import wandb as _wb
            cols = ["arch", "act", "n_seeds", "mCE_mean", "mCE_std"]
            table = _wb.Table(columns=cols)
            for k, a in agg.items():
                table.add_data(a["arch"], a["act"], a["n_seeds"],
                               a["mCE_accuracy_mean"], a["mCE_accuracy_std"])
            wandb_run.log({"ood_agg/summary": table})
        except Exception:
            pass
        wandb_run.finish()


if __name__ == "__main__":
    main()
