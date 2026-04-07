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
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, "/home/ubuntu/ResAct")
from nelu import NELU
from nelu.cuda_kernel import NELUCUDA

# Import model builders from main script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_cifar_tinyimagenet import build_model, CIFAR100_MEAN, CIFAR100_STD

RESULTS_DIR = Path("/home/ubuntu/ResAct/results")
DATA_DIR = Path("/home/ubuntu/ResAct/data")

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


def make_loader(images, labels, batch_size=128):
    """Convert numpy arrays to normalized DataLoader."""
    # Normalize same as training
    mean = np.array(CIFAR100_MEAN).reshape(1, 1, 1, 3)
    std = np.array(CIFAR100_STD).reshape(1, 1, 1, 3)

    images = images.astype(np.float32) / 255.0
    images = (images - mean) / std
    images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW

    dataset = TensorDataset(
        torch.tensor(images, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=2, pin_memory=True)


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
def evaluate(model, loader, device="cuda"):
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def eval_all_corruptions(model, cifar100c_dir, device="cuda", severities=(1, 2, 3, 4, 5)):
    """Evaluate on all corruptions and severities. Returns dict."""
    results = {}
    for corruption in tqdm(CORRUPTIONS, desc="Corruptions"):
        accs = []
        for severity in severities:
            images, labels = load_corruption(cifar100c_dir, corruption, severity)
            loader = make_loader(images, labels)
            acc = evaluate(model, loader, device)
            accs.append(acc)
        results[corruption] = {
            "per_severity": accs,
            "mean": np.mean(accs),
        }
    results["mCE_accuracy"] = np.mean([r["mean"] for r in results.values()])
    return results


def main():
    parser = argparse.ArgumentParser(description="OOD evaluation on CIFAR-100-C")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint to evaluate")
    parser.add_argument("--relu-ckpt", type=str, default=None,
                        help="ReLU checkpoint for comparison")
    parser.add_argument("--gelu-ckpt", type=str, default=None,
                        help="GELU checkpoint for comparison")
    parser.add_argument("--nelu-ckpt", type=str, default=None,
                        help="NELU checkpoint for comparison")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download CIFAR-100-C
    cifar100c_dir = download_cifar100c(DATA_DIR)

    checkpoints = {}
    if args.checkpoint:
        checkpoints["model"] = args.checkpoint
    if args.relu_ckpt:
        checkpoints["ReLU"] = args.relu_ckpt
    if args.gelu_ckpt:
        checkpoints["GELU"] = args.gelu_ckpt
    if args.nelu_ckpt:
        checkpoints["NELU"] = args.nelu_ckpt

    if not checkpoints:
        # Auto-find checkpoints
        ckpt_dir = RESULTS_DIR / "checkpoints"
        if ckpt_dir.exists():
            for f in sorted(ckpt_dir.glob("*_best.pt")):
                name = f.stem.replace("_best", "")
                checkpoints[name] = str(f)

    if not checkpoints:
        print("No checkpoints found. Train models first with main_cifar_tinyimagenet.py")
        return

    all_results = {}
    for name, ckpt_path in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating: {name}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        model, arch, act = load_checkpoint(ckpt_path, device)
        results = eval_all_corruptions(model, cifar100c_dir, device)
        all_results[name] = results

        print(f"\n  Mean accuracy across all corruptions: {results['mCE_accuracy']:.2f}%")
        del model
        torch.cuda.empty_cache()

    # Print comparison table
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  COMPARISON")
        print(f"{'='*60}")
        names = list(all_results.keys())
        header = f"  {'Corruption':<22}"
        for n in names:
            header += f" {n:>10}"
        if len(names) == 2:
            header += f" {'Delta':>8}"
        print(header)
        print("  " + "-" * (24 + 12 * len(names) + (10 if len(names) == 2 else 0)))

        for corruption in CORRUPTIONS:
            line = f"  {corruption:<22}"
            vals = []
            for n in names:
                v = all_results[n][corruption]["mean"]
                line += f" {v:>9.2f}%"
                vals.append(v)
            if len(vals) == 2:
                delta = vals[1] - vals[0]
                line += f" {delta:>+7.2f}%"
            print(line)

        print("  " + "-" * (24 + 12 * len(names) + (10 if len(names) == 2 else 0)))
        line = f"  {'MEAN':<22}"
        vals = []
        for n in names:
            v = all_results[n]["mCE_accuracy"]
            line += f" {v:>9.2f}%"
            vals.append(v)
        if len(vals) == 2:
            delta = vals[1] - vals[0]
            line += f" {delta:>+7.2f}%"
        print(line)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "ood_cifar100c.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
