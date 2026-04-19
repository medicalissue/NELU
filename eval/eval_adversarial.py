"""Adversarial robustness evaluation using AutoAttack.

Evaluates model robustness under Linf-bounded adversarial perturbations
using AutoAttack (Croce & Hein, 2020), the standard benchmark for
adversarial robustness.

Usage:
    python eval/eval_adversarial.py \
        --model convnext_tiny --activation nelu \
        --checkpoint results/imagenet_convnext_tiny_nelu/checkpoint-best.pth \
        --data-path /data/imagenet/val \
        --output results/adversarial/convnext_tiny_nelu.json \
        --eps 4/255 --n-samples 5000

Requires: pip install autoattack
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import timm

from nelu import NELU, NiLU
from train.act_swap import replace_activation


def build_model(model_name, activation, num_classes=1000):
    """Create a timm model with activation swap applied."""
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    if activation == "nelu":
        n = replace_activation(model, nn.GELU, NELU)
        if n == 0:
            n = replace_activation(model, nn.ReLU, NELU)
        print(f"Replaced {n} activations -> NELU")
    elif activation == "nilu":
        n = replace_activation(model, nn.SiLU, NiLU)
        if n == 0:
            n = replace_activation(model, nn.ReLU, NiLU)
        print(f"Replaced {n} activations -> NiLU")

    return model


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded checkpoint from {checkpoint_path}")


def parse_eps(eps_str):
    """Parse epsilon string like '4/255' or '0.03' to float."""
    if "/" in eps_str:
        num, den = eps_str.split("/")
        return float(num) / float(den)
    return float(eps_str)


@torch.no_grad()
def evaluate_clean(model, loader, device):
    """Evaluate clean accuracy on the given data loader."""
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)
    return 100.0 * correct / total


def parse_args():
    p = argparse.ArgumentParser(description="Adversarial robustness evaluation with AutoAttack")
    p.add_argument("--model", type=str, required=True, help="timm model name")
    p.add_argument("--activation", type=str, default="gelu",
                   choices=["relu", "gelu", "silu", "nelu", "nilu"])
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    p.add_argument("--data-path", type=str, required=True,
                   help="Path to ImageNet validation set")
    p.add_argument("--output", type=str, required=True, help="Output JSON path")
    p.add_argument("--eps", type=str, default="4/255",
                   help="Perturbation budget (e.g. '4/255' or '0.03')")
    p.add_argument("--norm", type=str, default="Linf",
                   choices=["Linf", "L2"], help="Threat model norm")
    p.add_argument("--n-samples", type=int, default=5000,
                   help="Number of validation samples to attack")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--version", type=str, default="standard",
                   choices=["standard", "plus", "rand"],
                   help="AutoAttack version")
    return p.parse_args()


def main():
    args = parse_args()
    eps = parse_eps(args.eps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed for reproducible sample selection
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Adversarial evaluation: {args.model} ({args.activation})")
    print(f"  eps={eps:.6f} ({args.eps}), norm={args.norm}")
    print(f"  n_samples={args.n_samples}, version={args.version}")

    # Build and load model
    model = build_model(args.model, args.activation, args.num_classes)
    load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)
    model.eval()

    # Dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(args.data_path, transform=transform)

    # Random subset
    n_total = len(dataset)
    n_samples = min(args.n_samples, n_total)
    indices = random.sample(range(n_total), n_samples)
    subset = Subset(dataset, indices)

    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Gather all images and labels into tensors for AutoAttack
    print(f"\nLoading {n_samples} samples...")
    all_images = []
    all_labels = []
    for inputs, targets in loader:
        all_images.append(inputs)
        all_labels.append(targets)
    x_test = torch.cat(all_images, dim=0).to(device)
    y_test = torch.cat(all_labels, dim=0).to(device)

    # Clean accuracy on subset
    with torch.no_grad():
        clean_outputs = model(x_test)
        _, clean_preds = clean_outputs.max(1)
        clean_correct = clean_preds.eq(y_test).sum().item()
        clean_acc = 100.0 * clean_correct / n_samples
    print(f"Clean accuracy on {n_samples} samples: {clean_acc:.2f}%")

    # Run AutoAttack
    try:
        from autoattack import AutoAttack
    except ImportError:
        print("\nERROR: autoattack not installed. Install with: pip install autoattack")
        print("Saving clean-only results.")
        results = {
            "model": args.model,
            "activation": args.activation,
            "eps": eps,
            "eps_str": args.eps,
            "norm": args.norm,
            "n_samples": n_samples,
            "clean_acc": clean_acc,
            "robust_acc": None,
            "error": "autoattack not installed",
        }
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        return

    print(f"\nRunning AutoAttack ({args.version})...")
    adversary = AutoAttack(model, norm=args.norm, eps=eps,
                           version=args.version, verbose=True)

    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)

    # Robust accuracy
    with torch.no_grad():
        adv_outputs = model(x_adv)
        _, adv_preds = adv_outputs.max(1)
        robust_correct = adv_preds.eq(y_test).sum().item()
        robust_acc = 100.0 * robust_correct / n_samples

    print(f"\nClean accuracy:  {clean_acc:.2f}%")
    print(f"Robust accuracy: {robust_acc:.2f}%")

    # Per-attack breakdown (if available)
    per_attack = {}
    if hasattr(adversary, "individual_accuracy"):
        per_attack = adversary.individual_accuracy

    # Save results
    results = {
        "model": args.model,
        "activation": args.activation,
        "checkpoint": args.checkpoint,
        "eps": eps,
        "eps_str": args.eps,
        "norm": args.norm,
        "n_samples": n_samples,
        "version": args.version,
        "seed": args.seed,
        "clean_acc": clean_acc,
        "robust_acc": robust_acc,
        "per_attack": per_attack,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
