"""Evaluate trained models on ImageNet-C corruption benchmark.

Measures corruption robustness across 19 corruption types at 5
severity levels. Reports per-corruption accuracy, per-severity
accuracy, and mean Corruption Error (mCE) relative to AlexNet.

Usage:
    python eval/eval_imagenet_c.py \
        --model convnext_tiny --activation nelu \
        --checkpoint results/imagenet_convnext_tiny_nelu/checkpoint-best.pth \
        --imagenet-c-path /data/ImageNet-C \
        --output results/imagenet_c/convnext_tiny_nelu.json

Requires: ImageNet-C dataset (https://zenodo.org/record/2235448)
"""

import argparse
import json
import os
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm
from timm.utils import accuracy

from nelu import NELU, NiLU
from train.act_swap import replace_activation


# AlexNet error rates on ImageNet-C (from Hendrycks & Dietterich, 2019)
# Used as the denominator for mCE computation
ALEXNET_ERR = {
    "gaussian_noise": 0.886,
    "shot_noise": 0.894,
    "impulse_noise": 0.923,
    "defocus_blur": 0.820,
    "glass_blur": 0.826,
    "motion_blur": 0.786,
    "zoom_blur": 0.798,
    "snow": 0.867,
    "frost": 0.827,
    "fog": 0.819,
    "brightness": 0.565,
    "contrast": 0.853,
    "elastic_transform": 0.646,
    "pixelate": 0.718,
    "jpeg_compression": 0.607,
    "speckle_noise": 0.845,
    "gaussian_blur": 0.787,
    "spatter": 0.718,
    "saturate": 0.658,
}

CORRUPTIONS = list(ALEXNET_ERR.keys())
SEVERITIES = [1, 2, 3, 4, 5]


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
    # Handle DDP state dicts
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded checkpoint from {checkpoint_path}")
    if "epoch" in ckpt:
        print(f"  Checkpoint epoch: {ckpt['epoch']}")
    if "best_acc" in ckpt:
        print(f"  Checkpoint best_acc: {ckpt['best_acc']:.2f}%")


@torch.no_grad()
def evaluate_corruption(model, data_dir, corruption, severity, device,
                        batch_size=128, num_workers=8):
    """Evaluate model on a single corruption at a single severity level."""
    corruption_dir = os.path.join(data_dir, corruption, str(severity))
    if not os.path.isdir(corruption_dir):
        print(f"  WARNING: {corruption_dir} not found, skipping")
        return None

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(corruption_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.topk(1, dim=1)
        correct += predicted.eq(targets.view_as(predicted)).sum().item()
        total += inputs.size(0)

    return 100.0 * correct / total if total > 0 else 0.0


def compute_mce(corruption_accs):
    """Compute mean Corruption Error (mCE) from per-corruption accuracies.

    mCE is the mean across corruptions of (model_err / alexnet_err),
    where errors are averaged across severity levels.
    """
    ces = []
    for corruption in CORRUPTIONS:
        if corruption not in corruption_accs:
            continue
        accs = corruption_accs[corruption]
        if not accs:
            continue
        model_err = 100.0 - sum(accs.values()) / len(accs)
        alexnet_err = ALEXNET_ERR[corruption] * 100.0
        ces.append(model_err / alexnet_err)
    return sum(ces) / len(ces) if ces else float("nan")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate on ImageNet-C")
    p.add_argument("--model", type=str, required=True, help="timm model name")
    p.add_argument("--activation", type=str, default="gelu",
                   choices=["relu", "gelu", "silu", "nelu", "nilu"])
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    p.add_argument("--imagenet-c-path", type=str, required=True,
                   help="Path to ImageNet-C dataset root")
    p.add_argument("--output", type=str, required=True, help="Output JSON path")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--corruptions", type=str, nargs="+", default=None,
                   help="Subset of corruptions to evaluate (default: all 19)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build and load model
    model = build_model(args.model, args.activation, args.num_classes)
    load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)
    model.eval()

    corruptions = args.corruptions if args.corruptions else CORRUPTIONS

    # Evaluate all corruptions x severities
    results = {
        "model": args.model,
        "activation": args.activation,
        "checkpoint": args.checkpoint,
        "per_corruption": {},
        "per_severity": defaultdict(list),
    }

    all_accs = []
    corruption_accs = {}  # corruption -> {severity -> acc}

    for corruption in corruptions:
        print(f"\nEvaluating {corruption}...")
        corruption_accs[corruption] = {}
        corruption_results = {}

        for severity in SEVERITIES:
            acc = evaluate_corruption(model, args.imagenet_c_path, corruption,
                                      severity, device, args.batch_size,
                                      args.num_workers)
            if acc is not None:
                corruption_results[str(severity)] = acc
                corruption_accs[corruption][severity] = acc
                results["per_severity"][str(severity)].append(acc)
                all_accs.append(acc)
                print(f"  severity {severity}: {acc:.2f}%")

        if corruption_results:
            mean_acc = sum(corruption_results.values()) / len(corruption_results)
            corruption_results["mean"] = mean_acc
            results["per_corruption"][corruption] = corruption_results
            print(f"  mean: {mean_acc:.2f}%")

    # Compute summary statistics
    severity_means = {}
    for sev, accs in results["per_severity"].items():
        severity_means[sev] = sum(accs) / len(accs) if accs else 0
    results["per_severity"] = severity_means

    results["mean_accuracy"] = sum(all_accs) / len(all_accs) if all_accs else 0
    results["mCE"] = compute_mce(corruption_accs)

    print(f"\n{'=' * 60}")
    print(f"Mean accuracy across all corruptions/severities: {results['mean_accuracy']:.2f}%")
    print(f"mCE (relative to AlexNet): {results['mCE']:.4f}")
    for sev in SEVERITIES:
        print(f"  Severity {sev}: {severity_means.get(str(sev), 0):.2f}%")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
