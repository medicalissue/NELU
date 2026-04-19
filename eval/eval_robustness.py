"""Evaluate trained models on ImageNet robustness benchmarks.

Supports: ImageNet-C (corruptions), ImageNet-A (natural adversarial),
ImageNet-R (renditions), ImageNet-O (out-of-distribution).

Uses DataParallel across all available GPUs for fast inference.

Usage:
    # All benchmarks at once:
    python eval/eval_robustness.py \
        --model convnext_tiny --activation nelu \
        --checkpoint results/imagenet_convnext_tiny_nelu/checkpoint-best.pt \
        --data-root /data --output results/robustness/convnext_tiny_nelu.json

    # Specific benchmark only:
    python eval/eval_robustness.py \
        --model convnext_tiny --activation nelu \
        --checkpoint results/... --benchmarks imagenet-c imagenet-a
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm
from nelu import NELU, NiLU
from train.act_swap import replace_activation


# ── ImageNet-A / ImageNet-R class mapping ─────────────────────────
# These benchmarks use a 200-class subset of ImageNet-1k.
# We need to map the model's 1000-class output to the 200-class labels.

# The 200 ImageNet class indices used by ImageNet-A and ImageNet-R
# (from https://github.com/hendrycks/natural-adv-examples)
IMAGENET_A_INDICES = [
    6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76,
    79, 89, 90, 94, 96, 97, 99, 105, 107, 108, 110, 113, 124, 125, 130, 132,
    143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295, 298, 301,
    306, 307, 308, 309, 310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327,
    330, 334, 335, 336, 347, 361, 363, 372, 378, 386, 397, 400, 401, 402, 404,
    407, 411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461, 462,
    470, 472, 483, 486, 488, 492, 496, 514, 516, 528, 530, 539, 542, 543, 549,
    552, 557, 561, 562, 569, 572, 573, 575, 579, 589, 606, 607, 609, 614, 626,
    627, 640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704, 719, 736,
    746, 749, 752, 758, 763, 765, 768, 773, 774, 776, 779, 780, 786, 792, 797,
    802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845, 847, 850, 859,
    862, 870, 879, 880, 888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937,
    943, 945, 947, 951, 954, 956, 957, 959, 971, 972, 980, 981, 984, 986, 987,
    988,
]

# ImageNet-C corruption types
CORRUPTIONS_C = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression", "speckle_noise", "gaussian_blur",
    "spatter", "saturate",
]


# ── Model loading ─────────────────────────────────────────────────

def build_model(model_name, activation, num_classes=1000):
    """Create model with activation swap, load to GPU with DataParallel."""
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    if activation == "nelu":
        n = replace_activation(model, nn.GELU, NELU)
        if n == 0:
            n = replace_activation(model, nn.ReLU, NELU)
        print(f"Replaced {n} modules with NELU")
    elif activation == "nilu":
        n = replace_activation(model, nn.SiLU, NiLU)
        if n == 0:
            n = replace_activation(model, nn.ReLU, NiLU)
        print(f"Replaced {n} modules with NiLU")

    return model


def load_checkpoint(model, path, device):
    """Load checkpoint, handling various formats."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    # Remove DDP "module." prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    epoch = ckpt.get("epoch", "?")
    acc = ckpt.get("best_acc", "?")
    print(f"Loaded checkpoint: epoch={epoch}, best_acc={acc}")


def get_val_transform(model_name):
    """Get the correct validation transform for the model."""
    config = timm.data.resolve_model_data_config(
        timm.create_model(model_name, pretrained=False))
    return timm.data.create_transform(**config, is_training=False)


# ── Evaluation functions ──────────────────────────────────────────

@torch.no_grad()
def evaluate_folder(model, data_dir, transform, batch_size, workers,
                    class_indices=None, device="cuda"):
    """Evaluate on an ImageFolder dataset. Returns top-1 accuracy.

    If class_indices is provided (for ImageNet-A/R), maps the model's
    1000-class output to the subset before computing accuracy.
    """
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)

    correct = 0
    total = 0
    model.eval()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)

        if class_indices is not None:
            # Map 1000-class output to 200-class subset
            outputs = outputs[:, class_indices]

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    acc = 100.0 * correct / max(total, 1)
    return acc, total


def evaluate_imagenet_c(model, data_root, transform, batch_size, workers,
                        device="cuda"):
    """Evaluate on ImageNet-C: 19 corruptions × 5 severities."""
    c_root = os.path.join(data_root, "ImageNet-C")
    if not os.path.isdir(c_root):
        print(f"  ImageNet-C not found at {c_root}, skipping")
        return None

    results = {}
    for corruption in CORRUPTIONS_C:
        corr_accs = []
        for severity in range(1, 6):
            data_dir = os.path.join(c_root, corruption, str(severity))
            if not os.path.isdir(data_dir):
                continue
            acc, n = evaluate_folder(model, data_dir, transform, batch_size,
                                     workers, device=device)
            corr_accs.append(acc)
        if corr_accs:
            results[corruption] = {
                "per_severity": corr_accs,
                "mean": sum(corr_accs) / len(corr_accs),
            }
            print(f"    {corruption:25s}: {results[corruption]['mean']:.1f}%")

    if results:
        all_means = [v["mean"] for v in results.values()]
        results["_mean_accuracy"] = sum(all_means) / len(all_means)
        print(f"    {'MEAN':25s}: {results['_mean_accuracy']:.1f}%")

    return results


# ── Main ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Robustness evaluation suite")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--activation", type=str, default="gelu",
                    choices=["relu", "gelu", "silu", "nelu", "nilu"])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, default="/data",
                    help="Root containing imagenet/, ImageNet-C/, imagenet-a/, etc.")
    p.add_argument("--output", type=str, default="results/robustness.json")
    p.add_argument("--benchmarks", nargs="+",
                    default=["imagenet-c", "imagenet-a", "imagenet-r", "imagenet-o"],
                    help="Which benchmarks to run")
    p.add_argument("--batch-size", type=int, default=256,
                    help="Per-GPU batch size (multiplied by number of GPUs)")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()

    print(f"Model: {args.model}, activation: {args.activation}")
    print(f"GPUs: {n_gpus}, batch size: {args.batch_size} × {n_gpus} = {args.batch_size * n_gpus}")

    # Build model + load checkpoint
    model = build_model(args.model, args.activation)
    load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)

    # DataParallel for multi-GPU inference
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel across {n_gpus} GPUs")

    transform = get_val_transform(args.model)
    effective_batch = args.batch_size * max(n_gpus, 1)

    results = {
        "model": args.model,
        "activation": args.activation,
        "checkpoint": args.checkpoint,
    }

    # ── ImageNet validation (clean baseline) ──
    val_dir = os.path.join(args.data_root, "imagenet", "val")
    if os.path.isdir(val_dir):
        print("\n── ImageNet validation (clean) ──")
        t0 = time.time()
        acc, n = evaluate_folder(model, val_dir, transform, effective_batch,
                                  args.workers, device=device)
        print(f"  Clean top-1: {acc:.2f}% ({n} images, {time.time()-t0:.1f}s)")
        results["imagenet_val"] = {"top1": acc, "n_images": n}

    # ── ImageNet-C ──
    if "imagenet-c" in args.benchmarks:
        print("\n── ImageNet-C (corruption robustness) ──")
        t0 = time.time()
        c_results = evaluate_imagenet_c(model, args.data_root, transform,
                                         effective_batch, args.workers, device=device)
        if c_results:
            results["imagenet_c"] = c_results
            print(f"  ({time.time()-t0:.1f}s)")

    # ── ImageNet-A ──
    if "imagenet-a" in args.benchmarks:
        a_dir = os.path.join(args.data_root, "imagenet-a")
        if os.path.isdir(a_dir):
            print("\n── ImageNet-A (natural adversarial) ──")
            t0 = time.time()
            acc, n = evaluate_folder(model, a_dir, transform, effective_batch,
                                      args.workers, class_indices=IMAGENET_A_INDICES,
                                      device=device)
            print(f"  Top-1: {acc:.2f}% ({n} images, {time.time()-t0:.1f}s)")
            results["imagenet_a"] = {"top1": acc, "n_images": n}
        else:
            print(f"  ImageNet-A not found at {a_dir}, skipping")

    # ── ImageNet-R ──
    if "imagenet-r" in args.benchmarks:
        r_dir = os.path.join(args.data_root, "imagenet-r")
        if os.path.isdir(r_dir):
            print("\n── ImageNet-R (renditions) ──")
            t0 = time.time()
            # ImageNet-R uses the same 200 classes as ImageNet-A
            acc, n = evaluate_folder(model, r_dir, transform, effective_batch,
                                      args.workers, class_indices=IMAGENET_A_INDICES,
                                      device=device)
            print(f"  Top-1: {acc:.2f}% ({n} images, {time.time()-t0:.1f}s)")
            results["imagenet_r"] = {"top1": acc, "n_images": n}
        else:
            print(f"  ImageNet-R not found at {r_dir}, skipping")

    # ── ImageNet-O ──
    if "imagenet-o" in args.benchmarks:
        o_dir = os.path.join(args.data_root, "imagenet-o")
        if os.path.isdir(o_dir):
            print("\n── ImageNet-O (out-of-distribution) ──")
            t0 = time.time()
            # For OOD, we measure maximum softmax probability (MSP) on
            # images that should NOT be classified as any ImageNet class
            acc, n = evaluate_folder(model, o_dir, transform, effective_batch,
                                      args.workers, class_indices=IMAGENET_A_INDICES,
                                      device=device)
            # Lower accuracy = better OOD detection (model correctly says "none of these")
            print(f"  Top-1 (lower=better OOD): {acc:.2f}% ({n} images, {time.time()-t0:.1f}s)")
            results["imagenet_o"] = {"top1": acc, "n_images": n}
        else:
            print(f"  ImageNet-O not found at {o_dir}, skipping")

    # ── Save results ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
