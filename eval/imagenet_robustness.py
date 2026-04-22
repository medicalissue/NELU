"""ImageNet-1k robustness evaluation.

Evaluates a trained checkpoint on the four standard robustness benchmarks
used alongside ImageNet in the paper:

* **ImageNet-C** — 19 corruption types × 5 severity levels
  (Hendrycks & Dietterich 2019). We report per-corruption accuracy and a
  uniform mean across corruption × severity.
* **ImageNet-A** — 7500 naturally adversarial images from a 200-class
  subset of ImageNet-1k (Hendrycks et al. 2021). The model's 1000-way
  logits are masked to the 200 subset indices.
* **ImageNet-R** — 30k renditions (paintings, sculptures, cartoons) of
  a different 200-class subset (Hendrycks et al. 2020). Class mapping is
  distinct from ImageNet-A.
* **ImageNet-O** — 2000 out-of-distribution images sharing the
  ImageNet-A class list; reported as top-1 accuracy where *lower* is
  better (the model should not confidently classify these).

Directory layout assumed under ``--data-root`` (matches the repository's
prepare_data.sh / EBS snapshot):

    data-root/
      imagenet/val/<wnid>/*.JPEG
      ImageNet-C/<corruption>/<severity>/<wnid>/*.JPEG
      imagenet-a/<wnid>/*.JPEG
      imagenet-r/<wnid>/*.JPEG
      imagenet-o/<wnid>/*.JPEG
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

import timm

from train.swap import apply_gate_normalization


# ── Class-index subsets ─────────────────────────────────────────────────
# Lists copied from the official repositories. ImageNet-A and ImageNet-R
# both select 200 ImageNet-1k classes, but the two subsets only partially
# overlap — do not conflate them.

IMAGENET_A_INDICES: tuple[int, ...] = (
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
)

# ImageNet-R subset from https://github.com/hendrycks/imagenet-r.
IMAGENET_R_INDICES: tuple[int, ...] = (
    1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84,
    90, 94, 96, 97, 99, 100, 105, 107, 113, 122, 125, 130, 132, 144, 145, 147,
    148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203,
    207, 208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260,
    263, 265, 267, 269, 276, 277, 281, 288, 289, 291, 292, 293, 296, 299, 301,
    308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340,
    341, 344, 347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388, 390, 393,
    397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 448, 457, 462,
    463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583,
    587, 593, 594, 596, 609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724,
    763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852,
    866, 875, 883, 889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945,
    947, 948, 949, 951, 953, 954, 957, 963, 965, 967, 980, 981, 983, 988,
)

CORRUPTIONS: tuple[str, ...] = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression", "speckle_noise", "gaussian_blur",
    "spatter", "saturate",
)


# ── Model + checkpoint ─────────────────────────────────────────────────


def build_model(model_name: str, activation: str) -> nn.Module:
    model = timm.create_model(model_name, pretrained=False, num_classes=1000)
    n = apply_gate_normalization(model, activation)
    if activation in {"nelu", "nilu"}:
        print(f"[eval] swapped {n} activation modules for {activation.upper()}")
    return model


def load_checkpoint(model: nn.Module, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[eval] warning: missing keys ({len(missing)}): {missing[:4]}...")
    if unexpected:
        print(f"[eval] warning: unexpected keys ({len(unexpected)}): {unexpected[:4]}...")
    epoch = ckpt.get("epoch", "?")
    print(f"[eval] loaded checkpoint {path} (epoch={epoch})")


def build_transform(model_name: str):
    cfg = timm.data.resolve_model_data_config(
        timm.create_model(model_name, pretrained=False)
    )
    return timm.data.create_transform(**cfg, is_training=False)


# ── Accuracy on a single ImageFolder ─────────────────────────────────────


@torch.no_grad()
def evaluate_folder(
    model: nn.Module,
    root: str,
    transform,
    batch_size: int,
    workers: int,
    class_indices: Sequence[int] | None = None,
    device: str = "cuda",
) -> tuple[float, int]:
    dataset = datasets.ImageFolder(root, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
    )

    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        if class_indices is not None:
            logits = logits[:, list(class_indices)]
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)

    return (100.0 * correct / max(total, 1)), total


def evaluate_imagenet_c(
    model: nn.Module, data_root: str, transform,
    batch_size: int, workers: int, device: str = "cuda",
) -> dict | None:
    c_root = Path(data_root) / "ImageNet-C"
    if not c_root.is_dir():
        print(f"[eval] ImageNet-C not found at {c_root}; skipping")
        return None

    out: dict = {}
    for corruption in CORRUPTIONS:
        per_severity: list[float] = []
        for severity in range(1, 6):
            d = c_root / corruption / str(severity)
            if not d.is_dir():
                continue
            acc, _ = evaluate_folder(
                model, str(d), transform, batch_size, workers, device=device
            )
            per_severity.append(acc)
        if per_severity:
            out[corruption] = {
                "per_severity": per_severity,
                "mean": sum(per_severity) / len(per_severity),
            }
            print(f"  {corruption:<25s} {out[corruption]['mean']:6.2f}%")

    if out:
        means = [v["mean"] for v in out.values()]
        out["_mean"] = sum(means) / len(means)
        print(f"  {'MEAN':<25s} {out['_mean']:6.2f}%")
    return out


# ── CLI ────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ImageNet robustness evaluation")
    p.add_argument("--model", required=True,
                   help="timm model name, e.g. convnext_tiny")
    p.add_argument("--activation", default="gelu",
                   choices=["relu", "gelu", "silu", "nelu", "nilu"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="/data")
    p.add_argument("--output", default="results/imagenet_robustness.json")
    p.add_argument(
        "--benchmarks", nargs="+",
        default=["clean", "imagenet-c", "imagenet-a", "imagenet-r", "imagenet-o"],
        choices=["clean", "imagenet-c", "imagenet-a", "imagenet-r", "imagenet-o"],
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count()

    model = build_model(args.model, args.activation)
    load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"[eval] DataParallel across {n_gpus} GPUs")

    transform = build_transform(args.model)
    batch = args.batch_size * max(n_gpus, 1)
    results: dict = {
        "model": args.model,
        "activation": args.activation,
        "checkpoint": args.checkpoint,
    }

    if "clean" in args.benchmarks:
        val_dir = Path(args.data_root) / "imagenet" / "val"
        if val_dir.is_dir():
            print("\n── ImageNet val (clean) ──")
            t0 = time.time()
            acc, n = evaluate_folder(
                model, str(val_dir), transform, batch, args.workers, device=device,
            )
            print(f"  top-1: {acc:.2f}%  ({n} images, {time.time()-t0:.1f}s)")
            results["clean"] = {"top1": acc, "n": n}

    if "imagenet-c" in args.benchmarks:
        print("\n── ImageNet-C ──")
        t0 = time.time()
        c = evaluate_imagenet_c(
            model, args.data_root, transform, batch, args.workers, device=device,
        )
        if c:
            results["imagenet_c"] = c
            print(f"  ({time.time()-t0:.1f}s)")

    for name, indices, lower_is_better in (
        ("imagenet-a", IMAGENET_A_INDICES, False),
        ("imagenet-r", IMAGENET_R_INDICES, False),
        ("imagenet-o", IMAGENET_A_INDICES, True),
    ):
        if name not in args.benchmarks:
            continue
        root = Path(args.data_root) / name
        if not root.is_dir():
            print(f"\n[eval] {name} not found at {root}; skipping")
            continue
        banner = f"{name} ({'lower is better' if lower_is_better else 'higher is better'})"
        print(f"\n── {banner} ──")
        t0 = time.time()
        acc, n = evaluate_folder(
            model, str(root), transform, batch, args.workers,
            class_indices=indices, device=device,
        )
        print(f"  top-1: {acc:.2f}%  ({n} images, {time.time()-t0:.1f}s)")
        results[name.replace("-", "_")] = {"top1": acc, "n": n}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[eval] results saved to {out_path}")


if __name__ == "__main__":
    main()
