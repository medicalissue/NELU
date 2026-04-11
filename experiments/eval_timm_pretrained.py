#!/usr/bin/env python3
"""Evaluate a timm pretrained checkpoint on ImageNet val.

Used for §4.3 baseline numbers. Loads the timm pretrained weights for
the requested model id, applies the model's own default eval transform
(via timm.data.resolve_data_config), and reports top-1/top-5.

Single-GPU is fine — eval over 50k images is ~1-2 min.

Usage:
    python experiments/eval_timm_pretrained.py \
        --model efficientnet_b2.ra_in1k --data /data/imagenet/val
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

import timm
from timm.data import create_transform, resolve_data_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="timm model id with pretrained tag, e.g. "
                        "'efficientnet_b2.ra_in1k'")
    p.add_argument("--data", required=True,
                   help="ImageNet val directory (path/val)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output", default=None,
                   help="Optional JSON output path")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model} (pretrained)...")
    model = timm.create_model(args.model, pretrained=True)
    model = model.to(device).eval()

    # Use the model's own default eval transform — interp / crop_pct /
    # input_size all come from its pretrained_cfg, so we don't have to
    # hand-encode anything.
    cfg = resolve_data_config({}, model=model, verbose=True)
    transform = create_transform(**cfg, is_training=False)

    ds = datasets.ImageFolder(args.data, transform=transform)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)

    print(f"  {len(ds)} images, batch={args.batch_size}, "
          f"input_size={cfg.get('input_size')}, crop_pct={cfg.get('crop_pct')}")

    correct1, correct5, total = 0, 0, 0
    t0 = time.time()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                logits = model(images)
            _, pred = logits.topk(5, 1, True, True)
            correct1 += (pred[:, 0] == labels).sum().item()
            correct5 += (pred == labels.unsqueeze(1)).any(1).sum().item()
            total += labels.size(0)
    dt = time.time() - t0
    top1 = 100.0 * correct1 / total
    top5 = 100.0 * correct5 / total
    print(f"\n  {args.model}")
    print(f"  top1={top1:.2f}%  top5={top5:.2f}%  ({total} imgs in {dt:.0f}s)")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"model": args.model, "top1": top1, "top5": top5,
                       "n": total, "secs": dt}, f, indent=2)
        print(f"  saved → {args.output}")


if __name__ == "__main__":
    main()
