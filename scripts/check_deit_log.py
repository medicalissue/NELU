#!/usr/bin/env python3
"""Quick viewer for DeiT-III / ConvNeXt JSON training logs.

Usage:
    python scripts/check_deit_log.py                              # DeiT default
    python scripts/check_deit_log.py results/imagenet/convnext_tiny_nelu/log.txt
    python scripts/check_deit_log.py --tail 10                    # last 10 epochs only
    python scripts/check_deit_log.py --watch                      # auto-refresh every 30s
"""

import argparse
import json
import os
import sys
import time


def print_table(log_path, tail=0):
    if not os.path.exists(log_path):
        print(f"not found: {log_path}")
        return

    rows = []
    for line in open(log_path):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not rows:
        print("(empty log)")
        return

    if tail > 0:
        rows = rows[-tail:]

    best = 0
    # Scan all rows for global best (even if we're only showing tail)
    for line in open(log_path):
        try:
            d = json.loads(line)
            a = d.get("test_acc1", 0)
            if a > best:
                best = a
        except Exception:
            pass

    header = f"{'ep':>4} {'train_loss':>11} {'test_loss':>10} {'top1':>7} {'top5':>7} {'best':>7} {'lr':>10}"
    print(header)
    print("-" * len(header))

    for d in rows:
        a1 = d.get("test_acc1", 0)
        print(
            f"{d.get('epoch', 0):>4} "
            f"{d.get('train_loss', 0):>11.4f} "
            f"{d.get('test_loss', 0):>10.4f} "
            f"{a1:>7.2f} "
            f"{d.get('test_acc5', 0):>7.2f} "
            f"{best:>7.2f} "
            f"{d.get('train_lr', d.get('lr', 0)):>10.6f}"
        )

    total_ep = rows[-1].get("epoch", 0) + 1
    max_ep = 800  # DeiT default, adjust for ConvNeXt (300) etc.
    print("-" * len(header))
    print(f"  epochs done: {total_ep}  |  best top1: {best:.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", nargs="?",
                    default="results/imagenet/deit3_base_nelu/log.txt",
                    help="Path to log.txt")
    p.add_argument("--tail", type=int, default=0,
                    help="Show only last N epochs")
    p.add_argument("--watch", action="store_true",
                    help="Auto-refresh every 30s")
    args = p.parse_args()

    if args.watch:
        while True:
            os.system("clear")
            print(f"[{time.strftime('%H:%M:%S')}] {args.log}\n")
            print_table(args.log, tail=args.tail or 10)
            time.sleep(30)
    else:
        print_table(args.log, tail=args.tail)


if __name__ == "__main__":
    main()
