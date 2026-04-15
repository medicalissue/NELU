#!/usr/bin/env python3
"""Salvage the polluted ConvNeXt-T run 164tlkm0.

What happened: an 8-way parallel CIFAR ablation was launched while
WANDB_RUN_ID=164tlkm0 and WANDB_RESUME=must were still exported in the
user's shell. All 8 CIFAR processes tried to resume the ConvNeXt run,
clobbering its name / config and appending CIFAR metrics to its
history. The per-channel ConvNeXt training itself (checkpoints on disk
+ S3) is unaffected.

This script:
  1. Downloads the full history of 164tlkm0.
  2. Filters to ConvNeXt-only keys (drops CIFAR pollution by key prefix).
  3. Creates a NEW clean wandb run named `convnext_tiny_nelu_perchannel`
     in the same project and re-logs every ConvNeXt epoch 0..max.
  4. Prints the new run URL and suggested next commands.

Run this ONCE on any machine with wandb logged in. It does NOT delete
the original 164tlkm0 — you can do that manually from the UI afterward
if you want (or keep it as a historical artifact).

Usage:
    python scripts/recover_164tlkm0.py
    python scripts/recover_164tlkm0.py --dry-run
    python scripts/recover_164tlkm0.py --new-name my_chosen_name
"""

import argparse
import json
from collections import defaultdict

import wandb


# Keys that belong to ConvNeXt's FB-main.py logging. Anything NOT in
# one of these prefixes is assumed to be CIFAR pollution and discarded.
CONVNEXT_KEY_PREFIXES = (
    "Global Test/",
    "Global Train/",
    "Rank-0 Batch Wise/",
    "epoch",
    "_step",
    "_runtime",
    "_timestamp",
)

CIFAR_KEY_PREFIXES = (
    "train/",
    "test/",
    "gamma/",
    "lr",   # CIFAR script logged scalar "lr"
)


def is_convnext_key(k: str) -> bool:
    return any(k.startswith(p) for p in CONVNEXT_KEY_PREFIXES)


def is_cifar_key(k: str) -> bool:
    return any(k.startswith(p) for p in CIFAR_KEY_PREFIXES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-run", default="medicalissues/nelu/164tlkm0")
    ap.add_argument("--new-name", default="convnext_tiny_nelu_perchannel")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch + inspect but don't create new run.")
    args = ap.parse_args()

    api = wandb.Api()
    src = api.run(args.src_run)
    print(f"Source run: {src.path}")
    print(f"  state: {src.state}")
    print(f"  name (currently): {src.name}")
    print(f"  config keys: {list(src.config.keys())[:10]}")
    print()

    # Pull FULL history
    print("Downloading full history... (may take a minute)")
    rows = list(src.scan_history())
    print(f"  {len(rows)} raw history rows")

    # Separate ConvNeXt-only rows by "epoch" key value.
    # Keep the LAST value for each (epoch, key) pair in the ConvNeXt range
    # (epoch <= 149 roughly — the killed run ended there).
    #
    # We group rows by epoch (if present) and keep only ConvNeXt-prefixed
    # keys. Rows without an "epoch" key are assumed to be batch-wise logs
    # and are dropped (we only need epoch-level training curves).
    convnext_rows_by_epoch = {}
    cifar_row_count = 0
    dropped = 0
    for row in rows:
        ep = row.get("epoch")
        if ep is None:
            dropped += 1
            continue
        # Detect CIFAR pollution rows and skip
        # A row is CIFAR if it has any cifar-prefixed key OR if epoch is
        # very small AND it has any cifar-prefixed key
        has_cifar_key = any(is_cifar_key(k) for k in row.keys())
        has_convnext_key = any(is_convnext_key(k) and k != "epoch"
                               for k in row.keys())

        if has_cifar_key and not has_convnext_key:
            # pure CIFAR row → drop
            cifar_row_count += 1
            continue

        # Keep ConvNeXt keys only from this row
        clean = {k: v for k, v in row.items() if is_convnext_key(k)}
        if not clean or "epoch" not in clean:
            dropped += 1
            continue
        # Deduplicate: if we already have a row for this epoch, merge
        # (later rows win on conflicts — they're usually the "after eval"
        # state which is what we want).
        ep_int = int(ep)
        existing = convnext_rows_by_epoch.get(ep_int, {})
        existing.update(clean)
        convnext_rows_by_epoch[ep_int] = existing

    epochs_sorted = sorted(convnext_rows_by_epoch.keys())
    print()
    print(f"  ConvNeXt-epoch rows kept: {len(epochs_sorted)}")
    print(f"    epoch range: {min(epochs_sorted)}..{max(epochs_sorted)}")
    print(f"  CIFAR rows dropped: {cifar_row_count}")
    print(f"  non-epoch rows dropped: {dropped}")
    print()

    # Peek at first and last ConvNeXt rows
    if epochs_sorted:
        first = convnext_rows_by_epoch[epochs_sorted[0]]
        last = convnext_rows_by_epoch[epochs_sorted[-1]]
        print(f"  first kept row (ep {epochs_sorted[0]}):")
        for k, v in list(first.items())[:8]:
            print(f"    {k}: {v}")
        print(f"  last kept row (ep {epochs_sorted[-1]}):")
        for k, v in list(last.items())[:8]:
            print(f"    {k}: {v}")

    # Reconstruct a clean config
    convnext_config = {
        "model": "convnext_tiny",
        "activation": "nelu",
        "gamma_mode": "per_channel",
        "dataset": "imagenet1k",
        "batch_size_total": 4096,
        "batch_size_per_gpu": 512,
        "num_gpus": 8,
        "lr": 4e-3,
        "warmup_epochs": 20,
        "epochs": 300,
        "drop_path": 0.1,
        "weight_decay": 0.05,
        "model_ema_decay": 0.9999,
        "recovered_from": args.src_run,
        "recovery_note": ("Original 164tlkm0 was polluted by a parallel "
                          "CIFAR ablation that resumed the run via a stale "
                          "WANDB_RUN_ID env var. This run is the ConvNeXt "
                          "history re-logged cleanly. Non-ConvNeXt metrics "
                          "were filtered out by key prefix."),
    }

    if args.dry_run:
        print("\n[dry-run] Would create new run with name "
              f"{args.new_name!r} and re-log {len(epochs_sorted)} epochs.")
        return

    print(f"\nCreating clean run: {args.new_name}")
    new_run = wandb.init(
        project=src.project,
        entity=src.entity,
        name=args.new_name,
        config=convnext_config,
        tags=["recovered", "convnext-t", "nelu", "per-channel"],
        notes=f"Recovered from {args.src_run} (polluted by CIFAR ablation)",
        reinit=True,
    )
    new_run.define_metric("epoch")
    new_run.define_metric("Global Test/*", step_metric="epoch")
    new_run.define_metric("Global Train/*", step_metric="epoch")

    print(f"  Logging {len(epochs_sorted)} epochs...")
    for i, ep in enumerate(epochs_sorted):
        row = convnext_rows_by_epoch[ep]
        # Drop the internal wandb keys before logging
        log = {k: v for k, v in row.items()
               if not k.startswith("_") and v is not None}
        log["epoch"] = ep
        new_run.log(log)
        if (i + 1) % 25 == 0 or i == len(epochs_sorted) - 1:
            print(f"    logged {i+1}/{len(epochs_sorted)} (ep {ep})")

    final_top1 = None
    final_ema = None
    if epochs_sorted:
        last = convnext_rows_by_epoch[epochs_sorted[-1]]
        final_top1 = last.get("Global Test/test_acc1")
        final_ema = last.get("Global Test/test_acc1_ema")
    if final_top1 is not None:
        new_run.summary["final_test_acc1"] = final_top1
    if final_ema is not None:
        new_run.summary["final_test_acc1_ema"] = final_ema
    new_run.summary["n_epochs_recovered"] = len(epochs_sorted)

    new_run.finish()
    print()
    print("=" * 60)
    print(f"DONE. Clean run: {new_run.url}")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"  1. Verify in wandb UI that {args.new_name} has the clean")
    print(f"     ConvNeXt trajectory (ep 0..{max(epochs_sorted)}).")
    print("  2. (Optional) Delete the old 164tlkm0 from the wandb UI.")
    print("  3. When you RESUME ConvNeXt training, DO NOT set")
    print("     WANDB_RUN_ID — let wandb create a fresh run. Link to this")
    print("     clean run via config if you want continuity in the paper.")
    print()


if __name__ == "__main__":
    main()
