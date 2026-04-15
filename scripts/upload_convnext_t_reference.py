"""Upload the mmpretrain ConvNeXt-T reference training curve to wandb.

Creates a new run in the "nelu" project that you can overlay against
your NELU runs (y61na0ma, etc.) in the wandb UI.

Source: open-mmlab/mmpretrain's ConvNeXt-T 300ep run, recipe exactly
matching FB ConvNeXt-T (lr 4e-3, bs 4096, 20ep warmup, 300ep cosine
to 1e-6, drop_path 0.1, mixup 0.8, cutmix 1.0, randaug m9, label
smooth 0.1, EMA 0.9999, stochastic depth 0.1, layer scale 1e-6).
Final top1 = 82.158, matching paper 82.1.

IMPORTANT: mmengine's EMAHook swaps in EMA weights for validation, so
the `accuracy/top1` field in the log is the EMA model's top-1, NOT
the raw student. The raw model would be much higher in early epochs
(mmpretrain also publishes a no-ema checkpoint at 81.95% raw vs 82.14%
EMA). So the logged curve is a cold-EMA trajectory.

APPLES-TO-APPLES: compare this reference against your runs'
`Global Test/test_acc1_ema` (EMA val), NOT `Global Test/test_acc1`
(raw val). Both use `decay=0.9999` with no warmup, so the cold-start
shape is preserved.

Log URL: https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_32xb128_in1k_20221207-998cf3e9.json
README:  https://github.com/open-mmlab/mmpretrain/blob/main/configs/convnext/README.md

Usage (on any machine with wandb logged in):
    python scripts/upload_convnext_t_reference.py
"""

import json
import os
import urllib.request

import wandb


URL = ("https://download.openmmlab.com/mmclassification/v0/convnext/"
       "convnext-tiny_32xb128_in1k_20221207-998cf3e9.json")
CACHE = "/tmp/convnext_t_mmpretrain_log.json"


def fetch_log() -> str:
    if os.path.exists(CACHE) and os.path.getsize(CACHE) > 0:
        print(f"Using cached log: {CACHE}")
        return CACHE
    print(f"Downloading {URL}...")
    urllib.request.urlretrieve(URL, CACHE)
    return CACHE


def parse(path: str):
    """Return dict {epoch: {test_acc1, test_acc5, train_loss, lr}}."""
    per_ep = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue

            # Val entries: {accuracy/top1, accuracy/top5, step}
            if "accuracy/top1" in d:
                ep = d["step"]
                slot = per_ep.setdefault(ep, {})
                slot["test_acc1"] = d["accuracy/top1"]
                slot["test_acc5"] = d.get("accuracy/top5", 0.0)

            # Train entries: {lr, loss, epoch, step, ...} — many per epoch.
            # Keep the LAST one we see (end-of-epoch) for a clean scalar.
            elif "loss" in d and "epoch" in d:
                ep = d["epoch"]
                slot = per_ep.setdefault(ep, {})
                slot["train_loss"] = d["loss"]
                slot["train_lr"] = d.get("lr", 0.0)
    return per_ep


def main():
    path = fetch_log()
    per_ep = parse(path)

    epochs = sorted(per_ep.keys())
    print(f"Parsed {len(epochs)} epochs ({min(epochs)}..{max(epochs)})")
    final = per_ep[max(epochs)].get("test_acc1")
    print(f"Final val top1: {final:.3f}  (paper 82.1)")

    run = wandb.init(
        project="nelu",
        name="convnext_t_gelu_reference_mmpretrain_EMA",
        config={
            "model": "convnext_tiny",
            "activation": "gelu",
            "dataset": "imagenet1k",
            "epochs": 300,
            "batch_size": 4096,
            "lr": 4e-3,
            "warmup_epochs": 20,
            "drop_path": 0.1,
            "weight_decay": 0.05,
            "model_ema_decay": 0.9999,
            "final_top1_ema": final,
            "paper_top1": 82.1,
            "source": "mmpretrain",
            "source_url": URL,
            "reported_metric": "EMA model (cold start, decay 0.9999, no warmup)",
            "note": ("mmpretrain's ConvNeXt-T 300ep reproduction. "
                     "accuracy/top1 from mmengine's EMAHook — COMPARE TO "
                     "your run's Global Test/test_acc1_ema, NOT test_acc1 (raw)."),
        },
        tags=["reference", "baseline", "convnext-t", "gelu", "ema-curve"],
    )

    # Use epoch as explicit step axis so wandb UI can align with our runs.
    run.define_metric("epoch")
    for k in ("test_acc1_ema", "test_acc5_ema", "train_loss", "train_lr"):
        if "test" in k:
            run.define_metric(f"Global Test/{k}", step_metric="epoch")
        else:
            run.define_metric(f"Global Train/{k}", step_metric="epoch")

    for ep in epochs:
        m = per_ep[ep]
        log = {"epoch": ep}
        if "test_acc1" in m:
            # Source field is "accuracy/top1" which is the EMA model — log
            # under the matching key so users can overlay directly with
            # their Global Test/test_acc1_ema.
            log["Global Test/test_acc1_ema"] = m["test_acc1"]
            log["Global Test/test_acc5_ema"] = m["test_acc5"]
        if "train_loss" in m:
            log["Global Train/train_loss"] = m["train_loss"]
            log["Global Train/train_lr"] = m["train_lr"]
        wandb.log(log)

    run.finish()
    print(f"\nDone. Run: {run.url}")
    print("\nIn wandb UI:")
    print("  1. Filter by tag 'reference' or name "
          "'convnext_t_gelu_reference_mmpretrain_EMA'")
    print("  2. Overlay on y61na0ma in the same chart")
    print("  3. Compare Global Test/test_acc1_ema (both runs)")
    print("     NOT Global Test/test_acc1 — that's raw, the reference is EMA")
    print("  4. Set X-axis to 'epoch' for alignment")


if __name__ == "__main__":
    main()
