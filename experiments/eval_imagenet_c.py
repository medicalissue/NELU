#!/usr/bin/env python3
"""ImageNet-C corruption robustness evaluation.

Standard mCE protocol from Hendrycks & Dietterich (ICLR 2019):
    - 15 main corruption types × 5 severity levels
    - Per-corruption error = mean over 5 severities
    - mCE = mean over corruptions of (Error_model / Error_AlexNet)
    - Lower = better robustness

The 4 "extra" corruptions (gaussian_blur, saturate, spatter,
speckle_noise) are also evaluated and reported separately.

Usage:
    # Single model (timm pretrained name OR local checkpoint)
    python experiments/eval_imagenet_c.py \
        --model deit3_base \
        --act gelu \
        --data /data/ImageNet-C

    # Compare GELU baseline vs trained NELU
    python experiments/eval_imagenet_c.py \
        --model deit3_base \
        --gelu-pretrained \
        --nelu-ckpt results/imagenet/deit3_base_nelu/best.pt \
        --data /data/ImageNet-C \
        --wandb
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nelu import NELU
from nelu.cuda_kernel import NELUCUDA

# ── Config ─────────────────────────────────────────────────────────

# 15 main corruptions
MAIN_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]
EXTRA_CORRUPTIONS = ["gaussian_blur", "saturate", "spatter", "speckle_noise"]

# AlexNet ImageNet-C top-1 errors per corruption (Hendrycks 2019, Table 1)
# Used to normalize mCE.
ALEXNET_BASELINE_ERROR = {
    "gaussian_noise":    0.886,
    "shot_noise":        0.894,
    "impulse_noise":     0.923,
    "defocus_blur":      0.820,
    "glass_blur":        0.826,
    "motion_blur":       0.786,
    "zoom_blur":         0.798,
    "snow":              0.867,
    "frost":             0.827,
    "fog":               0.819,
    "brightness":        0.565,
    "contrast":          0.853,
    "elastic_transform": 0.646,
    "pixelate":          0.718,
    "jpeg_compression":  0.607,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Model configs (matching train_imagenet.py)
MODEL_CFGS = {
    "deit3_base": {
        "timm_pretrained": "deit3_base_patch16_224.fb_in1k",
        "timm_name":       "deit3_base_patch16_224",
    },
    "deit3_large": {
        "timm_pretrained": "deit3_large_patch16_224.fb_in1k",
        "timm_name":       "deit3_large_patch16_224",
    },
}


# ── Activation replacement (must match training) ──────────────────

def replace_act(model, act_name):
    """Replace all GELU modules with the requested activation."""
    if act_name == "gelu":
        return model
    if act_name == "nelu":
        new_act = NELU
    else:
        raise ValueError(f"Unknown act: {act_name}")

    def _replace(parent):
        for name, child in parent.named_children():
            if isinstance(child, nn.GELU):
                setattr(parent, name, new_act())
            else:
                _replace(child)

    _replace(model)
    return model


# ── Data loader ────────────────────────────────────────────────────

def make_loader(corruption_dir, batch_size, num_workers):
    """Build a loader for one (corruption, severity) directory.
    Expected layout: corruption_dir/<class>/*.JPEG"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = datasets.ImageFolder(corruption_dir, transform=transform)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


# ── Eval ───────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=True):
            logits = model(x)
        pred = logits.argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 1.0 - (correct / total)        # error rate


def find_corruption_dir(root, corruption, severity):
    """Locate the directory for a (corruption, severity).
    Tries multiple common layouts of ImageNet-C tarballs."""
    candidates = [
        Path(root) / corruption / str(severity),
        Path(root) / "noise" / corruption / str(severity),
        Path(root) / "blur" / corruption / str(severity),
        Path(root) / "weather" / corruption / str(severity),
        Path(root) / "digital" / corruption / str(severity),
        Path(root) / "extra" / corruption / str(severity),
    ]
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c
    return None


def evaluate_model(model, root, corruptions, batch_size, num_workers, device):
    """Returns: {corruption: {1..5: err, "mean": mean_err}, ...}"""
    results = {}
    for corruption in corruptions:
        per_severity = {}
        for severity in [1, 2, 3, 4, 5]:
            cdir = find_corruption_dir(root, corruption, severity)
            if cdir is None:
                print(f"  WARN: missing {corruption}/{severity}")
                continue
            loader = make_loader(cdir, batch_size, num_workers)
            t0 = time.time()
            err = eval_loader(model, loader, device)
            dt = time.time() - t0
            per_severity[severity] = err
            print(f"    {corruption}/{severity}: error={err*100:.2f}%  ({dt:.0f}s)")
        if per_severity:
            per_severity["mean"] = sum(per_severity.values()) / len(per_severity)
            results[corruption] = per_severity
    return results


def compute_mce(results):
    """mCE_f = mean over corruptions of Error_f,c / Error_AlexNet,c."""
    ratios = []
    for c in MAIN_CORRUPTIONS:
        if c not in results or "mean" not in results[c]:
            continue
        baseline = ALEXNET_BASELINE_ERROR.get(c)
        if baseline is None or baseline == 0:
            continue
        ratios.append(results[c]["mean"] / baseline)
    return float(sum(ratios) / len(ratios)) if ratios else float("nan")


# ── Model loading ──────────────────────────────────────────────────

def load_model(model_name, act, ckpt_path, device, gelu_pretrained=False):
    cfg = MODEL_CFGS[model_name]
    if act == "gelu" and gelu_pretrained:
        # Use timm's pretrained ImageNet weights
        model = timm.create_model(cfg["timm_pretrained"], pretrained=True,
                                  num_classes=1000)
    else:
        model = timm.create_model(cfg["timm_name"], pretrained=False,
                                  num_classes=1000)
        model = replace_act(model, act)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
            # Strip _orig_mod prefix from torch.compile
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"  missing keys: {len(missing)} (first: {missing[:3]})")
            if unexpected:
                print(f"  unexpected: {len(unexpected)} (first: {unexpected[:3]})")
    model = model.to(device)
    model.eval()
    return model


# ── Main ───────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deit3_base",
                   choices=list(MODEL_CFGS.keys()))
    p.add_argument("--data", default="/data/ImageNet-C",
                   help="Root of ImageNet-C corruptions")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--include-extra", action="store_true",
                   help="Also evaluate the 4 extra corruptions")
    # Single model
    p.add_argument("--act", default=None, choices=["gelu", "nelu"])
    p.add_argument("--checkpoint", default=None)
    # Comparison: GELU baseline vs trained NELU
    p.add_argument("--gelu-pretrained", action="store_true",
                   help="Use timm's pretrained GELU weights as baseline")
    p.add_argument("--nelu-ckpt", default=None,
                   help="Trained NELU checkpoint to compare")
    p.add_argument("--gelu-ckpt", default=None,
                   help="Trained GELU checkpoint (overrides --gelu-pretrained)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--output", default=None,
                   help="Output JSON path (default: results/ood_imagenet_c.json)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Decide which models to evaluate
    runs = []
    if args.act and args.checkpoint:
        runs.append((args.act.upper(), args.act, args.checkpoint, False))
    if args.gelu_pretrained or args.gelu_ckpt:
        runs.append(("GELU", "gelu", args.gelu_ckpt,
                     args.gelu_ckpt is None))
    if args.nelu_ckpt:
        runs.append(("NELU", "nelu", args.nelu_ckpt, False))
    if not runs:
        # Default: assume timm GELU pretrained as the baseline
        runs.append(("GELU", "gelu", None, True))

    corruptions = MAIN_CORRUPTIONS + (EXTRA_CORRUPTIONS if args.include_extra else [])

    # Optional wandb
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(project="nelu", group="ood_imagenet_c",
                                   name=f"imnet_c_{args.model}",
                                   config=vars(args), reinit=True)
        except Exception as e:
            print(f"  WARN: wandb init failed: {e}")
            wandb_run = None

    all_results = {}
    for name, act, ckpt, use_pretrained in runs:
        print(f"\n{'='*60}")
        print(f"  {name}  (model={args.model}, ckpt={ckpt or 'timm pretrained'})")
        print(f"{'='*60}")
        model = load_model(args.model, act, ckpt, device,
                           gelu_pretrained=use_pretrained)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  params: {n_params:.1f}M")

        results = evaluate_model(model, args.data, corruptions,
                                 args.batch_size, args.num_workers, device)
        mce = compute_mce(results)
        results["mCE"] = mce
        results["mean_error"] = float(
            sum(results[c]["mean"] for c in corruptions if c in results)
            / sum(1 for c in corruptions if c in results)
        )
        all_results[name] = results

        print(f"\n  {name}:  mCE = {mce:.4f}    "
              f"mean_error = {results['mean_error']*100:.2f}%")

        if wandb_run is not None:
            log_dict = {
                f"imnet_c/{name}/mCE": mce,
                f"imnet_c/{name}/mean_error": results["mean_error"],
            }
            for c in corruptions:
                if c in results:
                    log_dict[f"imnet_c/{name}/{c}"] = results[c]["mean"]
                    for s in [1, 2, 3, 4, 5]:
                        if s in results[c]:
                            log_dict[f"imnet_c/{name}/{c}_s{s}"] = results[c][s]
            wandb_run.log(log_dict)

        del model
        torch.cuda.empty_cache()

    # ── Save ───
    out_path = Path(args.output) if args.output else \
        Path("results") / f"ood_imagenet_c_{args.model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  saved → {out_path}")

    # ── Comparison table ───
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  COMPARISON  (mCE: lower is better, % error)")
        print(f"{'='*60}")
        names = list(all_results.keys())
        header = f"  {'Corruption':<22}"
        for n in names:
            header += f" {n:>10}"
        print(header)
        print("  " + "-" * (24 + 11 * len(names)))
        for c in corruptions:
            line = f"  {c:<22}"
            for n in names:
                if c in all_results[n]:
                    line += f"  {all_results[n][c]['mean']*100:>8.2f}%"
                else:
                    line += f"  {'-':>9}"
            print(line)
        print("  " + "-" * (24 + 11 * len(names)))
        line = f"  {'mCE':<22}"
        for n in names:
            line += f"  {all_results[n]['mCE']:>8.4f} "
        print(line)
        line = f"  {'mean_error':<22}"
        for n in names:
            line += f"  {all_results[n]['mean_error']*100:>8.2f}%"
        print(line)

        if wandb_run is not None:
            try:
                import wandb as _wb
                cols = ["corruption"] + names
                table = _wb.Table(columns=cols)
                for c in corruptions:
                    row = [c]
                    for n in names:
                        row.append(all_results[n].get(c, {}).get("mean",
                                                                   float("nan")))
                    table.add_data(*row)
                table.add_data("mCE",
                               *[all_results[n]["mCE"] for n in names])
                wandb_run.log({"imnet_c/comparison_table": table})
            except Exception:
                pass

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
