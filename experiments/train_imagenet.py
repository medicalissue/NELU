#!/usr/bin/env python3
"""ImageNet-1k ViT training — GELU vs NELU.

DeiT-III recipe (Touvron et al., 2022, "Revenge of the ViT")
matches README_revenge config exactly:
    Optimizer        : FusedLAMB (unscaled lr)
    LR / WD          : 3e-3 / 0.05
    WD groups        : decay on 2D weights only (no decay on bias/norm)
    Schedule         : cosine to 1e-5, 5-ep linear warmup from 1e-6
    Loss             : BCE (no label smoothing, no target threshold)
    Augmentation     : RRC + HFlip + ColorJitter(0.3) + 3-Augment
    Mixup / Cutmix   : 0.8 / 1.0
    Drop path        : 0.2 (B), 0.45 (L)
    Repeated Aug     : disabled
    EMA              : disabled
    Erasing          : disabled
    Gradient clip    : 1.0
    Train resolution : 192    Eval resolution  : 224 (eval crop ratio 1.0)
    Effective batch  : 2048 (B), 1024 (L)
    Epochs           : 800

GELU baseline: use the DeiT-III pretrained checkpoint (timm fb_in1k).
NELU: train from scratch with identical recipe.

Usage (H100×8):
    # GELU eval (pretrained)
    python train_imagenet.py --model deit3_base --act gelu --data /data/imagenet --eval-only
    python train_imagenet.py --model deit3_large --act gelu --data /data/imagenet --eval-only

    # NELU from scratch
    torchrun --nproc_per_node=8 train_imagenet.py \
        --model deit3_base --act nelu --data /data/imagenet --wandb
    torchrun --nproc_per_node=8 train_imagenet.py \
        --model deit3_large --act nelu --data /data/imagenet --wandb
"""

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm

import timm
from timm.data import Mixup
from timm.loss import BinaryCrossEntropy
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nelu import NELU
from nelu import NELUCUDA
_NELU_CLS = NELUCUDA if NELUCUDA is not None else NELU

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# ── Model configs ────────────────────────────────────────────────

MODEL_CFGS = {
    # DeiT-III recipe (Touvron et al., README_revenge config)
    # Effective batch matches paper via grad_accum on 8×40GB A100.
    #
    # ViT-B: 128 × accum 2 × 8 GPU = 2048 effective  (paper: 64×32=2048)
    "deit3_base": {
        "timm_name": "deit3_base_patch16_224",
        "timm_pretrained": "deit3_base_patch16_224.fb_in1k",
        "drop_path": 0.2,
        "batch_per_gpu": 128,
        "grad_accum": 2,
    },
    # ViT-L: 64 × accum 2 × 8 GPU = 1024 effective  (paper: 32×32=1024)
    "deit3_large": {
        "timm_name": "deit3_large_patch16_224",
        "timm_pretrained": "deit3_large_patch16_224.fb_in1k",
        "drop_path": 0.45,
        "batch_per_gpu": 64,
        "grad_accum": 2,
    },
}


# ── Activation ───────────────────────────────────────────────────

def replace_act(model, act_name):
    if act_name == "gelu":
        return model
    target = _NELU_CLS
    for name, child in model.named_children():
        if isinstance(child, (nn.GELU, nn.ReLU)):
            setattr(model, name, target())
        else:
            replace_act(child, act_name)
    return model


# ── 3-Augment (DeiT-III) ────────────────────────────────────────

class ThreeAugment:
    """Randomly apply one of: grayscale, solarize, gaussian blur."""
    def __call__(self, img):
        op = torch.randint(0, 3, (1,)).item()
        if op == 0:
            return transforms.functional.rgb_to_grayscale(img, num_output_channels=3)
        elif op == 1:
            return transforms.functional.solarize(img, threshold=128)
        else:
            from PIL import ImageFilter
            return img.filter(ImageFilter.GaussianBlur(
                radius=torch.empty(1).uniform_(0.1, 2.0).item()))


# ── Data ─────────────────────────────────────────────────────────

def build_dataloaders(args, train_size=192):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(train_size,
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        ThreeAugment(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(224 / 1.0),  # eval-crop-ratio = 1.0
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    train_ds = datasets.ImageFolder(os.path.join(args.data, "train"), train_transform)
    val_ds = datasets.ImageFolder(os.path.join(args.data, "val"), val_transform)

    if args.distributed:
        train_sampler = DistributedSampler(train_ds)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = val_sampler = None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None), num_workers=args.workers,
        pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
        sampler=val_sampler)
    return train_loader, val_loader, train_sampler


# ── Training ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, scaler,
                    mixup_fn, criterion, device, epoch, args, raw):
    """Training loop with gradient accumulation.

    For grad_accum > 1, we scale the per-micro-batch loss by 1/grad_accum
    and only call optimizer.step() after all micro-batches. Non-final
    micro-steps skip DDP gradient sync via model.no_sync() to save NCCL
    bandwidth.
    """
    model.train()
    accum = max(1, args.grad_accum)
    total_loss, total = 0.0, 0
    pbar = tqdm(loader, desc=f"train {epoch}",
                disable=not args.is_main, ncols=100)

    optimizer.zero_grad(set_to_none=True)
    accum_count = 0

    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        is_last_micro = (accum_count + 1) == accum

        # Skip gradient sync on non-final micro-steps
        if args.distributed and not is_last_micro and hasattr(model, "no_sync"):
            sync_ctx = model.no_sync()
        else:
            sync_ctx = nullcontext()

        with sync_ctx:
            with torch.amp.autocast("cuda", enabled=args.amp):
                loss = criterion(model(images), labels) / accum
            scaler.scale(loss).backward()

        total_loss += loss.item() * accum * images.size(0)
        total += images.size(0)
        accum_count += 1

        if is_last_micro:
            # DeiT-III: gradient clipping at norm 1.0
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0

        pbar.set_postfix(loss=f"{total_loss/total:.4f}")

    # Drop any tail micro-batches that didn't complete a full accum window
    # (with DistributedSampler + drop_last=True, num_batches is constant,
    # so this is a defensive no-op in practice)
    if accum_count > 0:
        optimizer.zero_grad(set_to_none=True)

    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    correct1, correct5, total = 0, 0, 0
    for images, labels in tqdm(loader, desc="eval",
                                disable=not args.is_main, ncols=100):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            output = model(images)
        _, pred = output.topk(5, 1, True, True)
        correct1 += (pred[:, 0] == labels).sum().item()
        correct5 += (pred == labels.unsqueeze(1)).any(1).sum().item()
        total += labels.size(0)
    if args.distributed:
        stats = torch.tensor([correct1, correct5, total],
                             device=device, dtype=torch.float64)
        dist.all_reduce(stats)
        correct1, correct5, total = stats.tolist()
    return 100.0 * correct1 / total, 100.0 * correct5 / total


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deit3_base",
                        choices=list(MODEL_CFGS.keys()))
    parser.add_argument("--act", default="nelu", choices=["gelu", "nelu"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Per-GPU micro batch (auto from config if not set)")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="Gradient accumulation steps (auto from config)")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = MODEL_CFGS[args.model]
    if args.batch_size is None:
        args.batch_size = cfg["batch_per_gpu"]
    if args.grad_accum is None:
        args.grad_accum = cfg.get("grad_accum", 1)
    if args.output_dir is None:
        args.output_dir = f"results/imagenet/{args.model}_{args.act}"

    # Distributed
    args.distributed = int(os.environ.get("RANK", -1)) != -1
    if args.distributed:
        dist.init_process_group("nccl")
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        args.local_rank = args.rank = 0
        args.world_size = 1
        device = torch.device("cuda")
    args.is_main = args.rank == 0

    torch.manual_seed(args.seed + args.rank)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.is_main:
        eff = args.batch_size * args.grad_accum * args.world_size
        print(f"{args.model} + {args.act} | "
              f"micro {args.batch_size} × accum {args.grad_accum} × "
              f"{args.world_size} GPU = {eff} effective | "
              f"{args.epochs} epochs")

    # Model
    if args.act == "gelu" and args.eval_only:
        model = timm.create_model(cfg["timm_pretrained"], pretrained=True)
    else:
        model = timm.create_model(cfg["timm_name"], pretrained=False,
                                  num_classes=1000,
                                  drop_path_rate=cfg["drop_path"])
    model = replace_act(model, args.act)
    model = model.to(device)

    if args.compile:
        model = torch.compile(model)

    # DeiT-III does NOT use EMA (per official README_revenge config)
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank])
    raw = model.module if args.distributed else model
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod

    if args.is_main:
        n = sum(p.numel() for p in raw.parameters()) / 1e6
        print(f"Params: {n:.1f}M")

    # Eval only
    if args.eval_only:
        _, val_loader, _ = build_dataloaders(args, train_size=224)
        t1, t5 = evaluate(model, val_loader, device, args)
        if args.is_main:
            print(f"top1={t1:.2f}%  top5={t5:.2f}%")
        return

    # Data
    train_loader, val_loader, train_sampler = build_dataloaders(args, train_size=192)

    # Mixup + Cutmix
    mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0,
                     label_smoothing=0.0, num_classes=1000)

    # BCE loss (DeiT-III)
    criterion = BinaryCrossEntropy(target_threshold=0.0)

    # ── Per-parameter weight decay groups (DeiT-III convention) ──
    # Bias and 1-D params (LayerNorm scale, LayerScale, etc.) get NO
    # weight decay, matching timm's add_weight_decay() helper.
    decay_params, no_decay_params = [], []
    for name, p in raw.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    param_groups = [
        {"params": decay_params,    "weight_decay": args.wd},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if args.is_main:
        print(f"  WD groups: {len(decay_params)} decay, "
              f"{len(no_decay_params)} no-decay")

    # LAMB optimizer (DeiT-III default)
    try:
        from apex.optimizers import FusedLAMB
        optimizer = FusedLAMB(param_groups, lr=args.lr)
        if args.is_main:
            print("Using FusedLAMB (apex)")
    except ImportError:
        optimizer = optim.AdamW(param_groups, lr=args.lr)
        if args.is_main:
            print("LAMB not available, using AdamW")

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # Cosine schedule — counted in OPTIMIZER steps (not data batches)
    opt_steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = opt_steps_per_epoch * args.epochs
    warmup_steps = opt_steps_per_epoch * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-6 / args.lr, step / max(warmup_steps, 1))
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(1e-5 / args.lr, 0.5 * (1 + math.cos(math.pi * t)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume — auto-detect last.pt unless --resume given
    start_epoch, best_acc = 0, 0.0
    last_path = f"{args.output_dir}/last.pt"
    resume_path = args.resume
    if resume_path is None and os.path.exists(last_path):
        resume_path = last_path
    if resume_path is not None and os.path.exists(resume_path):
        if args.is_main:
            print(f"  → resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        if "rng_torch" in ckpt:
            torch.set_rng_state(ckpt["rng_torch"].cpu())
            if torch.cuda.is_available() and "rng_cuda" in ckpt:
                torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng_cuda"]])
        if args.is_main:
            print(f"  → resumed: start_epoch={start_epoch}  best_acc={best_acc:.2f}%")

    # Wandb
    if args.wandb and args.is_main and HAS_WANDB:
        wandb.init(project="nelu", group="imagenet",
                   name=f"{args.model}_{args.act}", config=vars(args))

    # Train
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                               scaler, mixup_fn, criterion, device,
                               epoch, args, raw)

        t1, t5 = evaluate(model, val_loader, device, args)
        is_best = t1 > best_acc
        best_acc = max(best_acc, t1)

        if args.is_main:
            print(f"[{epoch+1}/{args.epochs}] loss={loss:.4f} "
                  f"top1={t1:.2f}% top5={t5:.2f}% best={best_acc:.2f}%")
            ckpt = {
                "model": raw.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if args.amp else None,
                "epoch": epoch,
                "best_acc": best_acc,
                "rng_torch": torch.get_rng_state(),
            }
            if torch.cuda.is_available():
                ckpt["rng_cuda"] = torch.cuda.get_rng_state_all()
            tmp = f"{args.output_dir}/last.pt.tmp"
            torch.save(ckpt, tmp)
            os.replace(tmp, f"{args.output_dir}/last.pt")
            if is_best:
                torch.save(ckpt, f"{args.output_dir}/best.pt")
            if args.wandb and HAS_WANDB:
                wandb.log({"epoch": epoch, "train_loss": loss,
                           "val_top1": t1, "val_top5": t5,
                           "best_top1": best_acc,
                           "lr": optimizer.param_groups[0]["lr"]})

    if args.is_main:
        print(f"\nBest: {best_acc:.2f}%")
        with open(f"{args.output_dir}/result.json", "w") as f:
            json.dump({"model": args.model, "act": args.act,
                       "best_top1": best_acc}, f, indent=2)

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
