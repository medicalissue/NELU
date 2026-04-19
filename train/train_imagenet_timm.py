"""ImageNet training for EfficientNet with NELU/NiLU activation swap.

Self-contained distributed training script that uses timm for model
creation and data loading, with a custom training loop that supports
gamma/entropy logging for gated activations.

Usage:
    torchrun --nproc_per_node=8 train/train_imagenet_timm.py \
        --model efficientnet_b2 --activation nilu \
        --data-dir /data/imagenet --output results/effnet_b2_nilu \
        --epochs 450 --lr 0.016 --opt rmsproptf --sched step \
        --decay-epochs 2.4 --decay-rate 0.97 --amp

Requires: timm, torch >= 2.0
"""

import argparse
import json
import math
import os
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import timm
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import model_parameters
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import ModelEmaV3, AverageMeter, accuracy, reduce_tensor

from nelu import NELU, NiLU, collect_gamma_stats
from train.act_swap import replace_activation, swap_gelu_to_nelu, swap_silu_to_nilu


# ---------------------------------------------------------------------------
#  Activation swap logic
# ---------------------------------------------------------------------------

_SWAP_TABLE = {
    "nelu": (nn.GELU, NELU, "GELU -> NELU"),
    "nilu": (nn.SiLU, NiLU, "SiLU -> NiLU"),
    "gelu": None,  # no swap needed, model already uses GELU
    "silu": None,  # no swap needed, model already uses SiLU
    "relu": None,
}


def apply_activation_swap(model, activation, **kwargs):
    """Apply activation swap and return the count of replaced modules."""
    entry = _SWAP_TABLE.get(activation)
    if entry is None:
        return 0
    src_cls, tgt_cls, desc = entry
    n = replace_activation(model, src_cls, tgt_cls, **kwargs)
    if n == 0:
        # Try swapping ReLU as fallback
        n = replace_activation(model, nn.ReLU, tgt_cls, **kwargs)
        if n > 0:
            desc = f"ReLU -> {tgt_cls.__name__}"
    print(f"[activation swap] {desc}: replaced {n} modules")
    return n


# ---------------------------------------------------------------------------
#  Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ImageNet training with activation swap (timm-based)")

    # Custom args
    p.add_argument("--activation", type=str, default="silu",
                   choices=["relu", "gelu", "silu", "nelu", "nilu"],
                   help="Activation function to use")
    p.add_argument("--gamma-init", type=float, default=1e-4,
                   help="Initial gamma for NELU/NiLU")
    p.add_argument("--config", type=str, default=None,
                   help="YAML config file (overrides CLI defaults)")
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--wandb-project", type=str, default="nelu-imagenet")

    # Model
    p.add_argument("--model", type=str, default="efficientnet_b0")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--drop", type=float, default=0.0, help="Dropout rate")
    p.add_argument("--drop-connect", type=float, default=0.0,
                   help="Drop connect rate (EfficientNet)")
    p.add_argument("--drop-path", type=float, default=0.0, help="Drop path rate")

    # Data
    p.add_argument("--data-dir", type=str, default="/data/imagenet")
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("-b", "--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)

    # Augmentation
    p.add_argument("--aa", type=str, default=None, help="AutoAugment policy")
    p.add_argument("--remode", type=str, default="pixel", help="Random erasing mode")
    p.add_argument("--reprob", type=float, default=0.0, help="Random erasing prob")
    p.add_argument("--color-jitter", type=float, default=0.0)
    p.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing")

    # Optimizer
    p.add_argument("--opt", type=str, default="rmsproptf")
    p.add_argument("--opt-eps", type=float, default=0.001)
    p.add_argument("--lr", type=float, default=0.016)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--momentum", type=float, default=0.9)

    # Scheduler
    p.add_argument("--sched", type=str, default="step")
    p.add_argument("--decay-epochs", type=float, default=2.4)
    p.add_argument("--decay-rate", type=float, default=0.97)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--warmup-lr", type=float, default=1e-6)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--epochs", type=int, default=450)

    # Training
    p.add_argument("--amp", action="store_true", help="Use AMP (mixed precision)")
    p.add_argument("--model-ema", action="store_true")
    p.add_argument("--model-ema-decay", type=float, default=0.9999)
    p.add_argument("--clip-grad", type=float, default=None)

    # Checkpointing
    p.add_argument("--output", type=str, default="results/imagenet")
    p.add_argument("--resume", type=str, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
#  Distributed setup
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialize distributed training from torchrun environment variables."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def is_primary(rank):
    return rank == 0


# ---------------------------------------------------------------------------
#  Training and evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(epoch, model, loader, optimizer, loss_fn, device, amp_autocast,
                    scaler, model_ema, clip_grad, rank, world_size):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    num_updates = epoch * len(loader)

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        with amp_autocast:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_grad is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        if world_size > 1:
            loss = reduce_tensor(loss, world_size)
            acc1 = reduce_tensor(acc1, world_size)
            acc5 = reduce_tensor(acc5, world_size)

        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))
        num_updates += 1

    return OrderedDict(loss=losses.avg, top1=top1.avg, top5=top5.avg)


@torch.no_grad()
def validate(model, loader, loss_fn, device, amp_autocast, rank, world_size):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with amp_autocast:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        if world_size > 1:
            loss = reduce_tensor(loss, world_size)
            acc1 = reduce_tensor(acc1, world_size)
            acc5 = reduce_tensor(acc5, world_size)

        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

    return OrderedDict(loss=losses.avg, top1=top1.avg, top5=top5.avg)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load YAML config if provided
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            attr = k.replace("-", "_")
            if hasattr(args, attr) and attr != "config":
                setattr(args, attr, v)

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if is_primary(rank):
        os.makedirs(args.output, exist_ok=True)
        print(f"Training {args.model} with activation={args.activation}")
        print(f"World size: {world_size}, rank: {rank}")

    # -- Model --
    model = timm.create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_connect_rate=args.drop_connect,
    )

    # Apply activation swap
    n_swapped = apply_activation_swap(model, args.activation, gamma_init=args.gamma_init)
    if is_primary(rank):
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,}, swapped activations: {n_swapped}")

    model = model.to(device)

    # EMA
    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV3(model, decay=args.model_ema_decay, device=device)

    # DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # -- Data --
    data_config = resolve_data_config(vars(args), model=model)
    if args.input_size:
        data_config["input_size"] = (3, args.input_size, args.input_size)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    dataset_train = create_dataset("", root=train_dir, is_training=True)
    dataset_val = create_dataset("", root=val_dir, is_training=False)

    loader_train = create_loader(
        dataset_train, input_size=data_config["input_size"],
        batch_size=args.batch_size, is_training=True,
        re_prob=args.reprob, re_mode=args.remode,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_workers=args.workers,
        distributed=world_size > 1,
        pin_memory=True,
    )
    loader_val = create_loader(
        dataset_val, input_size=data_config["input_size"],
        batch_size=args.batch_size, is_training=False,
        num_workers=args.workers,
        distributed=world_size > 1,
        pin_memory=True,
    )

    # -- Optimizer & scheduler --
    optimizer = create_optimizer_v2(model, opt=args.opt, lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    momentum=args.momentum,
                                    eps=args.opt_eps)

    scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        sched=args.sched,
        num_epochs=args.epochs,
        decay_epochs=args.decay_epochs,
        decay_rate=args.decay_rate,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=args.warmup_lr,
        min_lr=args.min_lr,
    )

    # -- Loss --
    if args.smoothing > 0:
        from timm.loss import LabelSmoothingCrossEntropy
        loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    # -- AMP --
    scaler = None
    amp_autocast = torch.cuda.amp.autocast(enabled=False)
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
        amp_autocast = torch.cuda.amp.autocast()

    # -- Resume --
    start_epoch = 0
    best_acc = 0.0

    if args.resume and os.path.isfile(args.resume):
        if is_primary(rank):
            print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        if model_ema is not None and "model_ema" in ckpt:
            model_ema.load_state_dict(ckpt["model_ema"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

    # -- wandb --
    wandb_run = None
    if args.wandb and is_primary(rank):
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"{args.model}_{args.activation}",
                config=vars(args),
            )
        except ImportError:
            print("WARNING: wandb not installed")

    # -- Training loop --
    if is_primary(rank):
        print(f"\nStarting training from epoch {start_epoch} to {args.epochs}")

    for epoch in range(start_epoch, args.epochs):
        if world_size > 1 and hasattr(loader_train, "sampler"):
            loader_train.sampler.set_epoch(epoch)

        t0 = time.time()
        train_metrics = train_one_epoch(
            epoch, model, loader_train, optimizer, loss_fn, device,
            amp_autocast, scaler, model_ema, args.clip_grad, rank, world_size,
        )

        # Evaluate (use EMA model if available)
        eval_model = model_ema.module if model_ema is not None else model
        val_metrics = validate(eval_model, loader_val, loss_fn, device,
                               amp_autocast, rank, world_size)

        scheduler.step(epoch + 1)
        elapsed = time.time() - t0

        is_best = val_metrics["top1"] > best_acc
        if is_best:
            best_acc = val_metrics["top1"]

        # Gamma stats
        gamma_stats = {}
        if args.activation in ("nelu", "nilu") and is_primary(rank):
            raw_model = model.module if hasattr(model, "module") else model
            gamma_stats = collect_gamma_stats(raw_model)

        if is_primary(rank):
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"train loss {train_metrics['loss']:.4f} top1 {train_metrics['top1']:.2f} | "
                  f"val loss {val_metrics['loss']:.4f} top1 {val_metrics['top1']:.2f} "
                  f"top5 {val_metrics['top5']:.2f} | "
                  f"best {best_acc:.2f} | {elapsed:.1f}s", end="")
            if gamma_stats:
                print(f" | gamma_mean {gamma_stats.get('nelu/gamma/mean', 0):.4f}", end="")
            print()

            if wandb_run:
                import wandb
                log_data = {
                    "train/loss": train_metrics["loss"],
                    "train/top1": train_metrics["top1"],
                    "val/loss": val_metrics["loss"],
                    "val/top1": val_metrics["top1"],
                    "val/top5": val_metrics["top5"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
                log_data.update(gamma_stats)
                wandb.log(log_data, step=epoch)

            # Save checkpoint
            raw_model = model.module if hasattr(model, "module") else model
            state = {
                "epoch": epoch,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
                "args": vars(args),
            }
            if model_ema is not None:
                state["model_ema"] = model_ema.state_dict()
            if scaler is not None:
                state["scaler"] = scaler.state_dict()

            torch.save(state, os.path.join(args.output, "checkpoint.pt"))
            if is_best:
                torch.save(state, os.path.join(args.output, "checkpoint-best.pt"))

    # Final summary
    if is_primary(rank):
        print(f"\nTraining complete. Best top-1: {best_acc:.2f}%")
        result = {
            "model": args.model,
            "activation": args.activation,
            "best_top1": best_acc,
            "epochs": args.epochs,
        }
        with open(os.path.join(args.output, "result.json"), "w") as f:
            json.dump(result, f, indent=2)

        if wandb_run:
            import wandb
            wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
