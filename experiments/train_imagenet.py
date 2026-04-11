#!/usr/bin/env python3
"""[DEPRECATED — kept only for MODEL_CFGS reference] Custom ImageNet trainer.

ImageNet §4.3 runs were moved to bit-exact upstream stacks:
    ConvNeXt-T   → /home/ubuntu/convnext-train/main.py     (FB ConvNeXt)
    EffNet-B2    → /home/ubuntu/NELU/timm-train/train.py   (timm/train.py)
                   via experiments/train_imagenet_timm.py wrapper
    DeiT-III B   → /home/ubuntu/deit-train/main.py         (FB deit)

Each clone has been patched with `--act` activation swap. The recipes
encoded here in MODEL_CFGS are now superseded by the documented CLIs in
run_h100.sh phases 3-5. This file is left in place because eval helpers
may still consult its dict. Do not invoke as an entrypoint.

Original docstring follows.
================================================================
ImageNet-1k training — three architectures × two activations.

Used for §4.3 of the paper:

    ConvNeXt-T   GELU (timm pretrained) vs NELU (from scratch)
    EffNet-B2    SiLU (timm pretrained) vs NiLU (from scratch)
    DeiT-III B   GELU (timm pretrained) vs NELU (from scratch)

Baselines are timm pretrained checkpoints — they are NOT trained
here, only evaluated via `--eval-baseline`. Only the OUR variant
(NELU / NiLU) is trained from scratch using a recipe matched to
the original training procedure for each architecture.

Per-arch recipes are encoded in MODEL_CFGS. Optimizer / scheduler /
augmentation use timm helpers so the recipes follow the published
configs without hand-coded reimplementations.

Usage (8-GPU H100/A100):
    # Baseline pretrained eval
    python experiments/train_imagenet.py --model convnext_tiny --eval-baseline \
        --data /data/imagenet
    # Train OUR variant from scratch
    torchrun --nproc_per_node=8 experiments/train_imagenet.py \
        --model convnext_tiny --act nelu --data /data/imagenet --wandb
"""

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

# torch.compile safety net: long DDP runs must not die from a Dynamo error.
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 512

import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from tqdm import tqdm

import timm
from timm.data import create_transform, Mixup
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import ModelEmaV3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nelu import NELU, NiLU, NELUCUDA, NiLUCUDA

# Prefer fused CUDA kernels when available, fall back to Python+compile.
_NELU_CLS = NELUCUDA if NELUCUDA is not None else NELU
_NILU_CLS = NiLUCUDA if NiLUCUDA is not None else NiLU

_ACT_CLS = {
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "nelu": _NELU_CLS,
    "nilu": _NILU_CLS,
}

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ── Per-arch recipes ─────────────────────────────────────────────
#
# Each cfg encodes the published training recipe for that arch.
# Field meanings:
#   timm_name        : timm model id (no pretrained weights)
#   timm_pretrained  : timm id with pretrained ImageNet weights
#   baseline_act     : activation in the pretrained model — replaced
#                      by our variant (gelu→nelu, silu→nilu)
#   train_size       : training resolution
#   eval_size        : eval resolution
#   eval_crop_pct    : eval-time center-crop ratio
#   batch_per_gpu    : per-GPU micro batch
#   grad_accum       : gradient accumulation steps
#   epochs / warmup_epochs
#   opt / lr / wd / opt_kwargs
#   sched            : timm scheduler name
#   sched_kwargs     : extra kwargs for create_scheduler_v2
#   loss             : "bce" | "ce_ls"
#   label_smoothing
#   mixup / cutmix
#   auto_augment     : timm auto-augment string ("3aug" = DeiT-III
#                      3-Augment, otherwise rand-augment policy)
#   color_jitter     : ColorJitter strength
#   re_prob          : random-erase probability
#   ema / ema_decay
#   clip_grad
#
MODEL_CFGS = {
    # ─── ConvNeXt-Tiny: ConvNeXt paper recipe (Liu et al., 2022) ────
    "convnext_tiny": {
        "timm_name":       "convnext_tiny",
        "timm_pretrained": "convnext_tiny.fb_in1k",
        "baseline_act":    "gelu",
        "drop_path":       0.1,
        "train_size":      224,
        "eval_size":       224,
        "eval_crop_pct":   0.875,
        "batch_per_gpu":   128,
        "grad_accum":      4,           # 128 × 4 × 8 = 4096 effective
        "epochs":          300,
        "warmup_epochs":   20,
        "opt":             "adamw",
        "opt_kwargs":      {},
        "lr":              4e-3,
        "wd":              0.05,
        "sched":           "cosine",
        "sched_kwargs":    {"min_lr": 1e-6, "warmup_lr": 1e-6},
        "loss":            "ce_ls",
        "label_smoothing": 0.1,
        "mixup":           0.8,
        "cutmix":          1.0,
        "auto_augment":    "rand-m9-mstd0.5-inc1",
        "color_jitter":    0.4,
        "re_prob":         0.25,
        "ema":             True,
        "ema_decay":       0.9999,
        "clip_grad":       None,
    },
    # ─── EfficientNet-B2: timm `ra_in1k` recipe ─────────────────────
    # Matches the Wightman RA recipe used for `efficientnet_b2.ra_in1k`
    # (same family as b0.ra_in1k, just B2's image size + dropout):
    #   --opt rmsproptf --opt-eps .001 --weight-decay 1e-5
    #   --sched step --decay-rate .97 --decay-epochs 2.4
    #   --warmup-epochs 5 --warmup-lr 1e-6 --epochs 450
    #   --aa rand-m9-mstd0.5 --reprob 0.2 --remode pixel
    #   --drop 0.3 --drop-connect 0.2
    #   --model-ema --model-ema-decay 0.9999
    # train@256 / eval@288 / crop_pct=1.0 — timm B2 default cfg.
    # No mixup. RMSpropTF momentum=0.9 (timm default).
    "efficientnet_b2": {
        "timm_name":       "efficientnet_b2",
        "timm_pretrained": "efficientnet_b2.ra_in1k",
        "baseline_act":    "silu",
        "drop_path":       0.2,            # --drop-connect 0.2
        "drop_rate":       0.3,            # --drop 0.3 (classifier)
        "train_size":      256,            # timm B2 train_size
        "eval_size":       288,            # timm B2 test_input_size
        "eval_crop_pct":   1.0,
        "batch_per_gpu":   384,            # 384 × 8 = 3072 effective
        "grad_accum":      1,
        "epochs":          450,
        "warmup_epochs":   5,
        "opt":             "rmsproptf",
        "opt_kwargs":      {"eps": 0.001, "alpha": 0.9, "momentum": 0.9},
        "lr":              0.048,          # same as B0 ra (matched bs 3072)
        "wd":              1e-5,
        "sched":           "step",
        "sched_kwargs":    {"decay_rate": 0.97, "decay_epochs": 2.4,
                            "min_lr": 1e-6, "warmup_lr": 1e-6},
        "loss":            "ce_ls",
        "label_smoothing": 0.1,
        "mixup":           0.0,
        "cutmix":          0.0,
        "auto_augment":    "rand-m9-mstd0.5",
        "color_jitter":    0.4,
        "re_prob":         0.2,
        "ema":             True,
        "ema_decay":       0.9999,
        "clip_grad":       None,
    },
    # ─── DeiT-III ViT-B: README_revenge config ──────────────────────
    "deit3_base": {
        "timm_name":       "deit3_base_patch16_224",
        "timm_pretrained": "deit3_base_patch16_224.fb_in1k",
        "baseline_act":    "gelu",
        "drop_path":       0.2,
        "train_size":      192,
        "eval_size":       224,
        "eval_crop_pct":   1.0,
        "batch_per_gpu":   128,
        "grad_accum":      2,           # 128 × 2 × 8 = 2048 effective
        "epochs":          800,
        "warmup_epochs":   5,
        "opt":             "fusedlamb",  # falls back to lamb / adamw
        "opt_kwargs":      {},
        "lr":              3e-3,
        "wd":              0.05,
        "sched":           "cosine",
        "sched_kwargs":    {"min_lr": 1e-5, "warmup_lr": 1e-6},
        "loss":            "bce",
        "label_smoothing": 0.0,
        "mixup":           0.8,
        "cutmix":          1.0,
        "auto_augment":    "3aug_src",   # DeiT-III 3-Augment + SRC
        "color_jitter":    0.3,
        "re_prob":         0.0,
        "ema":             False,
        "ema_decay":       0.0,
        "clip_grad":       1.0,
    },
}


# ── Activation replacement ───────────────────────────────────────

def replace_act(model, baseline_act, our_act):
    """Replace every instance of the baseline activation with ours."""
    if our_act == baseline_act:
        return model
    src_cls = _ACT_CLS[baseline_act]
    dst_cls = _ACT_CLS[our_act]
    for name, child in model.named_children():
        if isinstance(child, src_cls):
            setattr(model, name, dst_cls())
        else:
            replace_act(child, baseline_act, our_act)
    return model


# ── 3-Augment (DeiT-III) ─────────────────────────────────────────

class _ThreeAugment:
    """Randomly apply one of: grayscale, solarize, gaussian blur.
    Operates on PIL images, mirrors timm/DeiT-III convention."""
    def __call__(self, img):
        op = torch.randint(0, 3, (1,)).item()
        if op == 0:
            from torchvision.transforms import functional as F
            return F.rgb_to_grayscale(img, num_output_channels=3)
        if op == 1:
            from torchvision.transforms import functional as F
            return F.solarize(img, threshold=128)
        from PIL import ImageFilter
        return img.filter(ImageFilter.GaussianBlur(
            radius=torch.empty(1).uniform_(0.1, 2.0).item()))


def build_train_transform(cfg):
    """Recipe-aware training transform.

    For most archs we delegate to timm's create_transform with the
    `auto_augment` policy from cfg. For DeiT-III's "3aug_src" we
    reproduce the official Simple-Random-Crop + 3-Augment pipeline
    from facebookresearch/deit datasets.py:new_data_aug_generator
    (the `--src` branch). timm's create_transform doesn't expose
    either as an auto_augment string, so we build it manually."""
    from torchvision import transforms as T

    if cfg["auto_augment"] == "3aug_src":
        # Matches deit/datasets.py new_data_aug_generator(--src):
        #   Resize(short=img_size) → RandomCrop(img_size, pad=4, reflect)
        #   → HFlip → 3-Augment (RandomChoice[gray|solarize|blur])
        #   → ColorJitter → ToTensor → Normalize
        return T.Compose([
            T.Resize(cfg["train_size"],
                     interpolation=T.InterpolationMode.BICUBIC),
            T.RandomCrop(cfg["train_size"], padding=4,
                         padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            _ThreeAugment(),
            T.ColorJitter(cfg["color_jitter"],
                          cfg["color_jitter"],
                          cfg["color_jitter"]),
            T.ToTensor(),
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

    return create_transform(
        input_size=cfg["train_size"],
        is_training=True,
        color_jitter=cfg["color_jitter"],
        auto_augment=cfg["auto_augment"],
        interpolation="bicubic",
        re_prob=cfg["re_prob"],
        re_mode="pixel",
        re_count=1,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )


def build_val_transform(cfg):
    return create_transform(
        input_size=cfg["eval_size"],
        is_training=False,
        interpolation="bicubic",
        crop_pct=cfg["eval_crop_pct"],
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )


# ── Data ─────────────────────────────────────────────────────────

def _worker_init_fn(worker_id):
    import numpy as _np
    import random as _rnd
    base = torch.initial_seed() % (2**32)
    _np.random.seed(base + worker_id)
    _rnd.seed(base + worker_id)


def build_dataloaders(args, cfg):
    train_tf = build_train_transform(cfg)
    val_tf = build_val_transform(cfg)

    train_ds = datasets.ImageFolder(os.path.join(args.data, "train"), train_tf)
    val_ds = datasets.ImageFolder(os.path.join(args.data, "val"), val_tf)

    if args.distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = val_sampler = None

    gen = torch.Generator()
    gen.manual_seed(args.seed + args.rank)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None), num_workers=args.workers,
        pin_memory=True, sampler=train_sampler, drop_last=True,
        worker_init_fn=_worker_init_fn, generator=gen,
        persistent_workers=args.workers > 0)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
        sampler=val_sampler, worker_init_fn=_worker_init_fn,
        persistent_workers=args.workers > 0)
    return train_loader, val_loader, train_sampler


# ── Optimizer / scheduler / loss / mixup ─────────────────────────

def build_optimizer(cfg, raw):
    """Use timm's create_optimizer_v2 — handles per-param WD groups
    (no decay on bias/norm) and FusedLAMB fallback."""
    opt_name = cfg["opt"]
    if opt_name == "fusedlamb":
        try:
            from apex.optimizers import FusedLAMB  # noqa: F401
            opt_name = "fusedlamb"
        except ImportError:
            opt_name = "lamb"
    return create_optimizer_v2(
        raw,
        opt=opt_name,
        lr=cfg["lr"],
        weight_decay=cfg["wd"],
        filter_bias_and_bn=True,
        **cfg["opt_kwargs"],
    )


def build_scheduler(cfg, optimizer):
    sched, num_epochs = create_scheduler_v2(
        optimizer,
        sched=cfg["sched"],
        num_epochs=cfg["epochs"],
        warmup_epochs=cfg["warmup_epochs"],
        **cfg["sched_kwargs"],
    )
    return sched


def build_criterion(cfg, mixup_active):
    if cfg["loss"] == "bce":
        # DeiT-III uses BCE with NO target thresholding so soft labels
        # from mixup pass through unchanged. target_threshold=0.0 would
        # binarize every positive component to 1.0 — wrong for mixup.
        return BinaryCrossEntropy(target_threshold=None,
                                  smoothing=cfg["label_smoothing"])
    # ce_ls — when mixup is active timm's mixup already produces soft
    # targets, so plain SoftTargetCrossEntropy is appropriate; otherwise
    # use LabelSmoothingCE.
    if mixup_active:
        from timm.loss import SoftTargetCrossEntropy
        return SoftTargetCrossEntropy()
    if cfg["label_smoothing"] > 0:
        return LabelSmoothingCrossEntropy(smoothing=cfg["label_smoothing"])
    return nn.CrossEntropyLoss()


def build_mixup(cfg):
    if cfg["mixup"] > 0 or cfg["cutmix"] > 0:
        return Mixup(
            mixup_alpha=cfg["mixup"],
            cutmix_alpha=cfg["cutmix"],
            label_smoothing=cfg["label_smoothing"],
            num_classes=1000,
        ), True
    return None, False


# ── Training ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, scaler,
                    mixup_fn, criterion, device, epoch, args, raw,
                    ema, cfg):
    """Training loop with gradient accumulation. Per-step LR via
    timm scheduler.step_update."""
    model.train()
    accum = max(1, args.grad_accum)
    total_loss, total = 0.0, 0
    pbar = tqdm(loader, desc=f"train {epoch}",
                disable=not args.is_main, ncols=100)

    optimizer.zero_grad(set_to_none=True)
    accum_count = 0
    num_updates = epoch * (len(loader) // accum)

    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        is_last_micro = (accum_count + 1) == accum
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
            if cfg["clip_grad"] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    raw.parameters(), max_norm=cfg["clip_grad"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            num_updates += 1
            if scheduler is not None and hasattr(scheduler, "step_update"):
                scheduler.step_update(num_updates=num_updates)
            if ema is not None:
                ema.update(raw)
            accum_count = 0

        pbar.set_postfix(loss=f"{total_loss/total:.4f}")

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
    parser.add_argument("--model", default="convnext_tiny",
                        choices=list(MODEL_CFGS.keys()))
    parser.add_argument("--act", default=None,
                        choices=["gelu", "silu", "nelu", "nilu"],
                        help="Activation. Defaults to baseline_act for "
                             "the chosen model.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override cfg epochs.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Per-GPU micro batch (auto from cfg).")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="Gradient accumulation steps (auto from cfg).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--eval-only", action="store_true",
                        help="Eval only — load weights from --resume.")
    parser.add_argument("--eval-baseline", action="store_true",
                        help="Eval the timm pretrained baseline (no train).")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = MODEL_CFGS[args.model]

    # Default activation: baseline if eval-baseline, else baseline_act.
    if args.eval_baseline:
        args.act = cfg["baseline_act"]
    elif args.act is None:
        args.act = cfg["baseline_act"]

    if args.batch_size is None:
        args.batch_size = cfg["batch_per_gpu"]
    if args.grad_accum is None:
        args.grad_accum = cfg.get("grad_accum", 1)
    if args.epochs is not None:
        cfg = {**cfg, "epochs": args.epochs}
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

    import random as _rnd
    import numpy as _np
    seed_for_rank = args.seed + args.rank
    torch.manual_seed(seed_for_rank)
    torch.cuda.manual_seed_all(seed_for_rank)
    _np.random.seed(seed_for_rank)
    _rnd.seed(seed_for_rank)
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)

    if args.is_main:
        eff = args.batch_size * args.grad_accum * args.world_size
        print(f"{args.model} + {args.act} | "
              f"micro {args.batch_size} × accum {args.grad_accum} × "
              f"{args.world_size} GPU = {eff} effective | "
              f"{cfg['epochs']} epochs")

    # Model
    use_pretrained = args.eval_baseline and args.act == cfg["baseline_act"]
    if use_pretrained:
        model = timm.create_model(cfg["timm_pretrained"], pretrained=True)
    else:
        create_kwargs = dict(
            pretrained=False, num_classes=1000,
            drop_path_rate=cfg["drop_path"])
        if "drop_rate" in cfg:
            create_kwargs["drop_rate"] = cfg["drop_rate"]
        model = timm.create_model(cfg["timm_name"], **create_kwargs)
    model = replace_act(model, cfg["baseline_act"], args.act)
    model = model.to(device)

    if args.compile:
        model = torch.compile(model)

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank])
    raw = model.module if args.distributed else model
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod

    if args.is_main:
        n = sum(p.numel() for p in raw.parameters()) / 1e6
        print(f"Params: {n:.1f}M")

    # Eval-only / eval-baseline path
    if args.eval_only or args.eval_baseline:
        _, val_loader, _ = build_dataloaders(args, cfg)
        if args.eval_only and args.resume:
            ckpt = torch.load(args.resume, map_location=device,
                              weights_only=False)
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
            raw.load_state_dict(state, strict=False)
            if args.is_main:
                print(f"  loaded {args.resume}")
        t1, t5 = evaluate(model, val_loader, device, args)
        if args.is_main:
            print(f"top1={t1:.2f}%  top5={t5:.2f}%")
        return

    # Data
    train_loader, val_loader, train_sampler = build_dataloaders(args, cfg)

    # Mixup + criterion
    mixup_fn, mixup_active = build_mixup(cfg)
    criterion = build_criterion(cfg, mixup_active)

    # Optimizer + scheduler
    optimizer = build_optimizer(cfg, raw)
    scheduler = build_scheduler(cfg, optimizer)

    if args.is_main:
        print(f"  opt={cfg['opt']}  sched={cfg['sched']}  "
              f"loss={cfg['loss']}  mixup={cfg['mixup']}/{cfg['cutmix']}  "
              f"aug={cfg['auto_augment']}  ema={cfg['ema']}")

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # EMA
    ema = None
    if cfg["ema"]:
        ema = ModelEmaV3(raw, decay=cfg["ema_decay"], device=device)

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
        if "scheduler" in ckpt and scheduler is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                pass
        if "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if "ema" in ckpt and ema is not None and ckpt["ema"] is not None:
            ema.module.load_state_dict(ckpt["ema"])
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
        try:
            wandb.init(project="nelu", group="imagenet",
                       name=f"{args.model}_{args.act}", config=vars(args))
        except Exception as e:
            print(f"  WARN: wandb init failed: {e}")

    # Train
    for epoch in range(start_epoch, cfg["epochs"]):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            mixup_fn, criterion, device, epoch, args, raw, ema, cfg)

        if scheduler is not None and hasattr(scheduler, "step"):
            scheduler.step(epoch + 1)

        eval_model = ema.module if ema is not None else model
        t1, t5 = evaluate(eval_model, val_loader, device, args)
        is_best = t1 > best_acc
        best_acc = max(best_acc, t1)

        if args.is_main:
            print(f"[{epoch+1}/{cfg['epochs']}] loss={loss:.4f} "
                  f"top1={t1:.2f}% top5={t5:.2f}% best={best_acc:.2f}%")
            ckpt = {
                "model": raw.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": (scheduler.state_dict()
                              if scheduler is not None and
                              hasattr(scheduler, "state_dict") else None),
                "scaler": scaler.state_dict() if args.amp else None,
                "ema": ema.module.state_dict() if ema is not None else None,
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
                try:
                    wandb.log({"epoch": epoch, "train_loss": loss,
                               "val_top1": t1, "val_top5": t5,
                               "best_top1": best_acc,
                               "lr": optimizer.param_groups[0]["lr"]})
                except Exception:
                    pass

    if args.is_main:
        print(f"\nBest: {best_acc:.2f}%")
        with open(f"{args.output_dir}/result.json", "w") as f:
            json.dump({"model": args.model, "act": args.act,
                       "best_top1": best_acc}, f, indent=2)

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
