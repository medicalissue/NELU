"""MedMNIST v2 training with activation comparison (data-scarce regime).

Separate entry point from ``train/cifar.py`` so the verified CIFAR-100
campaign code is untouched. We reuse only the audited activation-swap
functions and the checkpoint/wandb/signal helpers; everything that differs
for MedMNIST — the 12 official 2D datasets, the four task types, the
per-task loss, the official ``Evaluator`` scoring, and best-val-AUC model
selection — lives here.

Protocol is matched 1:1 to the official MedMNIST v2 baseline so our
numbers drop straight next to Table 3 / Table 5 of Yang et al. (Sci Data
2022):

  * backbone  : torchvision ResNet-18 / ResNet-50, trained from scratch,
                 native 28x28 input (no upsampling, no pretrained weights)
  * optimizer : Adam, lr 1e-3
  * schedule  : MultiStepLR milestones [50, 75], gamma 0.1
  * epochs    : 100      batch size: 128
  * transform : ToTensor + Normalize(mean=[.5]*3, std=[.5]*3); 1-ch
                 datasets copied to 3 channels; NO crop/flip augmentation
                 (the reference script uses none — gains must hold here)
  * loss      : CrossEntropy for multi-class/binary/ordinal,
                 BCEWithLogits for the multi-label ChestMNIST
  * selection : checkpoint with the highest *validation AUC*; that
                 checkpoint is scored on the test split
  * metrics   : official ``medmnist.Evaluator`` -> (AUC, ACC)
  * repeats   : >=3 seeds; report per-dataset mean + 12-dataset average

Usage:
    python train/medmnist.py --dataset pathmnist --model resnet18 \
        --activation nelu --seed 42
"""

import argparse
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from gate_norm import NELU, NiLU
from train.swap import replace_activation, replace_activation_auto_axes

# Reuse the audited helpers verbatim — these are battle-tested in the CIFAR
# campaign and have no CIFAR-specific assumptions.
from train.cifar import (
    TrainingInterrupted,
    _orig_module,
    _request_interruption,
    init_wandb_run,
    interruption_requested,
    save_checkpoint,
)

# medmnist is the authoritative source for splits, channels, classes, task
# type, and the scoring contract. We never re-split or re-implement metrics.
import medmnist
from medmnist import INFO, Evaluator

# The 12 official 2D datasets, ordered by training-set size (ascending).
# This ordering is the natural data-scarcity axis for the headline figure:
# no artificial subsampling — the benchmark itself spans 546 -> 165k.
MEDMNIST2D_BY_TRAIN_SIZE = [
    "breastmnist",     #    546 train  (2-class)
    "retinamnist",     #  1,080 train  (5-class ordinal)
    "pneumoniamnist",  #  4,708 train  (2-class)
    "dermamnist",      #  7,007 train  (7-class)
    "bloodmnist",      # 11,959 train  (8-class)
    "organcmnist",     # 12,975 train  (11-class)
    "organsmnist",     # 13,932 train  (11-class)
    "organamnist",     # 34,561 train  (11-class)
    "chestmnist",      # 78,468 train  (14-label multi-label)
    "pathmnist",       # 89,996 train  (9-class, test = different center)
    "octmnist",        # 97,477 train  (4-class)
    "tissuemnist",     # 165,466 train (8-class)
]


# ---------------------------------------------------------------------------
#  Model factory — torchvision ResNet-18/50, activation-swapped
# ---------------------------------------------------------------------------

_SUPPORTED_MODELS = ("resnet18", "resnet50")


def _split_block_relus(model: nn.Module) -> int:
    """Give every torchvision residual block one ReLU module per call site.

    Stock torchvision ``BasicBlock``/``Bottleneck`` reuse a single
    ``self.relu`` at 2 (BasicBlock) or 3 (Bottleneck) points inside one
    forward pass. With a *scalar* ReLU that is harmless, but a per-channel
    activation (NELU/NiLU carry length-C learnable γ_c, β_c) cannot be
    shared across call sites whose channel counts differ — in Bottleneck
    the post-conv1/conv2 ReLUs see ``width`` channels while the post-add
    ReLU sees ``expansion*width``. Sharing one γ vector there is a hard
    shape error (observed: ``gamma.view([1,256,1,1])`` on a 64-ch input).

    Fix: replace each block's shared ``self.relu`` with independent
    ``relu1``/``relu2``[/``relu3``] modules and rebind ``forward`` so each
    call site has its own activation. The subsequent activation swap then
    installs a correctly-sized NELU per site. Returns #blocks rewired.
    The CIFAR ResNet family is unaffected — this only touches torchvision
    BasicBlock/Bottleneck instances.
    """
    from torchvision.models.resnet import BasicBlock, Bottleneck

    def basic_forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu2(out)

    def bottleneck_forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu3(out)

    n = 0
    for m in model.modules():
        if isinstance(m, BasicBlock):
            m.relu1 = nn.ReLU(inplace=True)
            m.relu2 = nn.ReLU(inplace=True)
            del m.relu
            m.forward = basic_forward.__get__(m, BasicBlock)
            n += 1
        elif isinstance(m, Bottleneck):
            m.relu1 = nn.ReLU(inplace=True)
            m.relu2 = nn.ReLU(inplace=True)
            m.relu3 = nn.ReLU(inplace=True)
            del m.relu
            m.forward = bottleneck_forward.__get__(m, Bottleneck)
            n += 1
    return n


def build_model(name, activation="relu", num_classes=10, *, gamma_init=1.0):
    """torchvision ResNet-18/50 from random init, with the activation swap.

    torchvision ResNets reuse one ``self.relu`` per block at multiple call
    sites; :func:`_split_block_relus` first splits those into independent
    per-site modules so per-channel NELU/NiLU γ_c is correctly sized
    (Bottleneck's post-conv vs post-add ReLUs see different channel
    counts). The 3-channel stem is left as-is (grayscale MedMNIST is
    tripled by the data pipeline). Post-activation ordering (Conv -> BN
    -> ReLU), same as the CIFAR ResNet family.
    """
    if name not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model: {name!r}. Supported: {_SUPPORTED_MODELS}"
        )
    ctor = getattr(torchvision.models, name)
    model = ctor(weights=None, num_classes=num_classes)

    if activation == "relu":
        return model

    # Split shared block ReLUs before swapping so each call site gets its
    # own correctly-sized activation.
    nb = _split_block_relus(model)
    if nb:
        print(f"Split shared block ReLUs in {nb} residual blocks")

    relu_types = (nn.ReLU,)
    if activation in {"gelu", "silu"}:
        factory = (lambda: nn.GELU()) if activation == "gelu" else (lambda: nn.SiLU())
        n = replace_activation(model, relu_types, factory)
        print(f"Swapped {n} ReLU -> {activation}")
    elif activation == "nelu":
        n = replace_activation_auto_axes(
            model, relu_types, NELU, activation_order="post",
            gamma_init=gamma_init,
        )
        print(f"Swapped {n} ReLU -> NELU (post-activation axes)")
    elif activation == "nilu":
        n = replace_activation_auto_axes(
            model, relu_types, NiLU, activation_order="post",
            gamma_init=gamma_init,
        )
        print(f"Swapped {n} ReLU -> NiLU (post-activation axes)")
    else:
        raise ValueError(f"Unknown activation: {activation}")

    # swap.py sizes per-channel γ_c/β_c from the *declaration-order*
    # adjacent conv. In torchvision Bottleneck the post-conv1 ReLU is
    # declared after conv3, so that heuristic assigns it conv3's channel
    # count (256) while its real input is conv1's (64) — a hard shape
    # error. Sidestep the heuristic entirely: reset γ_c/β_c to lazy and
    # let one dummy forward materialize each to its *actual* input
    # channel count (the module's _materialize uses z.size(channel_dim),
    # which is always correct). swap.py / ln_beta.py stay untouched, so
    # the CIFAR/ImageNet campaigns are unaffected.
    if activation in ("nelu", "nilu"):
        for mod in model.modules():
            if type(mod).__name__ in ("NELU", "NiLU"):
                mod.gamma = nn.UninitializedParameter()
                mod.beta = nn.UninitializedParameter()
        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(torch.zeros(2, 3, 28, 28))
        model.train(was_training)
    return model


# ---------------------------------------------------------------------------
#  Data — official medmnist package, official splits
# ---------------------------------------------------------------------------

def _task_of(flag):
    """Return one of: 'multi-class', 'binary-class', 'ordinal-regression',
    'multi-label'. ChestMNIST is the only multi-label dataset."""
    task = INFO[flag]["task"]
    if "multi-label" in task:
        return "multi-label"
    return task  # 'multi-class' | 'binary-class' | 'ordinal-regression'


def get_dataloaders(flag, data_dir, train_batch_size, val_batch_size=None,
                    num_workers=4):
    """Official MedMNIST2D loaders. Splits/labels come straight from the
    medmnist package; we never re-split.

    Normalization and (lack of) augmentation match the reference baseline
    exactly: ToTensor + Normalize(.5, .5), 1-ch -> 3-ch, no crop/flip.
    """
    if val_batch_size is None:
        val_batch_size = train_batch_size * 2

    info = INFO[flag]
    DataClass = getattr(medmnist, info["python_class"])

    # as_rgb=True copies 1-channel datasets to 3 channels so the stock
    # ResNet stem (Conv2d(3, ...)) is reused unchanged — matches baseline.
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    common = dict(transform=tfm, download=True, as_rgb=True, root=data_dir,
                  size=28)
    train_ds = DataClass(split="train", **common)
    val_ds = DataClass(split="val", **common)
    test_ds = DataClass(split="test", **common)

    train_loader = DataLoader(
        train_ds, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
#  Loss / targets — per task type
# ---------------------------------------------------------------------------

def _prepare_targets(targets, task):
    """medmnist labels are (N, 1) int for class tasks and (N, 14) for the
    multi-label ChestMNIST. Shape them for the loss."""
    if task == "multi-label":
        return targets.float()                 # (N, L) for BCEWithLogits
    return targets.squeeze(-1).long()          # (N,) for CrossEntropy


def _compute_loss(outputs, targets, task):
    if task == "multi-label":
        return F.binary_cross_entropy_with_logits(outputs, targets)
    return F.cross_entropy(outputs, targets)


def _scores_from_logits(outputs, task):
    """Normalized probabilities the official Evaluator expects:
    softmax for class/ordinal tasks, sigmoid for multi-label."""
    if task == "multi-label":
        return torch.sigmoid(outputs)
    return torch.softmax(outputs, dim=1)


# ---------------------------------------------------------------------------
#  Train / eval loops (AUC-aware; best-val-AUC selection)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, task,
                    scaler=None, use_amp=False, amp_dtype=torch.float16):
    model.train()
    total_loss = 0.0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if interruption_requested():
            raise TrainingInterrupted(
                f"Interruption requested at epoch {epoch}, step {batch_idx}."
            )
        inputs = inputs.to(device)
        targets = _prepare_targets(targets, task).to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            outputs = model(inputs)
            loss = _compute_loss(outputs, targets, task)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)
    scheduler.step()
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, device, flag, split, task, data_dir,
             use_amp=False, amp_dtype=torch.float16):
    """Score a split with the official medmnist Evaluator -> (AUC, ACC)."""
    model.eval()
    all_scores = []
    for inputs, _ in loader:
        inputs = inputs.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            outputs = model(inputs)
        scores = _scores_from_logits(outputs, task).float().cpu()
        all_scores.append(scores)
    y_score = torch.cat(all_scores, dim=0).numpy()
    evaluator = Evaluator(flag, split, size=28, root=data_dir)
    metrics = evaluator.evaluate(y_score)
    return float(metrics.AUC), float(metrics.ACC)


# ---------------------------------------------------------------------------
#  Args / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="MedMNIST v2 training with activation comparison"
    )
    p.add_argument("--dataset", type=str, required=True,
                   choices=MEDMNIST2D_BY_TRAIN_SIZE)
    p.add_argument("--model", type=str, default="resnet18",
                   choices=list(_SUPPORTED_MODELS))
    p.add_argument("--activation", type=str, default="relu",
                   choices=["relu", "gelu", "silu", "nelu", "nilu"])
    # Official MedMNIST baseline recipe — defaults match it exactly.
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--val_batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam"])
    p.add_argument("--milestones", type=int, nargs="+", default=[50, 75])
    p.add_argument("--lr_gamma", type=float, default=0.1)
    p.add_argument("--gamma_init", type=float, default=1.0,
                   help="Initial value of the learnable γ at step 0.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="results/medmnist")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--data_dir", type=str, default="/data/medmnist")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp_dtype", type=str, default="float16",
                   choices=["float16", "bfloat16"])
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str,
                   default="medmnist-gate-normalization")
    return p.parse_args()


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, _request_interruption)
    signal.signal(signal.SIGINT, _request_interruption)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    flag = args.dataset
    task = _task_of(flag)
    num_classes = len(INFO[flag]["label"])
    print(f"Dataset: {flag} | task: {task} | classes: {num_classes} | "
          f"train: {INFO[flag]['n_samples']['train']}")

    train_loader, val_loader, test_loader = get_dataloaders(
        flag, args.data_dir, args.batch_size, args.val_batch_size,
        args.num_workers,
    )

    model = build_model(
        args.model, args.activation, num_classes=num_classes,
        gamma_init=args.gamma_init,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, activation: {args.activation}, "
          f"params: {param_count:,}, device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.lr_gamma
    )

    use_amp = args.amp
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    saved_wandb_id = None
    start_epoch = 0
    best_val_auc = -1.0
    best_state = None
    training_log = []

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        _orig_module(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_auc = ckpt.get("best_acc", -1.0)
        training_log = ckpt.get("training_log", [])
        saved_wandb_id = ckpt.get("wandb_id")
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    wandb_run, saved_wandb_id = init_wandb_run(args, saved_wandb_id)

    interrupted = False
    try:
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, epoch,
                task, scaler=scaler if use_amp else None,
                use_amp=use_amp, amp_dtype=amp_dtype,
            )
            val_auc, val_acc = evaluate(
                model, val_loader, device, flag, "val", task,
                args.data_dir, use_amp=use_amp, amp_dtype=amp_dtype,
            )
            dt = time.time() - t0

            # Best-val-AUC model selection — the official contract.
            is_best = val_auc > best_val_auc
            if is_best:
                best_val_auc = val_auc
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in _orig_module(model).state_dict().items()
                }

            rec = {
                "epoch": epoch, "train_loss": train_loss,
                "val_auc": val_auc, "val_acc": val_acc,
                "best_val_auc": best_val_auc, "lr": scheduler.get_last_lr()[0],
                "epoch_time_s": dt,
            }
            training_log.append(rec)
            print(f"[{epoch:3d}/{args.epochs}] loss={train_loss:.4f} "
                  f"val_auc={val_auc:.4f} val_acc={val_acc:.4f} "
                  f"best_val_auc={best_val_auc:.4f} ({dt:.1f}s)")
            if wandb_run is not None:
                import wandb
                wandb.log(rec, step=epoch)

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": _orig_module(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_acc": best_val_auc,
                    "best_state": best_state,
                    "args": vars(args),
                    "training_log": training_log,
                    "wandb_id": saved_wandb_id,
                },
                args.output_dir,
            )
    except TrainingInterrupted as e:
        print(f"[interrupted] {e}")
        interrupted = True

    # Final scoring: load the best-val-AUC weights, score the test split
    # with the official Evaluator.
    if best_state is not None:
        _orig_module(model).load_state_dict(best_state)
    test_auc, test_acc = evaluate(
        model, test_loader, device, flag, "test", task, args.data_dir,
        use_amp=use_amp, amp_dtype=amp_dtype,
    )
    val_auc_final, val_acc_final = evaluate(
        model, val_loader, device, flag, "val", task, args.data_dir,
        use_amp=use_amp, amp_dtype=amp_dtype,
    )
    result = {
        "dataset": flag, "task": task, "model": args.model,
        "activation": args.activation, "seed": args.seed,
        "num_classes": num_classes,
        "train_size": INFO[flag]["n_samples"]["train"],
        "best_val_auc": best_val_auc,
        "test_auc": test_auc, "test_acc": test_acc,
        "val_auc": val_auc_final, "val_acc": val_acc_final,
        "epochs_run": len(training_log),
        "interrupted": interrupted,
    }
    out = Path(args.output_dir) / "result.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[result] test_auc={test_auc:.4f} test_acc={test_acc:.4f} "
          f"-> {out}")

    # Completion sentinel. The slot runner judges a job done by the
    # presence of ``<outdir>/complete`` (mirrors timm/CIFAR); only write
    # it when the run actually finished all epochs without a preempt, so
    # an interrupted run is resumed rather than skipped.
    if not interrupted:
        (Path(args.output_dir) / "complete").write_text(
            f"{flag} {args.model} {args.activation} s{args.seed} "
            f"auc={test_auc:.4f} acc={test_acc:.4f}\n"
        )
    if wandb_run is not None:
        import wandb
        wandb.run.summary.update(result)
        wandb.finish()


if __name__ == "__main__":
    main()
