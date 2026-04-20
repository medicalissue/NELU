"""CIFAR-100 training with activation comparison.

Self-contained script for training CIFAR-scale models with NELU/NiLU
activation swaps. Supports standard CIFAR-100 benchmarking models and
logs gamma dynamics for gated activations.

Usage:
    python train/train_cifar.py --model resnet20 --activation nelu --seed 42
    python train/train_cifar.py --model wrn28_10 --activation gelu --seed 123
    python train/train_cifar.py --model densenet100 --activation nilu --epochs 300
"""

import argparse
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from nelu import NELU, NiLU, collect_gamma_stats
from train.act_swap import replace_activation


# ---------------------------------------------------------------------------
#  CIFAR ResNet (He et al., 2015 -- proper CIFAR variant)
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class CIFARResNet(nn.Module):
    """ResNet for CIFAR (32x32). Depth must satisfy (depth-2) % 6 == 0."""
    def __init__(self, depth, num_classes=100, widen_factor=1):
        super().__init__()
        assert (depth - 2) % 6 == 0, f"Depth must be 6n+2, got {depth}"
        n = (depth - 2) // 6
        channels = [16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, channels[0], n, stride=1)
        self.layer2 = self._make_layer(channels[0], channels[1], n, stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[2], num_classes)

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, planes, num_blocks, stride):
        layers = [BasicBlock(in_planes, planes, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ---------------------------------------------------------------------------
#  Wide ResNet (Zagoruyko & Komodakis, 2016)
# ---------------------------------------------------------------------------

class WideBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.act(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.dropout(self.act(self.bn2(out)))
        out = self.conv2(out)
        return out + shortcut


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, num_classes=100):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen_factor
        channels = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, channels[0], 3, padding=1, bias=False)
        self.layer1 = self._make_layer(channels[0], channels[1], n, dropout_rate, stride=1)
        self.layer2 = self._make_layer(channels[1], channels[2], n, dropout_rate, stride=2)
        self.layer3 = self._make_layer(channels[2], channels[3], n, dropout_rate, stride=2)
        self.bn = nn.BatchNorm2d(channels[3])
        self.act = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[3], num_classes)

    def _make_layer(self, in_planes, planes, num_blocks, dropout_rate, stride):
        layers = [WideBlock(in_planes, planes, dropout_rate, stride)]
        for _ in range(1, num_blocks):
            layers.append(WideBlock(planes, planes, dropout_rate, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.act(self.bn(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ---------------------------------------------------------------------------
#  DenseNet-BC (Huang et al., 2017) -- growth rate 12
# ---------------------------------------------------------------------------

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super().__init__()
        inter = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, inter, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter)
        self.conv2 = nn.Conv2d(inter, growth_rate, 3, padding=1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.act(self.bn1(x)))
        out = self.conv2(self.act(self.bn2(out)))
        return torch.cat([x, out], 1)


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, 1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return F.avg_pool2d(self.conv(self.act(self.bn(x))), 2)


class DenseNet(nn.Module):
    def __init__(self, depth=100, growth_rate=12, reduction=0.5, num_classes=100):
        super().__init__()
        n_blocks = (depth - 4) // 6  # BC variant: 2 convs per block
        n_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, n_channels, 3, padding=1, bias=False)
        self.dense1, n_channels = self._make_dense(n_channels, growth_rate, n_blocks)
        out_ch = int(math.floor(n_channels * reduction))
        self.trans1 = Transition(n_channels, out_ch); n_channels = out_ch
        self.dense2, n_channels = self._make_dense(n_channels, growth_rate, n_blocks)
        out_ch = int(math.floor(n_channels * reduction))
        self.trans2 = Transition(n_channels, out_ch); n_channels = out_ch
        self.dense3, n_channels = self._make_dense(n_channels, growth_rate, n_blocks)
        self.bn = nn.BatchNorm2d(n_channels)
        self.act = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_channels, num_classes)

    def _make_dense(self, in_channels, growth_rate, n_blocks):
        layers = []
        ch = in_channels
        for _ in range(n_blocks):
            layers.append(Bottleneck(ch, growth_rate))
            ch += growth_rate
        return nn.Sequential(*layers), ch

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = self.act(self.bn(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ---------------------------------------------------------------------------
#  MobileNetV2 adapted for CIFAR (32x32)
# ---------------------------------------------------------------------------

def _make_mobilenetv2_cifar(num_classes=100):
    """MobileNetV2 with stride=1 in first conv for 32x32 inputs."""
    from torchvision.models import mobilenet_v2
    model = mobilenet_v2(num_classes=num_classes)
    # Original first conv: stride=2, 3->32, for 224x224
    # For 32x32: use stride=1 so feature maps stay large enough
    model.features[0][0] = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
    return model


# ---------------------------------------------------------------------------
#  ShuffleNet V1 (minimal CIFAR variant)
# ---------------------------------------------------------------------------

def _make_shufflenetv1_cifar(num_classes=100):
    """Minimal ShuffleNet V1 for CIFAR-100."""
    from torchvision.models import shufflenet_v2_x1_0
    model = shufflenet_v2_x1_0(num_classes=num_classes)
    # Adapt first conv for 32x32
    model.conv1[0] = nn.Conv2d(3, 24, 3, stride=1, padding=1, bias=False)
    # Remove the initial maxpool (too aggressive for 32x32)
    model.maxpool = nn.Identity()
    return model


# ---------------------------------------------------------------------------
#  Model factory
# ---------------------------------------------------------------------------

_ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "nelu": NELU,
    "nilu": NiLU,
}

# Map source -> target for activation swap
_SWAP_MAP = {
    "nelu": (nn.ReLU, NELU),
    "nilu": (nn.ReLU, NiLU),
    "gelu": (nn.ReLU, nn.GELU),
    "silu": (nn.ReLU, nn.SiLU),
}


def build_model(name, activation="relu", num_classes=100):
    """Create a model and apply activation swap if needed."""
    if name == "resnet20":
        model = CIFARResNet(20, num_classes=num_classes)
    elif name == "resnet56":
        model = CIFARResNet(56, num_classes=num_classes)
    elif name == "resnet110":
        model = CIFARResNet(110, num_classes=num_classes)
    elif name == "wrn28_10":
        model = WideResNet(28, 10, num_classes=num_classes)
    elif name == "densenet100":
        model = DenseNet(100, 12, num_classes=num_classes)
    elif name == "mobilenetv2":
        model = _make_mobilenetv2_cifar(num_classes)
    elif name == "shufflenetv1":
        model = _make_shufflenetv1_cifar(num_classes)
    else:
        raise ValueError(f"Unknown model: {name}. Supported: resnet20, resnet56, "
                         f"resnet110, wrn28_10, densenet100, mobilenetv2, shufflenetv1")

    # Apply activation swap (all models are built with ReLU by default)
    if activation != "relu" and activation in _SWAP_MAP:
        src_cls, tgt_cls = _SWAP_MAP[activation]
        n = replace_activation(model, src_cls, tgt_cls, rms_mode="last_3dims")
        print(f"Swapped {n} {src_cls.__name__} -> {tgt_cls.__name__}")

    return model


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_dataloaders(data_dir, train_batch_size, val_batch_size=None, num_workers=4):
    if val_batch_size is None:
        val_batch_size = train_batch_size * 2

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train_ds = datasets.CIFAR100(data_dir, train=True, download=True,
                                  transform=train_transform)
    test_ds = datasets.CIFAR100(data_dir, train=False, download=True,
                                 transform=test_transform)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=val_batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

class TrainingInterrupted(Exception):
    """Raised when SIGTERM/SIGINT requests a graceful stop."""
    pass


_INTERRUPTION_REQUESTED = False


def _request_interruption(signum, _frame):
    global _INTERRUPTION_REQUESTED
    _INTERRUPTION_REQUESTED = True
    print(f"\n[signal] Received signal {signum}; will stop after the current step.", flush=True)


def interruption_requested():
    return _INTERRUPTION_REQUESTED


def save_checkpoint(state, output_dir, interrupted=False):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "checkpoint.pt"
    torch.save(state, checkpoint_path)
    if interrupted:
        (output_path / "INTERRUPTED").touch()


def build_checkpoint_state(model, optimizer, scheduler, best_acc, args, wandb_id,
                           training_log, epoch):
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_acc": best_acc,
        "args": vars(args),
        "training_log": training_log,
        "wandb_id": wandb_id,
    }

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch,
                    scaler=None, use_amp=False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if interruption_requested():
            raise TrainingInterrupted(
                f"Interruption requested at epoch {epoch}, step {batch_idx}."
            )
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)
        if interruption_requested():
            raise TrainingInterrupted(
                f"Interruption requested at epoch {epoch}, step {batch_idx}."
            )
    scheduler.step()
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if interruption_requested():
            raise TrainingInterrupted(
                f"Interruption requested during evaluation at step {batch_idx}."
            )
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


def init_wandb_run(args, saved_wandb_id=None):
    if not args.wandb:
        return None, saved_wandb_id

    if args.resume and os.path.isfile(args.resume) and not saved_wandb_id:
        raise RuntimeError(
            f"Resume checkpoint {args.resume} is missing wandb_id; refusing to create a new wandb run."
        )

    try:
        import wandb
    except ImportError:
        print("WARNING: wandb not installed, disabling wandb logging")
        return None, saved_wandb_id

    init_kwargs = {
        "project": args.wandb_project,
        "name": f"{args.model}_{args.activation}_s{args.seed}",
        "config": vars(args),
    }
    if saved_wandb_id:
        init_kwargs["id"] = saved_wandb_id
        init_kwargs["resume"] = "must"
    else:
        init_kwargs["id"] = wandb.util.generate_id()
        init_kwargs["resume"] = "never"

    wandb_run = wandb.init(**init_kwargs)
    if saved_wandb_id and wandb_run.id != saved_wandb_id:
        raise RuntimeError(
            f"wandb resumed unexpected run id: expected {saved_wandb_id}, got {wandb_run.id}"
        )

    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    return wandb_run, wandb_run.id


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-100 training with activation comparison")
    p.add_argument("--model", type=str, default="resnet20",
                    choices=["resnet20", "resnet56", "resnet110", "wrn28_10",
                             "densenet100", "mobilenetv2", "shufflenetv1"])
    p.add_argument("--activation", type=str, default="relu",
                    choices=["relu", "gelu", "silu", "nelu", "nilu"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--val_batch_size", type=int, default=None,
                   help="Validation batch size; defaults to 2x train batch size")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="results/cifar")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--data_dir", type=str, default="/data")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="Use AMP (mixed precision)")
    p.add_argument("--compile", action="store_true", help="Use torch.compile")
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--wandb_project", type=str, default="nelu-cifar")
    p.add_argument("--config", type=str, default=None, help="YAML config (overrides defaults)")
    return p.parse_args()


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, _request_interruption)
    signal.signal(signal.SIGINT, _request_interruption)

    # Load YAML config if provided (overrides CLI defaults)
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k) and k != "config":
                setattr(args, k, v)

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # wandb init (deferred until after resume, so we can reuse the saved run ID)
    wandb_run = None
    saved_wandb_id = None

    # Data
    train_loader, test_loader = get_dataloaders(
        args.data_dir,
        args.batch_size,
        args.val_batch_size,
        args.num_workers,
    )
    # Model
    model = build_model(args.model, args.activation, num_classes=100)
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, activation: {args.activation}, "
          f"params: {param_count:,}, device: {device}")

    # torch.compile
    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("Model compiled with torch.compile")

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if args.amp else None

    # Optimizer + scheduler (MultiStepLR with 1-epoch linear warmup)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    main_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 120, 160], gamma=0.2)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, total_iters=1)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[1])

    # Resume
    start_epoch = 0
    best_acc = 0.0
    training_log = []

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        training_log = ckpt.get("training_log", [])
        saved_wandb_id = ckpt.get("wandb_id", None)
        print(f"  Resumed at epoch {start_epoch}, best_acc={best_acc:.2f}%")

    # wandb init — after resume so we can reuse the saved run ID
    wandb_run, wandb_id = init_wandb_run(args, saved_wandb_id)

    # Probe batch for gate entropy (fixed across training)
    probe_batch = next(iter(test_loader))[0][:32].to(device)

    # Training
    interrupted = False
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        try:
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer,
                                                     scheduler, device, epoch,
                                                     scaler=scaler, use_amp=args.amp)
            test_loss, test_acc = evaluate(model, test_loader, device, use_amp=args.amp)
        except TrainingInterrupted as e:
            interrupted = True
            print(f"\n{'='*60}")
            print(f"  TRAINING INTERRUPTED: {e}")
            print("  Saving checkpoint for automatic spot resume.")
            print(f"{'='*60}")
            interrupted_state = build_checkpoint_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_acc=best_acc,
                args=args,
                wandb_id=wandb_id,
                training_log=training_log,
                epoch=epoch - 1,
            )
            interrupted_state["interrupted_at_epoch"] = epoch
            save_checkpoint(interrupted_state, args.output_dir, interrupted=True)
            if wandb_run:
                import wandb
                wandb.log({"epoch": epoch, "interrupted": True, "interrupted_epoch": epoch})
                wandb.finish()
            break
        elapsed = time.time() - t0

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc

        # Gamma stats for gated activations
        gamma_stats = {}
        entropy_stats = {}
        if args.activation in ("nelu", "nilu"):
            gamma_stats = collect_gamma_stats(model)
            from train.gamma_logging import measure_gate_entropy
            entropy_stats = measure_gate_entropy(model, probe_batch, device)

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "time": elapsed,
        }
        log_entry.update(gamma_stats)
        log_entry.update(entropy_stats)
        training_log.append(log_entry)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train {train_acc:.2f}% | test {test_acc:.2f}% | "
              f"best {best_acc:.2f}% | lr {optimizer.param_groups[0]['lr']:.6f} | "
              f"{elapsed:.1f}s", end="")
        if gamma_stats:
            print(f" | gamma_mean {gamma_stats.get('nelu/gamma/mean', 0):.4f}", end="")
        print()

        if wandb_run:
            import wandb
            wandb_metrics = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/top1": train_acc,
                "time/epoch_seconds": elapsed,
                "test/loss": test_loss,
                "test/top1": test_acc,
                "best/test_top1": best_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
            wandb_metrics.update(gamma_stats)
            wandb_metrics.update(entropy_stats)
            wandb.log(wandb_metrics)

        # Save checkpoints (includes wandb_id for run continuity across spot interruptions)
        state = build_checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_acc=best_acc,
            args=args,
            wandb_id=wandb_id,
            training_log=training_log,
            epoch=epoch,
        )
        save_checkpoint(state, args.output_dir)
        if is_best:
            torch.save(state, os.path.join(args.output_dir, "checkpoint-best.pt"))

    if interrupted:
        print("\nTraining interrupted cleanly. Resume checkpoint saved.")
        return

    # Save final result.json
    result = {
        "model": args.model,
        "activation": args.activation,
        "seed": args.seed,
        "final_test_acc": training_log[-1]["test_acc"] if training_log else 0,
        "best_test_acc": best_acc,
        "epochs": args.epochs,
        "params": param_count,
        "training_log": training_log,
    }
    result_path = os.path.join(args.output_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_path}")
    print(f"Best test accuracy: {best_acc:.2f}%")

    if wandb_run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
