#!/usr/bin/env python3
"""CIFAR-10/100 training — ReLU vs GELU vs NELU.

Architectures
    CNN:  ResNet-20/56/110, WRN-28-10, DenseNet-100-12,
          MobileNetV2, ShuffleNetV1
    ViT:  ViT-Tiny/4

Training recipe (follows Swish / Mish papers):
    CNN  — SGD, lr=0.1, momentum=0.9, wd=5e-4,
           MultiStepLR [60,120,160] gamma=0.2, 1-epoch warmup,
           200 epochs, batch 128  (pytorch-cifar100 standard)
    ViT  — AdamW, lr=1e-3, wd=0.05, cosine + 5-epoch warmup,
           200 epochs, batch 128

Usage:
    python main_cifar_tinyimagenet.py --arch resnet20 --dataset cifar100 --act nelu
    python main_cifar_tinyimagenet.py --arch resnet20 --dataset cifar100 --act nelu --label-noise 0.2
    python main_cifar_tinyimagenet.py --all --wandb --amp --compile
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nelu import NELU

# ── Constants ────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

ARCHS = [
    "resnet20", "resnet56", "resnet110",
    "wrn28_10", "densenet100",
    "mobilenetv2", "shufflenetv1",
    "vit_tiny", "vit_small", "vit_base",
]
DATASETS = ["cifar10", "cifar100"]
ACTS = ["relu", "gelu", "nelu"]

CNN_ARCHS = {"resnet20", "resnet56", "resnet110", "wrn28_10",
             "densenet100", "mobilenetv2", "shufflenetv1"}
VIT_ARCHS = {"vit_tiny", "vit_small", "vit_base"}

CIFAR10_MEAN  = (0.4914, 0.4822, 0.4465)
CIFAR10_STD   = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


# ── Seed ─────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Activation replacement ───────────────────────────────────────────

_REPLACE_TYPES = (nn.ReLU, nn.ReLU6, nn.GELU)
_ACT_MAP = {"relu": nn.ReLU, "gelu": nn.GELU, "nelu": NELU}


def replace_activations(model: nn.Module, act_name: str) -> nn.Module:
    """Recursively swap all ReLU / ReLU6 / GELU modules."""
    target_cls = _ACT_MAP[act_name]
    for name, child in model.named_children():
        if isinstance(child, _REPLACE_TYPES):
            setattr(model, name, target_cls())
        else:
            replace_activations(child, act_name)
    if isinstance(model, nn.Sequential):
        for i, child in enumerate(model):
            if isinstance(child, _REPLACE_TYPES):
                model[i] = target_cls()
    return model


# ── Datasets ─────────────────────────────────────────────────────────────────
def get_cifar10(data_dir: Path):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_ds = datasets.CIFAR10(root=str(data_dir), train=True, download=True,
                                transform=train_transform)
    test_ds = datasets.CIFAR10(root=str(data_dir), train=False, download=True,
                               transform=test_transform)
    return train_ds, test_ds, 10


def get_cifar100(data_dir: Path):
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
    train_ds = datasets.CIFAR100(root=str(data_dir), train=True, download=True,
                                 transform=train_transform)
    test_ds = datasets.CIFAR100(root=str(data_dir), train=False, download=True,
                                transform=test_transform)
    return train_ds, test_ds, 100


def get_dataset(name: str, data_dir: Path):
    if name == "cifar10":
        return get_cifar10(data_dir)
    elif name == "cifar100":
        return get_cifar100(data_dir)
    raise ValueError(f"Unknown dataset: {name}")


# ── Label noise ──────────────────────────────────────────────────────────────
def apply_label_noise(dataset, noise_rate: float, num_classes: int, seed: int = 42):
    """Corrupt a fraction of training labels uniformly at random."""
    import numpy as np
    rng = np.random.RandomState(seed)

    if hasattr(dataset, "targets"):
        # CIFAR-style
        targets = list(dataset.targets)
        n = len(targets)
        n_noisy = int(noise_rate * n)
        noisy_idx = rng.choice(n, n_noisy, replace=False)
        for i in noisy_idx:
            orig = targets[i]
            new_label = rng.choice(num_classes)
            while new_label == orig:
                new_label = rng.choice(num_classes)
            targets[i] = new_label
        dataset.targets = targets
    elif hasattr(dataset, "data"):
        # HF wrapper — modify underlying data
        data = dataset.data
        n = len(data)
        n_noisy = int(noise_rate * n)
        noisy_idx = rng.choice(n, n_noisy, replace=False)
        noisy_set = set(noisy_idx.tolist())

        class NoisyWrapper:
            """Wraps an HF split and overrides labels for noisy indices."""
            def __init__(self, original, label_map):
                self._original = original
                self._label_map = label_map

            def __len__(self):
                return len(self._original)

            def __getitem__(self, idx):
                item = self._original[idx]
                if idx in self._label_map:
                    # Return a copy with noisy label
                    return {**item, "label": self._label_map[idx]}
                return item

        label_map = {}
        for i in noisy_idx:
            orig = data[i]["label"]
            new_label = rng.choice(num_classes)
            while new_label == orig:
                new_label = rng.choice(num_classes)
            label_map[int(i)] = int(new_label)
        dataset.data = NoisyWrapper(data, label_map)

    return dataset


# ── CIFAR-style ResNet (He et al. 2016) ──────────────────────────────
# ResNet-{6n+2}: 3 stages, feature maps 32->16->8, BasicBlock

class _BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.ReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.act2(out)


class _CIFARResNet(nn.Module):
    def __init__(self, n_blocks, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU()
        self.layer1 = self._make_layer(16, 16, n_blocks, stride=1)
        self.layer2 = self._make_layer(16, 32, n_blocks, stride=2)
        self.layer3 = self._make_layer(32, 64, n_blocks, stride=2)
        self.fc = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_layer(in_ch, out_ch, n_blocks, stride):
        layers = [_BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(_BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = nn.functional.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ── Model builders ──────────────────────────────────────────────────

def build_resnet20(num_classes, img_size):
    return _CIFARResNet(n_blocks=3, num_classes=num_classes)

def build_resnet56(num_classes, img_size):
    return _CIFARResNet(n_blocks=9, num_classes=num_classes)

def build_resnet110(num_classes, img_size):
    return _CIFARResNet(n_blocks=18, num_classes=num_classes)

def build_wrn28_10(num_classes, img_size):
    """WRN-28-10 from pytorch-cifar100 repo (pre-activation, with dropout)."""

    class _WRNBlock(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.residual = nn.Sequential(
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            )
            self.shortcut = nn.Sequential()
            if in_ch != out_ch or stride != 1:
                self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride)

        def forward(self, x):
            return self.residual(x) + self.shortcut(x)

    class _WRN(nn.Module):
        def __init__(self, depth, widen, num_classes):
            super().__init__()
            k = widen
            n = (depth - 4) // 6
            self.in_ch = 16
            self.init_conv = nn.Conv2d(3, 16, 3, 1, padding=1)
            self.conv2 = self._stack(_WRNBlock, 16 * k, n, 1)
            self.conv3 = self._stack(_WRNBlock, 32 * k, n, 2)
            self.conv4 = self._stack(_WRNBlock, 64 * k, n, 2)
            self.bn = nn.BatchNorm2d(64 * k)
            self.relu = nn.ReLU(inplace=True)
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(64 * k, num_classes)

        def _stack(self, block, out_ch, n, stride):
            strides = [stride] + [1] * (n - 1)
            layers = []
            for s in strides:
                layers.append(block(self.in_ch, out_ch, s))
                self.in_ch = out_ch
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.init_conv(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.relu(self.bn(x))
            x = self.avg_pool(x)
            return self.linear(x.view(x.size(0), -1))

    return _WRN(28, 10, num_classes)

def build_densenet100(num_classes, img_size):
    """DenseNet-BC (L=100, k=12) — standard CIFAR benchmark model."""
    from torchvision.models import DenseNet
    # DenseNet-BC-100-12: 3 dense blocks, each (100-4)/6=16 layers, growth_rate=12
    model = DenseNet(
        growth_rate=12,
        block_config=(16, 16, 16),
        num_init_features=24,  # 2 * growth_rate
        bn_size=4,
        num_classes=num_classes,
    )
    # Replace first conv: 7x7 -> 3x3 for CIFAR
    model.features.conv0 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove maxpool
    model.features.pool0 = nn.Identity()
    return model


def build_mobilenetv2(num_classes, img_size):
    """MobileNetV2 for CIFAR (from pytorch-cifar100 repo)."""

    class _LinearBottleNeck(nn.Module):
        def __init__(self, in_ch, out_ch, stride, t=6):
            super().__init__()
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, in_ch * t, 1),
                nn.BatchNorm2d(in_ch * t),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_ch * t, in_ch * t, 3, stride=stride,
                          padding=1, groups=in_ch * t),
                nn.BatchNorm2d(in_ch * t),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_ch * t, out_ch, 1),
                nn.BatchNorm2d(out_ch),
            )
            self.stride = stride
            self.in_ch = in_ch
            self.out_ch = out_ch

        def forward(self, x):
            r = self.residual(x)
            if self.stride == 1 and self.in_ch == self.out_ch:
                r += x
            return r

    class _MobileNetV2(nn.Module):
        def __init__(self, num_classes=100):
            super().__init__()
            self.pre = nn.Sequential(
                nn.Conv2d(3, 32, 1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True),
            )
            self.stage1 = _LinearBottleNeck(32, 16, 1, 1)
            self.stage2 = self._make(2, 16, 24, 2, 6)
            self.stage3 = self._make(3, 24, 32, 2, 6)
            self.stage4 = self._make(4, 32, 64, 2, 6)
            self.stage5 = self._make(3, 64, 96, 1, 6)
            self.stage6 = self._make(3, 96, 160, 1, 6)
            self.stage7 = _LinearBottleNeck(160, 320, 1, 6)
            self.conv1 = nn.Sequential(
                nn.Conv2d(320, 1280, 1),
                nn.BatchNorm2d(1280),
                nn.ReLU6(inplace=True),
            )
            self.conv2 = nn.Conv2d(1280, num_classes, 1)

        def _make(self, n, in_ch, out_ch, stride, t):
            layers = [_LinearBottleNeck(in_ch, out_ch, stride, t)]
            for _ in range(1, n):
                layers.append(_LinearBottleNeck(out_ch, out_ch, 1, t))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.pre(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.stage5(x)
            x = self.stage6(x)
            x = self.stage7(x)
            x = self.conv1(x)
            x = nn.functional.adaptive_avg_pool2d(x, 1)
            x = self.conv2(x)
            return x.view(x.size(0), -1)

    return _MobileNetV2(num_classes)


def build_shufflenetv1(num_classes, img_size):
    """ShuffleNetV2 for CIFAR (from pytorch-cifar100 repo)."""

    def _channel_split(x, split):
        return torch.split(x, split, dim=1)

    def _channel_shuffle(x, groups):
        b, c, h, w = x.size()
        cpg = c // groups
        x = x.view(b, groups, cpg, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, -1, h, w)

    class _ShuffleUnit(nn.Module):
        def __init__(self, in_ch, out_ch, stride):
            super().__init__()
            self.stride = stride
            self.in_ch = in_ch
            self.out_ch = out_ch
            if stride != 1 or in_ch != out_ch:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, 1), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
                    nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch),
                    nn.BatchNorm2d(in_ch),
                    nn.Conv2d(in_ch, out_ch // 2, 1), nn.BatchNorm2d(out_ch // 2), nn.ReLU(inplace=True),
                )
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch),
                    nn.BatchNorm2d(in_ch),
                    nn.Conv2d(in_ch, out_ch // 2, 1), nn.BatchNorm2d(out_ch // 2), nn.ReLU(inplace=True),
                )
            else:
                half = in_ch // 2
                self.residual = nn.Sequential(
                    nn.Conv2d(half, half, 1), nn.BatchNorm2d(half), nn.ReLU(inplace=True),
                    nn.Conv2d(half, half, 3, stride=stride, padding=1, groups=half),
                    nn.BatchNorm2d(half),
                    nn.Conv2d(half, half, 1), nn.BatchNorm2d(half), nn.ReLU(inplace=True),
                )
                self.shortcut = nn.Sequential()

        def forward(self, x):
            if self.stride == 1 and self.out_ch == self.in_ch:
                s, r = _channel_split(x, self.in_ch // 2)
            else:
                s, r = x, x
            s = self.shortcut(s)
            r = self.residual(r)
            x = torch.cat([s, r], dim=1)
            return _channel_shuffle(x, 2)

    class _ShuffleNetV2(nn.Module):
        def __init__(self, num_classes=100):
            super().__init__()
            out_ch = [116, 232, 464, 1024]
            self.pre = nn.Sequential(nn.Conv2d(3, 24, 3, padding=1), nn.BatchNorm2d(24))
            self.stage2 = self._make(24, out_ch[0], 3)
            self.stage3 = self._make(out_ch[0], out_ch[1], 7)
            self.stage4 = self._make(out_ch[1], out_ch[2], 3)
            self.conv5 = nn.Sequential(
                nn.Conv2d(out_ch[2], out_ch[3], 1), nn.BatchNorm2d(out_ch[3]), nn.ReLU(inplace=True),
            )
            self.fc = nn.Linear(out_ch[3], num_classes)

        def _make(self, in_ch, out_ch, repeat):
            layers = [_ShuffleUnit(in_ch, out_ch, 2)]
            for _ in range(repeat):
                layers.append(_ShuffleUnit(out_ch, out_ch, 1))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.pre(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.conv5(x)
            x = nn.functional.adaptive_avg_pool2d(x, 1)
            return self.fc(x.view(x.size(0), -1))

    return _ShuffleNetV2(num_classes)


def build_vit_tiny(num_classes, img_size):
    import timm
    ps = 4 if img_size <= 32 else 16
    return timm.create_model("vit_tiny_patch16_224", pretrained=False,
                             num_classes=num_classes, img_size=img_size,
                             patch_size=ps)

def build_vit_small(num_classes, img_size):
    import timm
    ps = 4 if img_size <= 32 else 16
    return timm.create_model("vit_small_patch16_224", pretrained=False,
                             num_classes=num_classes, img_size=img_size,
                             patch_size=ps)

def build_vit_base(num_classes, img_size):
    import timm
    ps = 4 if img_size <= 32 else 16
    return timm.create_model("vit_base_patch16_224", pretrained=False,
                             num_classes=num_classes, img_size=img_size,
                             patch_size=ps)


def build_model(arch, num_classes, img_size, act_name):
    builders = {
        "resnet20": build_resnet20,
        "resnet56": build_resnet56,
        "resnet110": build_resnet110,
        "wrn28_10": build_wrn28_10,
        "densenet100": build_densenet100,
        "mobilenetv2": build_mobilenetv2,
        "shufflenetv1": build_shufflenetv1,
        "vit_tiny": build_vit_tiny,
        "vit_small": build_vit_small,
        "vit_base": build_vit_base,
    }
    model = builders[arch](num_classes, img_size)

    # Replace activations
    # CNNs default to ReLU; ViTs default to GELU — always replace to requested
    model = replace_activations(model, act_name)
    return model


# ── Training / Evaluation ────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  train", leave=False, ncols=100)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100.*correct/total:.1f}%")

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


# ── Main training loop ───────────────────────────────────────────────────────
def run_experiment(arch: str, dataset_name: str, act_name: str, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # Dataset config
    img_size = 32
    epochs = args.epochs if args.epochs is not None else 200

    print(f"\n{'='*70}")
    print(f"  Experiment: arch={arch}  dataset={dataset_name}  act={act_name}")
    print(f"  epochs={epochs}  seed={args.seed}  device={device}")
    if args.label_noise > 0:
        print(f"  label_noise={args.label_noise}")
    print(f"{'='*70}")

    # Load data
    train_ds, test_ds, num_classes = get_dataset(dataset_name, DATA_DIR)

    # Apply label noise if requested
    if args.label_noise > 0:
        train_ds = apply_label_noise(train_ds, args.label_noise, num_classes, seed=args.seed)

    num_workers = 2  # keep low to avoid OOM on 16GB RAM
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # Build model
    model = build_model(arch, num_classes, img_size, act_name)
    model = model.to(device)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Optimizer & scheduler
    is_vit = arch in VIT_ARCHS
    if is_vit:
        lr = args.lr if args.lr is not None else 1e-3
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    else:
        lr = args.lr if args.lr is not None else 0.1
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    if is_vit:
        # Cosine with 5-epoch linear warmup (standard ViT recipe)
        warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3,
                                             total_iters=5)
        cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 5)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[5])
    else:
        # Standard pytorch-cifar100 recipe: milestones [60,120,160], gamma 0.2
        # with 1-epoch warmup (important for MobileNet/ShuffleNet)
        base_sched = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[60, 120, 160], gamma=0.2)
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=1)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, base_sched], milestones=[1])
    criterion = nn.CrossEntropyLoss()

    use_amp = args.amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Wandb
    wandb_run = None
    if args.wandb:
        import wandb
        config = {
            "arch": arch, "dataset": dataset_name, "act": act_name,
            "epochs": epochs, "seed": args.seed, "label_noise": args.label_noise,
            "lr": 1e-3 if is_vit else 0.1,
            "weight_decay": 0.05 if is_vit else 5e-4,
            "optimizer": "AdamW" if is_vit else "SGD",
            "params": param_count, "amp": use_amp, "compile": args.compile,
        }
        noise_tag = f"_noise{args.label_noise}" if args.label_noise > 0 else ""
        wandb_run = wandb.init(
            project="nelu",
            group=f"{dataset_name}_{arch}",
            name=f"{arch}_{dataset_name}_{act_name}{noise_tag}_s{args.seed}",
            config=config,
            reinit=True,
        )

    # ── Diagnostic: gate statistics ──────────────────────────────────
    @torch.no_grad()
    def compute_diagnostics(model, loader, device, use_amp, max_batches=5):
        """Compute gate entropy, binary fraction, weight norm on a few batches."""
        import math as _math

        # Use uncompiled model for hooks (torch.compile breaks hooks)
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw.eval()

        pre_acts = []
        hooks = []
        for m in raw.modules():
            if isinstance(m, (nn.GELU, NELU)):
                def _hook(module, inp, out, storage=pre_acts):
                    storage.append(inp[0].detach().float())
                hooks.append(m.register_forward_hook(_hook))

        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    raw(images)

        for h in hooks:
            h.remove()
        raw.train()

        if not pre_acts:
            return {}

        # Compute gate values
        all_gates = []
        for z in pre_acts:
            if act_name == "nelu":
                rms = z.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
                g = 0.5 * (1.0 + torch.erf(z / (rms * _math.sqrt(2))))
            else:
                g = 0.5 * (1.0 + torch.erf(z / _math.sqrt(2)))
            all_gates.append(g.cpu())
        del pre_acts

        gates = torch.cat([g.reshape(-1) for g in all_gates])
        g_clamped = gates.clamp(1e-7, 1 - 1e-7)
        entropy = -(g_clamped * g_clamped.log() + (1 - g_clamped) * (1 - g_clamped).log()).mean().item()
        binary_frac = ((gates < 0.05) | (gates > 0.95)).float().mean().item()
        gate_mean = gates.mean().item()
        gate_std = gates.std().item()
        del all_gates, gates

        # Weight norm
        w_norm = sum(p.pow(2).sum() for p in model.parameters() if p.dim() >= 2).sqrt().item()

        model.train()
        return {
            "gate_entropy": entropy,
            "binary_frac": binary_frac,
            "gate_mean": gate_mean,
            "gate_std": gate_std,
            "weight_norm": w_norm,
        }

    # ── Checkpoint directory ──────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    noise_tag = f"_noise{args.label_noise}" if args.label_noise > 0 else ""
    lr_tag = f"_lr{args.lr}" if args.lr is not None else ""
    run_tag = f"{arch}_{dataset_name}_{act_name}{noise_tag}{lr_tag}_s{args.seed}"
    ckpt_dir = RESULTS_DIR / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_name = run_tag
    last_path = ckpt_dir / f"{ckpt_name}_last.pt"

    # Train
    best_acc = 0.0
    start_epoch = 1
    history = {
        "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "lr": [],
        "gate_entropy": [], "binary_frac": [], "weight_norm": [], "gen_gap": [],
    }

    # ── Resume (auto-detect last.pt unless --no-resume) ───────────────
    if not args.no_resume and last_path.exists():
        print(f"  → resuming from {last_path}")
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if use_amp and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        history = ckpt.get("history", history)
        # RNG restore (move tensors to CPU — set_rng_state requires ByteTensor on CPU)
        if "rng" in ckpt:
            torch.set_rng_state(ckpt["rng"]["torch"].cpu())
            if torch.cuda.is_available() and "cuda" in ckpt["rng"]:
                torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng"]["cuda"]])
            np.random.set_state(ckpt["rng"]["numpy"])
            import random as _r
            _r.setstate(ckpt["rng"]["python"])
        print(f"  → resumed: start_epoch={start_epoch}  best_acc={best_acc:.2f}%")

    t0 = time.time()
    for epoch in range(start_epoch, epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\n  Epoch {epoch}/{epochs}  lr={lr_now:.6f}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                optimizer, scaler, device, use_amp)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, use_amp)
        scheduler.step()

        gen_gap = train_acc - test_acc

        # Diagnostics every 10 epochs + first and last
        diag = {}
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            diag = compute_diagnostics(model, test_loader, device, use_amp)

        print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
              f"test_loss={test_loss:.4f}  test_acc={test_acc:.2f}%  gap={gen_gap:.2f}%")
        if diag:
            print(f"  gate_entropy={diag.get('gate_entropy',0):.4f}  "
                  f"binary={diag.get('binary_frac',0):.1%}  "
                  f"||W||={diag.get('weight_norm',0):.1f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(lr_now)
        history["gen_gap"].append(gen_gap)
        history["gate_entropy"].append(diag.get("gate_entropy", None))
        history["binary_frac"].append(diag.get("binary_frac", None))
        history["weight_norm"].append(diag.get("weight_norm", None))

        # ── Save: full state every epoch (last.pt) + best (model only) ──
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        import random as _r
        rng_state = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": _r.getstate(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "best_acc": best_acc,
            "history": history,
            "rng": rng_state,
            "arch": arch, "dataset": dataset_name, "act": act_name,
        }
        # Atomic write: tmp then rename
        tmp_path = last_path.with_suffix(".pt.tmp")
        torch.save(ckpt_payload, tmp_path)
        tmp_path.replace(last_path)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "best_acc": best_acc,
                "arch": arch, "dataset": dataset_name, "act": act_name,
            }, ckpt_dir / f"{ckpt_name}_best.pt")

        if wandb_run:
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "test/loss": test_loss,
                "test/acc": test_acc,
                "lr": lr_now,
                "best_test_acc": best_acc,
                "gen_gap": gen_gap,
            }
            if diag:
                log_dict.update({
                    "gate/entropy": diag["gate_entropy"],
                    "gate/binary_frac": diag["binary_frac"],
                    "gate/mean": diag["gate_mean"],
                    "gate/std": diag["gate_std"],
                    "weight_norm": diag["weight_norm"],
                })
            wandb.log(log_dict)

    elapsed = time.time() - t0
    print(f"\n  Done. Best test acc: {best_acc:.2f}%  Time: {elapsed:.0f}s")

    # Save final checkpoint
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "epoch": epochs,
        "model_state_dict": raw_model.state_dict(),
        "best_acc": best_acc,
        "arch": arch, "dataset": dataset_name, "act": act_name,
    }, ckpt_dir / f"{ckpt_name}_final.pt")

    if wandb_run:
        wandb.log({"best_test_acc": best_acc, "total_time_s": elapsed})
        wandb_run.finish()

    # Save results JSON
    result_path = RESULTS_DIR / f"main_{run_tag}.json"
    result = {
        "arch": arch,
        "dataset": dataset_name,
        "act": act_name,
        "seed": args.seed,
        "label_noise": args.label_noise,
        "best_test_acc": best_acc,
        "final_test_acc": history["test_acc"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_gen_gap": history["gen_gap"][-1],
        "final_gate_entropy": history["gate_entropy"][-1],
        "final_binary_frac": history["binary_frac"][-1],
        "final_weight_norm": history["weight_norm"][-1],
        "epochs": epochs,
        "params": param_count,
        "time_s": elapsed,
        "history": history,
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to {result_path}")
    print(f"  Checkpoints in {ckpt_dir}/{ckpt_name}_*.pt")

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="NELU vs GELU: main experiments on CIFAR-10/100 and Tiny-ImageNet")
    parser.add_argument("--arch", type=str, default="resnet20",
                        choices=ARCHS,
                        help="Architecture to train")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=DATASETS,
                        help="Dataset to train on")
    parser.add_argument("--act", type=str, default="nelu", choices=ACTS,
                        help="Activation function: relu/gelu (baseline) or nelu (ours)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate (default: 0.1 CNN, 1e-3 ViT)")
    parser.add_argument("--label-noise", type=float, default=0.0,
                        help="Fraction of training labels to corrupt")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging (project=nelu)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile (requires PyTorch 2.0+)")
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision")
    parser.add_argument("--all", action="store_true",
                        help="Run ALL (arch, dataset, act) combinations")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing last.pt and start fresh")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override total epochs (for smoke tests)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all:
        results_summary = []
        for dataset_name in DATASETS:
            for arch in ARCHS:
                for act_name in ACTS:
                    result = run_experiment(arch, dataset_name, act_name, args)
                    results_summary.append({
                        "arch": arch, "dataset": dataset_name, "act": act_name,
                        "best_test_acc": result["best_test_acc"],
                    })

        # Print summary table
        print(f"\n{'='*70}")
        print("  SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Arch':<12} {'Dataset':<15} {'Act':<6} {'Best Acc':>8}")
        print(f"  {'-'*45}")
        for r in results_summary:
            print(f"  {r['arch']:<12} {r['dataset']:<15} {r['act']:<6} {r['best_test_acc']:>7.2f}%")

        # Save full summary
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = RESULTS_DIR / "main_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=2)
        print(f"\n  Summary saved to {summary_path}")
    else:
        run_experiment(args.arch, args.dataset, args.act, args)


if __name__ == "__main__":
    main()
