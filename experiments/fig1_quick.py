"""Quick 3D loss landscape from trained checkpoints — with tqdm.

3×3: rows = ReLU/GELU/NELU, cols = weight scale ×0.5/×1.0/×2.0.
Uses checkpoints from main_cifar_tinyimagenet.py training.

Usage:
    python experiments/fig1_quick.py
    python experiments/fig1_quick.py --dataset cifar100 --steps 21
"""

import argparse, sys
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nelu import NELU
from experiments.main_cifar_tinyimagenet import build_model

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "savefig.dpi": 300,
})

CIFAR10_M, CIFAR10_S = (0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)
CIFAR100_M, CIFAR100_S = (0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761)


def get_w(m): return [p.data.clone() for p in m.parameters()]
def set_w(m, w):
    for p, wi in zip(m.parameters(), w): p.data.copy_(wi)

def filt_norm(d, w):
    for di, wi in zip(d, w):
        if di.dim() <= 1: di.fill_(0)
        else:
            for i in range(di.shape[0]):
                di[i].mul_(wi[i].norm() / (di[i].norm() + 1e-10))
    return d

@torch.no_grad()
def eval_loss(model, loader, dev):
    model.eval()
    tot, n = 0., 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        tot += F.cross_entropy(model(x), y).item() * x.size(0)
        n += x.size(0)
    return tot / n

def landscape(model, loader, dev, steps=25, ext=1.0):
    w0 = get_w(model)
    torch.manual_seed(777)
    d1 = filt_norm([torch.randn_like(w) for w in w0], w0)
    d2 = filt_norm([torch.randn_like(w) for w in w0], w0)
    cs = np.linspace(-ext, ext, steps)
    Z = np.zeros((steps, steps))
    for i in tqdm(range(steps), desc="    grid", leave=False, ncols=80):
        a = cs[i]
        for j, b in enumerate(cs):
            set_w(model, [w + a*dd1 + b*dd2 for w, dd1, dd2 in zip(w0, d1, d2)])
            Z[j, i] = eval_loss(model, loader, dev)
    set_w(model, w0)
    return cs, Z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--ext", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = args.dataset
    nc = 10 if ds == "cifar10" else 100
    M, S = (CIFAR10_M, CIFAR10_S) if ds == "cifar10" else (CIFAR100_M, CIFAR100_S)
    ds_cls = datasets.CIFAR10 if ds == "cifar10" else datasets.CIFAR100

    te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(M, S)])
    eval_ld = DataLoader(ds_cls("./data", False, download=True, transform=te),
                         256, False, num_workers=2)

    acts = [("ReLU", "relu"), ("GELU", "gelu"), ("NELU", "nelu")]
    scales = [0.5, 1.0, 2.0]

    grids = {}
    for label, act in acts:
        ckpt_path = ckpt_dir / f"resnet20_{ds}_{act}_s42_best.pt"
        if not ckpt_path.exists():
            print(f"  SKIP {label}: {ckpt_path} not found")
            continue

        print(f"Loading {label}...")
        model = build_model("resnet20", nc, 32, act).to(dev)
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        w_orig = get_w(model)

        for scale in scales:
            print(f"  {label} × {scale}")
            set_w(model, w_orig)
            for p in model.parameters():
                if p.dim() >= 2:
                    p.data.mul_(scale)
            _, Z = landscape(model, eval_ld, dev, steps=args.steps, ext=args.ext)
            grids[(label, scale)] = Z
            set_w(model, w_orig)

    # Plot
    cs = np.linspace(-args.ext, args.ext, args.steps)
    X, Y = np.meshgrid(cs, cs)
    available_acts = [a for a in acts if (a[0], 1.0) in grids]

    fig = plt.figure(figsize=(8, 8))
    for row, (label, _) in enumerate(available_acts):
        for col, scale in enumerate(scales):
            ax = fig.add_subplot(len(available_acts), 3, row*3+col+1, projection='3d')
            Z = grids[(label, scale)]
            Zp = np.log(Z - Z.min() + 0.01)
            ax.plot_surface(X, Y, Zp, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True, alpha=0.9)
            ax.view_init(elev=30, azim=-60)
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('none'); ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')
            ax.grid(False)
            if row == 0:
                ax.set_title(f"$||W|| \\times {scale}$", fontweight="bold", pad=-2, fontsize=10)
            if col == 0:
                ax.text2D(-0.12, 0.5, label, transform=ax.transAxes,
                          fontsize=11, fontweight="bold", rotation=90, va="center")

    fig.subplots_adjust(wspace=0, hspace=0, left=0.06, right=0.98, bottom=0.02, top=0.95)
    fig.savefig(out_dir / "fig1_teaser.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig1_teaser.pdf", bbox_inches="tight", facecolor="white")
    print(f"\nSaved {out_dir}/fig1_teaser.{{png,pdf}}")
    plt.close()


if __name__ == "__main__":
    main()
