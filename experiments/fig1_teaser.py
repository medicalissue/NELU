"""Figure 1: 3D loss landscape — ReLU vs GELU vs NELU.

Mish-paper style. Each model trained separately, same random directions,
raw loss (no log), 51x51 grid. No weight scaling.
"""

import sys
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nelu import NELU
from main_cifar_tinyimagenet import build_model

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "savefig.dpi": 300,
})


def get_w(m): return [p.data.clone() for p in m.parameters()]
def set_w(m, w):
    for p, wi in zip(m.parameters(), w): p.data.copy_(wi)

def filt_norm(d, w):
    for di, wi in zip(d, w):
        if di.dim() <= 1:
            di.fill_(0)
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

def landscape(model, loader, dev, steps=51, ext=0.5):
    w0 = get_w(model)
    torch.manual_seed(777)
    d1 = filt_norm([torch.randn_like(w) for w in w0], w0)
    d2 = filt_norm([torch.randn_like(w) for w in w0], w0)
    cs = np.linspace(-ext, ext, steps)
    Z = np.zeros((steps, steps))
    for i, a in enumerate(cs):
        for j, b in enumerate(cs):
            set_w(model, [w + a*dd1 + b*dd2 for w, dd1, dd2 in zip(w0, d1, d2)])
            Z[j, i] = eval_loss(model, loader, dev)
        if (i+1) % 10 == 0:
            print(f"    {i+1}/{steps}")
    set_w(model, w0)
    return cs, Z


def main():
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    M, S = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(M, S)])
    loader = DataLoader(datasets.CIFAR10("./data", False, download=True, transform=te),
                        256, False, num_workers=2)

    acts = [
        ("ReLU", "relu", "resnet20_cifar10_relu_s42_best.pt"),
        ("GELU", "gelu", "resnet20_cifar10_gelu_s42_best.pt"),
        ("NELU", "nelu", "resnet20_cifar10_nelu_s42_best.pt"),
    ]

    steps = 51
    ext = 0.5  # smooth range

    grids = {}
    for aname, act_key, ckpt_name in acts:
        print(f"{aname}...")
        model = build_model("resnet20", 10, 32, act_key).to(dev)
        ckpt = torch.load(ckpt_dir / ckpt_name, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        cs, Z = landscape(model, loader, dev, steps=steps, ext=ext)
        grids[aname] = Z

    # ── Save raw data ──
    save_path = out_dir / "fig1_landscape_data.npz"
    np.savez(save_path,
             coords=cs,
             relu=grids["ReLU"],
             gelu=grids["GELU"],
             nelu=grids["NELU"],
             steps=steps, ext=ext)
    print(f"Saved raw data: {save_path}")

    # ── Render ──
    render(cs, grids, out_dir)


def render(cs, grids, out_dir):
    """Render from saved or computed data. Edit this to tweak visuals."""
    X, Y = np.meshgrid(cs, cs)
    acts = ["ReLU", "GELU", "NELU"]

    # ── 3D surface ──
    fig = plt.figure(figsize=(9, 3.2))
    vmin = min(grids[a].min() for a in acts)
    vmax = min(grids[a].max() for a in acts)

    for idx, aname in enumerate(acts):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        Z = grids[aname]
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=True, alpha=0.9,
                        vmin=vmin, vmax=vmax)
        ax.view_init(elev=30, azim=-60)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none'); ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        ax.grid(False)
        ax.text2D(0.5, -0.02, f"({chr(97+idx)}) {aname}",
                  transform=ax.transAxes, fontsize=10, fontweight="bold", ha="center")

    fig.subplots_adjust(wspace=0.02, left=0.02, right=0.98, bottom=0.08, top=0.95)
    fig.savefig(out_dir / "fig1_teaser.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig1_teaser.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved {out_dir}/fig1_teaser.{{png,pdf}}")
    plt.close()

    # ── 2D contour ──
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
    vmin_c = min(grids[a].min() for a in acts)
    vmax_c = np.percentile(np.concatenate([grids[a].ravel() for a in acts]), 95)

    for idx, aname in enumerate(acts):
        ax = axes[idx]
        Z = grids[aname]
        ax.contourf(X, Y, Z, levels=30, cmap="RdYlBu_r", vmin=vmin_c, vmax=vmax_c)
        ax.contour(X, Y, Z, levels=10, colors="white", linewidths=0.3, alpha=0.5)
        ax.plot(0, 0, "k*", markersize=5, zorder=5)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
        ax.text(0.5, -0.08, f"({chr(97+idx)}) {aname}",
                transform=ax.transAxes, fontsize=9, fontweight="bold", ha="center")

    fig.tight_layout(pad=0.3, w_pad=0.5)
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(out_dir / "fig1_teaser_contour.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig1_teaser_contour.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved {out_dir}/fig1_teaser_contour.{{png,pdf}}")
    plt.close()


def render_from_saved():
    """Re-render from saved .npz without recomputing."""
    out_dir = Path(__file__).resolve().parent.parent / "results"
    data = np.load(out_dir / "fig1_landscape_data.npz")
    cs = data["coords"]
    grids = {"ReLU": data["relu"], "GELU": data["gelu"], "NELU": data["nelu"]}
    render(cs, grids, out_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--render-only", action="store_true",
                   help="Re-render from saved .npz (no GPU needed)")
    args = p.parse_args()

    if args.render_only:
        render_from_saved()
    else:
        main()
