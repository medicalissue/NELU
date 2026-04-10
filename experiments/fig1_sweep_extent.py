"""Sweep different extents and render all — find the best-looking range."""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "savefig.dpi": 200,
    "mathtext.fontset": "cm",
})

out_dir = Path(__file__).resolve().parent.parent / "results"

# Load existing data to check if we can skip recomputation
existing = out_dir / "fig1_landscape_data.npz"
if existing.exists():
    d = np.load(existing)
    print(f"Existing data: steps={d['steps']}, ext={d['ext']}")

# We need to compute at multiple extents
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from nelu import NELU
from main_cifar_tinyimagenet import build_model

dev = "cuda" if torch.cuda.is_available() else "cpu"
M, S = (0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)
te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(M,S)])
loader = DataLoader(datasets.CIFAR10("./data",False,download=True,transform=te),
                    256,False,num_workers=2)

def get_w(m): return [p.data.clone() for p in m.parameters()]
def set_w(m,w):
    for p,wi in zip(m.parameters(),w): p.data.copy_(wi)

def filt_norm(d,w):
    for di,wi in zip(d,w):
        if di.dim()<=1: di.fill_(0)
        else:
            for i in range(di.shape[0]):
                di[i].mul_(wi[i].norm()/(di[i].norm()+1e-10))
    return d

@torch.no_grad()
def eval_loss(model, loader, dev):
    model.eval()
    tot,n=0.,0
    for x,y in loader:
        x,y=x.to(dev),y.to(dev)
        tot+=F.cross_entropy(model(x),y).item()*x.size(0); n+=x.size(0)
    return tot/n

def landscape(model, loader, dev, steps, ext):
    w0 = get_w(model)
    torch.manual_seed(777)
    d1 = filt_norm([torch.randn_like(w) for w in w0], w0)
    d2 = filt_norm([torch.randn_like(w) for w in w0], w0)
    cs = np.linspace(-ext, ext, steps)
    Z = np.zeros((steps,steps))
    for i,a in enumerate(cs):
        for j,b in enumerate(cs):
            set_w(model,[w+a*dd1+b*dd2 for w,dd1,dd2 in zip(w0,d1,d2)])
            Z[j,i] = eval_loss(model, loader, dev)
    set_w(model, w0)
    return cs, Z

# Load models
ckpt_dir = out_dir / "checkpoints"
acts = [
    ("ReLU","relu","resnet20_cifar10_relu_s42_best.pt"),
    ("GELU","gelu","resnet20_cifar10_gelu_s42_best.pt"),
    ("NELU","nelu","resnet20_cifar10_nelu_s42_best.pt"),
]

models = {}
for aname, akey, ckpt_name in acts:
    model = build_model("resnet20",10,32,akey).to(dev)
    ckpt = torch.load(ckpt_dir/ckpt_name, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    models[aname] = model

extents = [0.1, 0.2, 0.3, 0.5, 1.0]
steps = 41  # slightly coarser for speed

# Compute all
all_grids = {}
for ext in extents:
    for aname in ["ReLU","GELU","NELU"]:
        key = (aname, ext)
        print(f"  {aname} ext={ext}...")
        cs, Z = landscape(models[aname], loader, dev, steps, ext)
        all_grids[key] = (cs, Z)

# Save
np.savez(out_dir / "fig1_sweep_data.npz",
         **{f"{a}_{e}": all_grids[(a,e)][1] for a in ["ReLU","GELU","NELU"] for e in extents},
         extents=extents, steps=steps,
         **{f"coords_{e}": all_grids[("ReLU",e)][0] for e in extents})
print("Saved sweep data.")

# ── Render: 5 rows (extents) × 3 cols (activations) ──

# 3D
fig = plt.figure(figsize=(10, 14))
ls = LightSource(azdeg=315, altdeg=45)

for row, ext in enumerate(extents):
    for col, aname in enumerate(["ReLU","GELU","NELU"]):
        ax = fig.add_subplot(len(extents), 3, row*3+col+1, projection='3d')
        cs, Z = all_grids[(aname, ext)]
        X, Y = np.meshgrid(cs, cs)
        Z_smooth = gaussian_filter(Z, sigma=0.8)
        ax.plot_surface(X, Y, Z_smooth, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True, alpha=0.9)
        ax.view_init(elev=30, azim=-60)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.xaxis.pane.fill=False; ax.yaxis.pane.fill=False; ax.zaxis.pane.fill=False
        ax.xaxis.pane.set_edgecolor('none'); ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none'); ax.grid(False)
        if row==0:
            ax.set_title(aname, fontweight="bold", fontsize=10, pad=-2)
        if col==0:
            ax.text2D(-0.15, 0.5, f"ext={ext}", transform=ax.transAxes,
                     fontsize=9, fontweight="bold", rotation=90, va="center")

fig.subplots_adjust(wspace=0, hspace=0.02, left=0.08, right=0.98, bottom=0.02, top=0.96)
fig.savefig(out_dir / "fig1_sweep_3d.png", bbox_inches="tight", facecolor="white")
print("Saved fig1_sweep_3d.png")
plt.close()

# Contour
fig, axes = plt.subplots(len(extents), 3, figsize=(7, 10))
for row, ext in enumerate(extents):
    vmin = min(all_grids[(a,ext)][1].min() for a in ["ReLU","GELU","NELU"])
    vmax = np.percentile(np.stack([all_grids[(a,ext)][1] for a in ["ReLU","GELU","NELU"]]), 85)
    for col, aname in enumerate(["ReLU","GELU","NELU"]):
        ax = axes[row, col]
        cs, Z = all_grids[(aname, ext)]
        X, Y = np.meshgrid(cs, cs)
        ax.contourf(X, Y, Z, levels=30, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        ax.contour(X, Y, Z, levels=10, colors="white", linewidths=0.2, alpha=0.4)
        ax.plot(0, 0, "k*", markersize=3, zorder=5)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(0.3)
        if row==0:
            ax.set_title(aname, fontweight="bold", fontsize=9)
        if col==0:
            ax.set_ylabel(f"ext={ext}", fontsize=8, fontweight="bold")

fig.tight_layout(pad=0.2, h_pad=0.3, w_pad=0.3)
fig.savefig(out_dir / "fig1_sweep_contour.png", bbox_inches="tight", facecolor="white")
print("Saved fig1_sweep_contour.png")
plt.close()

print("\nDone.")
