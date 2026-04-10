"""Figure 2: Scale dependence corrupts both forward and backward behaviour.

(a) Forward:  Nonlinearity (best-affine-fit residual) vs input scale α.
(b) Backward: Gradient variance Var(f'(z)) vs input scale α.

Both evaluated for z ~ N(0, α² I_128) with α ∈ [0.5, 4].
"""

import math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

_INV_SQRT2 = 1.0 / math.sqrt(2)

# ── Activations ────────────────────────────────────────────────────
def relu(z):      return torch.relu(z)
def leaky(z):     return F.leaky_relu(z, 0.1)
def elu(z):       return F.elu(z)
def selu(z):      return F.selu(z)
def softplus(z):  return F.softplus(z)
def tanh_(z):     return torch.tanh(z)
def gelu(z):      return z * 0.5 * (1 + torch.erf(z * _INV_SQRT2))
def silu(z):      return z * torch.sigmoid(z)
def mish(z):      return z * torch.tanh(F.softplus(z))


# ── Metric 1: LS residual (forward nonlinearity) ──────────────────
@torch.no_grad()
def ls_residual(fn, d, alpha, n_samples=8192, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = (alpha * torch.randn(n_samples, d, generator=g)).double()
    y = fn(z.float()).double()

    y_mean = y.mean(dim=0, keepdim=True)
    z_mean = z.mean(dim=0, keepdim=True)
    yc = y - y_mean
    zc = z - z_mean

    sol = torch.linalg.lstsq(zc, yc)
    y_hat = zc @ sol.solution + y_mean

    resid = (y - y_hat).pow(2).sum()
    total = (y - y_mean).pow(2).sum()
    if total.item() < 1e-12:
        return 0.0
    return float(torch.sqrt(resid / total).item())


# ── Metric 2: gradient variance (backward) ────────────────────────
def grad_variance(fn, d, alpha, n_samples=8192, seed=0):
    """Var_z[ f'(z) ] where z ~ α·N(0, I_d).
    Computed elementwise via autograd (elementwise activations only)."""
    g = torch.Generator().manual_seed(seed)
    z = (alpha * torch.randn(n_samples, d, generator=g)).requires_grad_(True)
    y = fn(z)
    # Sum then backprop → df/dz elementwise
    y.sum().backward()
    grads = z.grad.detach().flatten()
    return float(grads.var().item())


def main():
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    d = 128
    # Dense α sweep for smooth curves
    alphas = np.geomspace(0.5, 4.0, 25)
    n_samples = 8192

    # Match fig2 color palette
    acts = [
        # name,      fn,       color,     linestyle, lw,   zorder
        ("GELU",     gelu,     "#4e79a7", "-",       1.6,  6),
        ("SiLU",     silu,     "#59a14f", "-",       1.2,  4),
        ("Mish",     mish,     "#76b7b2", "-",       1.2,  4),
        ("Softplus", softplus, "#edc948", "-",       1.0,  3),
        ("Tanh",     tanh_,    "#f28e2b", "-",       1.0,  3),
        ("ELU",      elu,      "#e15759", "-",       1.0,  3),
        ("SELU",     selu,     "#b07aa1", "-",       1.0,  3),
        ("ReLU",     relu,     "#000000", ":",       1.4,  7),
        ("LeakyReLU", leaky,   "#666666", ":",       1.0,  7),
    ]

    # ── Compute both metrics over α ────────────────────────────────
    print(f"d = {d}, n_samples = {n_samples}, α ∈ [{alphas[0]:.2f}, {alphas[-1]:.2f}]")
    nonlin = {name: np.zeros(len(alphas)) for name, *_ in acts}
    gvar   = {name: np.zeros(len(alphas)) for name, *_ in acts}
    for ai, a in enumerate(alphas):
        for name, fn, *_ in acts:
            nonlin[name][ai] = ls_residual(fn, d, float(a), n_samples)
            gvar[name][ai]   = grad_variance(fn, d, float(a), n_samples)

    # ── Figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))

    # (a) Forward nonlinearity
    ax = axes[0]
    for name, fn, color, ls, lw, z in acts:
        ax.plot(alphas, nonlin[name],
                color=color, linestyle=ls, lw=lw, label=name, zorder=z)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 1, 2, 4])
    ax.set_xticklabels(["0.5", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_xlabel(r"input scale  $\alpha$")
    ax.set_ylabel("nonlinearity (best-affine residual)")
    ax.set_xlim(alphas[0], alphas[-1])

    # (b) Gradient variance
    ax = axes[1]
    for name, fn, color, ls, lw, z in acts:
        ax.plot(alphas, gvar[name],
                color=color, linestyle=ls, lw=lw, label=name, zorder=z)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 1, 2, 4])
    ax.set_xticklabels(["0.5", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_xlabel(r"input scale  $\alpha$")
    ax.set_ylabel(r"$\mathrm{Var}(f'(z))$")
    ax.set_xlim(alphas[0], alphas[-1])
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5),
              handlelength=1.8, handletextpad=0.5, labelspacing=0.35)

    # ── Cleanup + labels ───────────────────────────────────────────
    for i, ax in enumerate(axes):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.4)
        ax.spines["bottom"].set_linewidth(0.4)
        ax.tick_params(width=0.4, length=3)
        ax.text(0.5, -0.32, f"({chr(97+i)})", transform=ax.transAxes,
                fontsize=9, fontweight="bold", ha="center")

    fig.tight_layout(pad=0.3, w_pad=1.2)
    fig.subplots_adjust(bottom=0.24, right=0.86)

    fig.savefig(out_dir / "fig2_scale_dep.png",
                bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig2_scale_dep.pdf",
                bbox_inches="tight", facecolor="white")
    print(f"Saved {out_dir}/fig2_scale_dep.{{png,pdf}}")
    plt.close()


if __name__ == "__main__":
    main()
