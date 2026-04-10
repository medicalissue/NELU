"""Supplementary (v3): α-variance and TRUE nonlinearity of RMS-normalized activations.

Distinguishes two properties:

  (1) Scale invariance — α-variance
        E_u[ Var_α[ direction(f(αu)) ] ]
        = 0 iff output direction is independent of input magnitude

  (2) Nonlinearity — residual after subtracting BEST linear approximation
        nonlin²(f) = E[‖f(z) − A z‖²] / E[‖f(z)‖²]
        where A is the least-squares linear map fit to f on the input distribution.
        = 0 iff f is (affine-)linear, regardless of scaling/rotation.

  Why LS residual and not angular deviation from identity?
  Because y = x/2 or y = diag(a)·x are linear but would show non-zero
  angular deviation. The LS residual correctly calls these "zero nonlinearity".

We sweep d ∈ {2, 8, 32, 128, 512} to show the asymptotic cleanup.
"""

import math, sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
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
    "legend.fontsize": 6.5,
    "savefig.dpi": 300,
})

_INV_SQRT2 = 1.0 / math.sqrt(2)

# ── Activations ────────────────────────────────────────────────────
def relu(z):        return torch.relu(z)
def leaky(z):       return F.leaky_relu(z, 0.1)
def elu(z):         return F.elu(z)
def selu(z):        return F.selu(z)
def celu(z):        return F.celu(z)
def gelu(z):        return z * 0.5 * (1 + torch.erf(z * _INV_SQRT2))
def silu(z):        return z * torch.sigmoid(z)
def mish(z):        return z * torch.tanh(F.softplus(z))
def softplus(z):    return F.softplus(z)
def tanh_(z):       return torch.tanh(z)
def hardswish(z):   return F.hardswish(z)

def _rms(z, eps=1e-6):
    return z.pow(2).mean(-1, keepdim=True).add(eps).sqrt().detach()

def relu_rms(z):       r = _rms(z); return r * torch.relu(z / r)
def leaky_rms(z):      r = _rms(z); return r * F.leaky_relu(z / r, 0.1)
def elu_rms(z):        r = _rms(z); return r * F.elu(z / r)
def selu_rms(z):       r = _rms(z); return r * F.selu(z / r)
def celu_rms(z):       r = _rms(z); return r * F.celu(z / r)
def gelu_rms(z):
    r = _rms(z)
    return z * 0.5 * (1 + torch.erf((z / r) * _INV_SQRT2))
def silu_rms(z):
    r = _rms(z); return z * torch.sigmoid(z / r)
def mish_rms(z):
    r = _rms(z); return z * torch.tanh(F.softplus(z / r))
def softplus_rms(z):   r = _rms(z); return r * F.softplus(z / r)
def tanh_rms(z):       r = _rms(z); return r * torch.tanh(z / r)
def hardswish_rms(z):  r = _rms(z); return r * F.hardswish(z / r)

ACTS = [
    ("ReLU",      relu,       relu_rms),
    ("LeakyReLU", leaky,      leaky_rms),
    ("ELU",       elu,        elu_rms),
    ("SELU",      selu,       selu_rms),
    ("CELU",      celu,       celu_rms),
    ("Softplus",  softplus,   softplus_rms),
    ("Tanh",      tanh_,      tanh_rms),
    ("HardSwish", hardswish,  hardswish_rms),
    ("SiLU",      silu,       silu_rms),
    ("Mish",      mish,       mish_rms),
    ("GELU",      gelu,       gelu_rms),
]

# ── Metrics ────────────────────────────────────────────────────────
@torch.no_grad()
def alpha_variance_deg(fn, d, alphas, n_dirs=512, seed=0):
    """Angular std of output direction across α, averaged over input directions."""
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n_dirs, d, generator=g)
    u = u / (u.norm(dim=-1, keepdim=True) + 1e-12)  # unit directions

    outs = []
    for a in alphas:
        y = fn(a * u)
        y = y / (y.norm(dim=-1, keepdim=True) + 1e-12)
        outs.append(y)
    Y = torch.stack(outs, dim=0)  # (A, n, d)

    ref_idx = len(alphas) // 2  # α=1.0 in middle
    y_ref = Y[ref_idx:ref_idx+1]
    cos = (Y * y_ref).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    ang = torch.arccos(cos)  # (A, n)  angle vs reference α
    return math.degrees(ang.std(dim=0).mean().item())


@torch.no_grad()
def _ls_residual_at_scale(fn, d, alpha, n_samples=8192, seed=0):
    """Relative L2 residual of f at input scale α:
        z ~ α · N(0, I_d)
        minimize over (A, b):  E[ ‖f(z) − (A z + b)‖² ]
    Returns sqrt(residual / variance)."""
    g = torch.Generator().manual_seed(seed)
    z = (alpha * torch.randn(n_samples, d, generator=g)).double()
    y = fn(z.float()).double()

    y_mean = y.mean(dim=0, keepdim=True)
    z_mean = z.mean(dim=0, keepdim=True)
    yc = y - y_mean
    zc = z - z_mean

    sol = torch.linalg.lstsq(zc, yc)
    X = sol.solution
    y_hat = zc @ X + y_mean

    resid = (y - y_hat).pow(2).sum()
    total = (y - y_mean).pow(2).sum()
    if total.item() < 1e-12:
        return 0.0
    return float(torch.sqrt(resid / total).item())


@torch.no_grad()
def nonlinearity_residual(fn, d, alphas, n_samples=8192, seed=0):
    """Scale-averaged nonlinearity.

    For each α ∈ alphas, compute LS residual under input distribution
    z ~ α·N(0, I_d), then average.  Also returns per-α values so we
    can diagnose scale-dependence of the expressivity itself.
    """
    per_alpha = []
    for a in alphas:
        per_alpha.append(_ls_residual_at_scale(fn, d, float(a),
                                               n_samples=n_samples, seed=seed))
    per_alpha = np.array(per_alpha)
    return float(per_alpha.mean()), per_alpha


@torch.no_grad()
def measure(plain_fn, rms_fn, d, alphas, n_dirs=512, n_samples=8192):
    pa_nl, pa_per = nonlinearity_residual(plain_fn, d, alphas, n_samples)
    ra_nl, ra_per = nonlinearity_residual(rms_fn,   d, alphas, n_samples)
    return dict(
        plain_alpha      = alpha_variance_deg(plain_fn, d, alphas, n_dirs),
        plain_nonlin     = pa_nl,
        plain_nonlin_per = pa_per,
        rms_alpha        = alpha_variance_deg(rms_fn,   d, alphas, n_dirs),
        rms_nonlin       = ra_nl,
        rms_nonlin_per   = ra_per,
    )


def main():
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    alphas = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    dims = [2, 8, 32, 128, 512]
    n_dirs = 512
    n_samples = 8192

    results = {name: {
        "plain_alpha":      np.zeros(len(dims)),
        "plain_nonlin":     np.zeros(len(dims)),
        "plain_nonlin_per": np.zeros((len(dims), len(alphas))),
        "rms_alpha":        np.zeros(len(dims)),
        "rms_nonlin":       np.zeros(len(dims)),
        "rms_nonlin_per":   np.zeros((len(dims), len(alphas))),
    } for name, _, _ in ACTS}

    for di, d in enumerate(dims):
        print(f"\n=== d = {d} ===")
        print(f"{'act':10s}  "
              f"{'plain α-var':>12s}  {'plain nonlin':>13s}  "
              f"{'rms α-var':>11s}  {'rms nonlin':>11s}")
        for name, pf, rf in ACTS:
            m = measure(pf, rf, d, alphas, n_dirs=n_dirs, n_samples=n_samples)
            results[name]["plain_alpha"][di]      = m["plain_alpha"]
            results[name]["plain_nonlin"][di]     = m["plain_nonlin"]
            results[name]["plain_nonlin_per"][di] = m["plain_nonlin_per"]
            results[name]["rms_alpha"][di]        = m["rms_alpha"]
            results[name]["rms_nonlin"][di]       = m["rms_nonlin"]
            results[name]["rms_nonlin_per"][di]   = m["rms_nonlin_per"]
            print(f"{name:10s}  {m['plain_alpha']:12.3f}  {m['plain_nonlin']:13.4f}  "
                  f"{m['rms_alpha']:11.3f}  {m['rms_nonlin']:11.4f}")

    # ── Per-α breakdown at d_ref (printed table for paper) ──
    d_ref_print = 128
    di_p = dims.index(d_ref_print)
    print(f"\n=== Per-α nonlinearity breakdown at d = {d_ref_print} ===")
    header = f"{'act':10s}  " + "  ".join([f"α={float(a):>5.2f}" for a in alphas]) + "   mean"
    print("PLAIN:")
    print(header)
    for name, _, _ in ACTS:
        vals = results[name]["plain_nonlin_per"][di_p]
        row = f"{name:10s}  " + "  ".join([f"{v:7.4f}" for v in vals]) + f"  {vals.mean():7.4f}"
        print(row)
    print("+RMS:")
    print(header)
    for name, _, _ in ACTS:
        vals = results[name]["rms_nonlin_per"][di_p]
        row = f"{name:10s}  " + "  ".join([f"{v:7.4f}" for v in vals]) + f"  {vals.mean():7.4f}"
        print(row)

    # ═══════════════════════════════════════════════════════════════
    # Plot 1: α-variance vs d
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    cmap = plt.cm.tab20
    for ai, (name, _, _) in enumerate(ACTS):
        col = cmap(ai / len(ACTS))
        axes[0].plot(dims, results[name]["plain_alpha"], 'o-',
                     color=col, lw=1.2, ms=3.5, label=name)
        axes[1].plot(dims, results[name]["rms_alpha"], 'o-',
                     color=col, lw=1.2, ms=3.5, label=name)
    for ax, title in zip(axes, ["Plain activations", "+ RMS stop-grad"]):
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims); ax.set_xticklabels([str(d) for d in dims])
        ax.set_xlabel("dimension $d$", fontsize=8)
        ax.set_ylabel(r"$\alpha$-variance (deg)", fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=7)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                   frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_supp_alpha_var_vs_d.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig_supp_alpha_var_vs_d.pdf", bbox_inches="tight", facecolor="white")
    print(f"\nSaved fig_supp_alpha_var_vs_d.{{png,pdf}}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # Plot 2: Nonlinearity vs d
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ai, (name, _, _) in enumerate(ACTS):
        col = cmap(ai / len(ACTS))
        axes[0].plot(dims, results[name]["plain_nonlin"], 'o-',
                     color=col, lw=1.2, ms=3.5, label=name)
        axes[1].plot(dims, results[name]["rms_nonlin"], 'o-',
                     color=col, lw=1.2, ms=3.5, label=name)
    for ax, title in zip(axes, ["Plain activations", "+ RMS stop-grad"]):
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims); ax.set_xticklabels([str(d) for d in dims])
        ax.set_xlabel("dimension $d$", fontsize=8)
        ax.set_ylabel("Nonlinearity (LS residual ratio)", fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=7)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                   frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_supp_nonlin_vs_d.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig_supp_nonlin_vs_d.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved fig_supp_nonlin_vs_d.{{png,pdf}}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # Plot 3: Trade-off — nonlinearity vs α-variance at d=128
    # ═══════════════════════════════════════════════════════════════
    d_ref = 128
    di = dims.index(d_ref)

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.0))

    # Sort labels by plain α-var for consistent label placement
    label_offsets = {
        # name: (dx, dy) in points
        "ReLU":      ( 5,  0),
        "LeakyReLU": ( 5,  6),
        "ELU":       ( 5, -3),
        "CELU":      ( 5,  4),
        "SELU":      ( 5, -3),
        "Tanh":      (-5, -8),
        "Softplus":  ( 5,  0),
        "HardSwish": ( 5,  4),
        "SiLU":      ( 5,  4),
        "Mish":      ( 5, -8),
        "GELU":      ( 5,  4),
    }

    for name, _, _ in ACTS:
        pa = results[name]["plain_alpha"][di]
        pn = results[name]["plain_nonlin"][di]
        ra = results[name]["rms_alpha"][di]
        rn = results[name]["rms_nonlin"][di]

        # Arrow plain → rms
        ax.annotate("", xy=(ra, rn), xytext=(pa, pn),
                    arrowprops=dict(arrowstyle="->", color="#999999",
                                    lw=0.6, alpha=0.5))
        ax.scatter(pa, pn, s=26, color="#c0392b", alpha=0.75,
                   edgecolors="white", linewidths=0.5, zorder=3)
        ax.scatter(ra, rn, s=46, color="#2471a3", alpha=0.9,
                   edgecolors="white", linewidths=0.5, zorder=4,
                   marker="s")

        dx, dy = label_offsets.get(name, (5, 3))
        ax.annotate(name, xy=(ra, rn),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.5, color="#1a3a5c", zorder=5)

    ax.set_xlabel(r"$\alpha$-variance (deg) $\longrightarrow$ scale dependence",
                  fontsize=8)
    ax.set_ylabel(r"Nonlinearity (LS residual) $\longrightarrow$ expressivity",
                  fontsize=8)
    ax.set_title(f"Scale-invariance vs nonlinearity trade-off ($d = {d_ref}$)",
                 fontsize=9, pad=6)
    ax.axvline(0, color="#666666", lw=0.4, linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, lw=0.4)

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#c0392b',
               markersize=6, label='plain'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2471a3',
               markersize=6, label='+ RMS stop-grad'),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_supp_tradeoff.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig_supp_tradeoff.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved fig_supp_tradeoff.{{png,pdf}}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # Plot 4: Grouped bar — nonlinearity only, d=128, sorted
    # ═══════════════════════════════════════════════════════════════
    names_sorted = sorted([n for n, _, _ in ACTS],
                          key=lambda n: -results[n]["rms_nonlin"][di])
    plain_nl = [results[n]["plain_nonlin"][di] for n in names_sorted]
    rms_nl   = [results[n]["rms_nonlin"][di]   for n in names_sorted]

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.0))
    x = np.arange(len(names_sorted)); w = 0.38
    ax.bar(x - w/2, plain_nl, w, color="#c0392b", alpha=0.85,
           label="plain", edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, rms_nl, w, color="#2471a3", alpha=0.85,
           label="+ RMS stop-grad", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names_sorted, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Nonlinearity (LS residual)", fontsize=8)
    ax.set_title(rf"Expressive power at $d = {d_ref}$ "
                 rf"(sorted by RMS variant)", fontsize=9, pad=6)
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_supp_nonlin_bar.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig_supp_nonlin_bar.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved fig_supp_nonlin_bar.{{png,pdf}}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # Plot 5: Nonlinearity vs α at d_ref — shows scale-dependence of
    # expressivity itself.  GELU changes a lot with α, NELU flat.
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    alpha_arr = alphas.numpy()
    for ai, (name, _, _) in enumerate(ACTS):
        col = cmap(ai / len(ACTS))
        axes[0].plot(alpha_arr, results[name]["plain_nonlin_per"][di],
                     'o-', color=col, lw=1.2, ms=3.5, label=name)
        axes[1].plot(alpha_arr, results[name]["rms_nonlin_per"][di],
                     'o-', color=col, lw=1.2, ms=3.5, label=name)
    for ax, title in zip(axes, ["Plain activations", "+ RMS stop-grad"]):
        ax.set_xscale("log", base=2)
        ax.set_xticks(alpha_arr)
        ax.set_xticklabels([f"{a:g}" for a in alpha_arr])
        ax.set_xlabel(r"input scale $\alpha$ ($z \sim \alpha\cdot\mathcal{N}(0,I)$)",
                      fontsize=8)
        ax.set_ylabel("Nonlinearity (LS residual)", fontsize=8)
        ax.set_title(title + f"  ($d = {d_ref}$)", fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=7)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                   frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_supp_nonlin_vs_alpha.png",
                bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig_supp_nonlin_vs_alpha.pdf",
                bbox_inches="tight", facecolor="white")
    print(f"Saved fig_supp_nonlin_vs_alpha.{{png,pdf}}")
    plt.close()


if __name__ == "__main__":
    main()
