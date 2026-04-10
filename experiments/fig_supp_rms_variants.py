"""Supplementary: Apply RMS stop-gradient normalization to every linear unit.

For each activation f, we compare:
    (plain)  f(αz)
    (RMS)    f_rms(z) with the gating/nonlinear argument divided by sg[rms(z)]

We measure how close the output direction stays to the input direction
(identity in θ_out vs θ_in) across scales α ∈ [0.25, 8].

Metric: MSE of θ_out against θ_in, averaged over α and θ_in.

Activations covered:
    ReLU, LeakyReLU, ELU, SELU, CELU, GELU, SiLU (Swish), Mish,
    Softplus, Tanh, HardSwish, HardSigmoid*x, GLU-style x·sigmoid(x)
"""

import math, sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

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

# ═══════════════════════════════════════════════════════════════════
# Plain activations
# ═══════════════════════════════════════════════════════════════════
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
def sigmoid_gate(z):return z * torch.sigmoid(z * 0.5)  # another variant

# ═══════════════════════════════════════════════════════════════════
# RMS-normalized variants
# For each "self-gated" activation f(z) = z · g(z), we replace the
# gating argument with z/sg[rms(z)], giving f_rms(z) = z · g(z/rms).
# For non-gated ones (ReLU/ELU/…), we apply the trick "z -> rms · f(z/rms)"
# which restores positive 1-homogeneity for any pointwise f.
# ═══════════════════════════════════════════════════════════════════
def _rms(z, eps=1e-6):
    return z.pow(2).mean(-1, keepdim=True).add(eps).sqrt().detach()

def relu_rms(z):
    r = _rms(z); return r * torch.relu(z / r)
def leaky_rms(z):
    r = _rms(z); return r * F.leaky_relu(z / r, 0.1)
def elu_rms(z):
    r = _rms(z); return r * F.elu(z / r)
def selu_rms(z):
    r = _rms(z); return r * F.selu(z / r)
def celu_rms(z):
    r = _rms(z); return r * F.celu(z / r)
def gelu_rms(z):  # == NELU
    r = _rms(z); return z * 0.5 * (1 + torch.erf((z / r) * _INV_SQRT2))
def silu_rms(z):
    r = _rms(z); return z * torch.sigmoid(z / r)
def mish_rms(z):
    r = _rms(z); return z * torch.tanh(F.softplus(z / r))
def softplus_rms(z):
    r = _rms(z); return r * F.softplus(z / r)
def tanh_rms(z):
    r = _rms(z); return r * torch.tanh(z / r)
def hardswish_rms(z):
    r = _rms(z); return r * F.hardswish(z / r)
def sigmoid_gate_rms(z):
    r = _rms(z); return z * torch.sigmoid((z / r) * 0.5)


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
    ("GELU",      gelu,       gelu_rms),    # GELU-RMS == NELU
    ("x·σ(z/2)",  sigmoid_gate, sigmoid_gate_rms),
]


@torch.no_grad()
def angle_mse_vs_identity(fn, alphas, n=500):
    """Compute MSE(θ_out, θ_in) averaged over α and θ_in.
    Uses smallest-signed-angle between θ_out and θ_in.
    """
    theta = torch.linspace(-math.pi, math.pi, n)
    z_unit = torch.stack([torch.cos(theta), torch.sin(theta)], -1)  # (n,2)

    mse_per_alpha = []
    curves_out = []
    for alpha in alphas:
        y = fn(alpha * z_unit)
        # Direction undefined if output is zero — handle it as "no rotation"
        norm = y.norm(dim=-1)
        safe = norm > 1e-8
        theta_out = torch.atan2(y[:, 1], y[:, 0])
        # Signed smallest angle between (θ_out, θ_in)
        diff = theta_out - theta
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        # Where output ~ 0, treat as zero rotation (origin is ambiguous)
        diff = torch.where(safe, diff, torch.zeros_like(diff))
        mse_per_alpha.append((diff.pow(2).mean().item()))
        curves_out.append(theta_out.numpy())

    return float(np.mean(mse_per_alpha)), curves_out


def main():
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    alphas = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])

    # Compute MSE for both plain and RMS variants
    results = []
    for name, plain_fn, rms_fn in ACTS:
        mse_plain, _ = angle_mse_vs_identity(plain_fn, alphas)
        mse_rms, _ = angle_mse_vs_identity(rms_fn, alphas)
        results.append((name, mse_plain, mse_rms))
        print(f"  {name:10s}  plain={math.degrees(math.sqrt(mse_plain)):6.2f}°  "
              f"rms={math.degrees(math.sqrt(mse_rms)):6.2f}°")

    # Sort by plain MSE descending for readability
    results.sort(key=lambda x: -x[1])
    names = [r[0] for r in results]
    mse_p = np.array([r[1] for r in results])
    mse_r = np.array([r[2] for r in results])

    # Convert MSE (rad²) to RMS angle in degrees
    rmsd_plain = np.degrees(np.sqrt(mse_p))
    rmsd_rms = np.degrees(np.sqrt(mse_r))

    # ─── Plot 1: Grouped bar chart ──────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.2))
    x = np.arange(len(names))
    w = 0.38

    b1 = ax.bar(x - w/2, rmsd_plain, w, label="plain",
                color="#c0392b", alpha=0.85, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w/2, rmsd_rms, w, label="+ RMS stop-grad",
                color="#2471a3", alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(r"RMS $|\theta_{\mathrm{out}}-\theta_{\mathrm{in}}|$ (deg)", fontsize=8)
    ax.set_title("Direction preservation vs identity across scales "
                 r"$\alpha\in\{0.25,0.5,1,2,4,8\}$",
                 fontsize=9, pad=6)
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)

    # Value labels
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 1.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=5.5, color="#444444")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_supp_rms_bar.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig_supp_rms_bar.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved fig_supp_rms_bar.{{png,pdf}}")
    plt.close()

    # ─── Plot 2: Grid of θ_out vs θ_in for all activations ──────────
    # Show top-8 most different (plain vs rms) to keep it readable
    diffs = rmsd_plain - rmsd_rms
    order = np.argsort(-diffs)[:8]
    sel_names = [names[i] for i in order]

    # Build lookup of (plain_fn, rms_fn) by name
    fn_lookup = {n: (pf, rf) for n, pf, rf in ACTS}

    n_cols = 4
    n_rows = 2
    fig = plt.figure(figsize=(7.2, 4.2))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.55, wspace=0.32,
                           left=0.07, right=0.99, top=0.92, bottom=0.12)

    theta = np.linspace(-np.pi, np.pi, 500)
    tin_deg = np.degrees(theta)

    for i, name in enumerate(sel_names):
        r, c = divmod(i, n_cols)
        ax = fig.add_subplot(gs[r, c])
        plain_fn, rms_fn = fn_lookup[name]

        ax.plot([-180, 180], [-180, 180], color="#bbbbbb",
                lw=0.5, linestyle="--", zorder=0)

        cmap_p = LinearSegmentedColormap.from_list("cp",
                    ["#f2d2cc", "#c0392b", "#561a11"], N=256)
        cmap_r = LinearSegmentedColormap.from_list("cr",
                    ["#cfe0ec", "#2471a3", "#0f344d"], N=256)

        for ai, alpha in enumerate(alphas):
            frac = ai / max(len(alphas) - 1, 1)
            z = torch.tensor(np.stack([np.cos(theta), np.sin(theta)], -1),
                             dtype=torch.float32)

            for fn, cmap, dash, tag in [
                (plain_fn, cmap_p, "-",  "plain"),
                (rms_fn,   cmap_r, "-",  "rms"),
            ]:
                y = fn(alpha * z).numpy()
                tout = np.degrees(np.arctan2(y[:, 1], y[:, 0]))
                # Break on discontinuity
                jumps = np.where(np.abs(np.diff(tout)) > 90)[0]
                segs = np.split(np.arange(len(tin_deg)), jumps + 1) \
                    if len(jumps) else [np.arange(len(tin_deg))]
                color = cmap(0.25 + 0.65 * frac)
                for seg in segs:
                    if len(seg) < 2: continue
                    ax.plot(tin_deg[seg], tout[seg], color=color,
                            lw=0.9, alpha=0.85)

        ax.set_title(name, fontsize=8, fontweight="bold", pad=3)
        ax.set_xticks([-180, 0, 180]); ax.set_yticks([-180, 0, 180])
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if r == n_rows - 1:
            ax.set_xlabel(r"$\theta_{\mathrm{in}}$", fontsize=7)
        if c == 0:
            ax.set_ylabel(r"$\theta_{\mathrm{out}}$", fontsize=7)

    # Legend: plain (red) vs rms (blue), α gradient
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="#c0392b", lw=1.4, label="plain"),
        Line2D([0], [0], color="#2471a3", lw=1.4, label="+ RMS stop-grad"),
        Line2D([0], [0], color="#bbbbbb", lw=0.8, linestyle="--",
               label="identity"),
    ]
    fig.legend(handles=legend_elems, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=3,
               frameon=False, fontsize=7.5, handlelength=1.8)

    fig.suptitle(r"$\theta_{\mathrm{out}}$ vs $\theta_{\mathrm{in}}$ "
                 r"for plain vs RMS-normalized variants "
                 r"(shade = $\alpha$ ∈ $\{0.25,…,8\}$)",
                 fontsize=9, y=0.98)

    fig.savefig(out_dir / "fig_supp_rms_grid.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig_supp_rms_grid.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved fig_supp_rms_grid.{{png,pdf}}")
    plt.close()


if __name__ == "__main__":
    main()
