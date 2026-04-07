"""
Figure 2: NELU Activation Analysis
===================================
(a) NELU shape for various rms scales rho
(b) NELU derivative in normalised coordinates (scale-invariant)
(c) GELU derivative histograms for inputs drawn from N(0, rho^2)
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# NeurIPS style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "mathtext.fontset": "cm",
})

# ---------------------------------------------------------------------------
# Helpers (pure numpy, no torch dependency needed for analytic plots)
# ---------------------------------------------------------------------------
from scipy.stats import norm as _norm

def phi_cdf(x):
    """Standard normal CDF  Phi(x)."""
    return _norm.cdf(x)

def phi_pdf(x):
    """Standard normal PDF  phi(x)."""
    return _norm.pdf(x)

def nelu(z, rho):
    """NELU(z) = z * Phi(z / rho)."""
    z_hat = z / rho
    return z * phi_cdf(z_hat)

def nelu_deriv_normalised(z_hat):
    """h(z_hat) = Phi(z_hat) + z_hat * phi(z_hat)  -- the universal curve."""
    return phi_cdf(z_hat) + z_hat * phi_pdf(z_hat)

def gelu_deriv(z):
    """GELU'(z) = Phi(z) + z * phi(z)  (same formula, but evaluated at raw z)."""
    return phi_cdf(z) + z * phi_pdf(z)


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
rhos = [0.5, 1.0, 2.0, 4.0]
colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]  # ColorBrewer Dark2

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

# ========================== Panel (a) ======================================
ax = axes[0]
z = np.linspace(-4, 4, 800)

for rho, c in zip(rhos, colors):
    style = {"linestyle": "--", "linewidth": 2.0} if rho == 1.0 else {"linewidth": 1.6}
    label = rf"$\rho={rho}$"
    if rho == 1.0:
        label += " (GELU)"
    ax.plot(z, nelu(z, rho), color=c, label=label, **style)

ax.set_xlabel(r"$z$")
ax.set_ylabel(r"NELU$(z)$")
ax.set_title("(a) NELU activation shape")
ax.legend(frameon=False)
ax.axhline(0, color="grey", linewidth=0.4, zorder=0)
ax.axvline(0, color="grey", linewidth=0.4, zorder=0)

# ========================== Panel (b) ======================================
ax = axes[1]
z_hat = np.linspace(-4, 4, 800)
h = nelu_deriv_normalised(z_hat)

# Plot all four rho on top of each other to show they collapse
for rho, c in zip(rhos, colors):
    ax.plot(z_hat, h, color=c, linewidth=1.4, alpha=0.7,
            label=rf"$\rho={rho}$")

ax.set_xlabel(r"$\hat{z} = z / \rho$")
ax.set_ylabel(r"$h(\hat{z})$")
ax.set_title(r"(b) NELU derivative (all $\rho$ collapse)")
ax.legend(frameon=False)
ax.axhline(0, color="grey", linewidth=0.4, zorder=0)
ax.axhline(1, color="grey", linewidth=0.4, linestyle=":", zorder=0)

# ========================== Panel (c) ======================================
ax = axes[2]
np.random.seed(42)
n_samples = 20_000

violin_data = []
positions = []
for i, (rho, c) in enumerate(zip(rhos, colors)):
    samples = np.random.normal(0, rho, size=n_samples)
    derivs = gelu_deriv(samples)
    violin_data.append(derivs)
    positions.append(i)

parts = ax.violinplot(violin_data, positions=positions, showmedians=True,
                      showextrema=False, widths=0.7)

for i, (pc, c) in enumerate(zip(parts["bodies"], colors)):
    pc.set_facecolor(c)
    pc.set_alpha(0.65)
    pc.set_edgecolor("black")
    pc.set_linewidth(0.6)

parts["cmedians"].set_color("black")
parts["cmedians"].set_linewidth(1.2)

ax.set_xticks(positions)
ax.set_xticklabels([rf"$\rho={r}$" for r in rhos])
ax.set_ylabel(r"GELU$'(z)$")
ax.set_title(r"(c) GELU derivative, $z \sim \mathcal{N}(0,\rho^2)$")
ax.axhline(0.5, color="grey", linewidth=0.4, linestyle=":", zorder=0)
ax.axhline(0, color="grey", linewidth=0.4, zorder=0)
ax.axhline(1, color="grey", linewidth=0.4, linestyle=":", zorder=0)
ax.set_ylim(-0.18, 1.25)

# ---------------------------------------------------------------------------
plt.tight_layout()

out_dir = Path(__file__).resolve().parent.parent / "results"
out_dir.mkdir(exist_ok=True)

fig.savefig(out_dir / "fig2_nelu_analysis.png")
fig.savefig(out_dir / "fig2_nelu_analysis.pdf")
print(f"Saved to {out_dir / 'fig2_nelu_analysis.png'}")
print(f"Saved to {out_dir / 'fig2_nelu_analysis.pdf'}")
plt.close(fig)
