r"""ResAct: Residual Activation — cross-layer linear mixing inside the activation.

Built on the odd-even decomposition of GELU::

    GELU(x) = 0.5·x + 0.5·x·erf(x/√2)
              \____/    \____________/
              linear     nonlinear (even)
              (½)        (½)

We generalize the fixed ½ on the linear part to a learnable sigmoid mix
between the current layer's input ``x_l`` and the previous activation's
output ``a_{l-1}``::

    y_l = GELU(x_l) − 0.5·x_l + 0.5·[σ(α)·x_l + (1 − σ(α))·a_{l-1}]
        = 0.5·x_l·erf(x_l/√2)
        + 0.5·[σ(α)·x_l + (1 − σ(α))·a_{l-1}]

where:
    α          : learnable scalar per layer
    σ(α)       : current-layer mix weight, ∈ (0, 1)
    a_{l-1}    : previous activation's output (cached buffer)

Init choices:
    α_init = 5  → σ ≈ 0.993 → near-vanilla GELU at start (safe baseline)
    α_init = 0  → σ = 0.5   → uniform mix at start (more aggressive)

The previous-activation buffer is module-local and cached at every
forward call. This works for sequential-style architectures where the
forward call order matches the layer ordering. The first layer (or any
layer whose previous-activation shape doesn't match the current input)
falls back to vanilla GELU automatically.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResActGELU(nn.Module):
    """ResAct on GELU's odd-even decomposition.

    Parameters
    ----------
    alpha_init : float
        Initial value of the learnable scalar α. ``σ(α)`` is the
        current-layer mix weight in the linear part. Default 5.0
        (≈ vanilla GELU at start).
    """

    def __init__(self, alpha_init: float = 5.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        # Previous activation output, cached per-module across forward
        # calls. Not registered as a buffer because we don't want it in
        # state_dict; recomputed every forward.
        self._prev: torch.Tensor | None = None

    def reset_prev(self) -> None:
        """Clear the cached previous activation output."""
        self._prev = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gelu_out = F.gelu(x)

        prev = self._prev
        if prev is None or prev.shape != x.shape or prev.device != x.device:
            # First call (or shape/device mismatch): fall back to vanilla.
            self._prev = gelu_out.detach()
            return gelu_out

        s = torch.sigmoid(self.alpha)
        # GELU(x) - 0.5*x + 0.5*[s*x + (1-s)*prev]
        y = gelu_out - 0.5 * x + 0.5 * (s * x + (1.0 - s) * prev)

        # Cache current GELU output (= activation output) as prev for next
        # layer. Detach so backward doesn't chain across layers through
        # the buffer (gradient flows through the standard residual path).
        self._prev = gelu_out.detach()
        return y

    def extra_repr(self) -> str:
        with torch.no_grad():
            s = torch.sigmoid(self.alpha).item()
        return f"alpha={self.alpha.item():.3f}, sigmoid(alpha)={s:.4f}"


def collect_resact_stats(model: nn.Module, prefix: str = "resact") -> dict:
    """Collect per-layer α and σ(α) values from every ResActGELU module.

    Returns flat dict suitable for ``wandb.log``::

        resact/alpha/layer_<i>
        resact/sigmoid_alpha/layer_<i>
        resact/alpha/{mean,min,max,std}
        resact/sigmoid_alpha/{mean,min,max,std}
    """
    alphas: list[float] = []
    sigmas: list[float] = []
    out: dict[str, float] = {}
    for m in model.modules():
        if not isinstance(m, ResActGELU):
            continue
        a = m.alpha.detach().float().item()
        s = torch.sigmoid(m.alpha.detach().float()).item()
        out[f"{prefix}/alpha/layer_{len(alphas)}"] = a
        out[f"{prefix}/sigmoid_alpha/layer_{len(sigmas)}"] = s
        alphas.append(a)
        sigmas.append(s)
    if alphas:
        for vals, key in [(alphas, "alpha"), (sigmas, "sigmoid_alpha")]:
            n = len(vals)
            mean = sum(vals) / n
            var = sum((x - mean) ** 2 for x in vals) / max(1, n - 1)
            out[f"{prefix}/{key}/mean"] = mean
            out[f"{prefix}/{key}/min"] = min(vals)
            out[f"{prefix}/{key}/max"] = max(vals)
            out[f"{prefix}/{key}/std"] = var ** 0.5
            out[f"{prefix}/{key}/n_modules"] = float(n)
    return out
