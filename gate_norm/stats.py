"""Reduction statistics used by Gate Normalization.

We centre the gate input: ``normed = (z - μ) / σ``. Given the reduction axes
and ``eps`` this module computes ``(μ, rsigma)`` once — the forward uses them
to build the gate input, the backward reuses them to avoid a second pass over
``z``. Statistics are computed in float32 regardless of input dtype so AMP
autocast doesn't underflow.
"""

from __future__ import annotations

import torch


def layer_stats(
    z: torch.Tensor, axes: tuple[int, ...], eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(μ, 1/σ)`` over ``axes`` in float32, both with ``keepdim=True``.

    Uses the identity ``Var = E[z²] - E[z]²`` so only two reductions are
    required; keeping ``rsigma`` rather than ``σ`` spares a divide on the hot
    path. ``eps`` is added inside the variance before sqrt, matching the
    LayerNorm convention.
    """
    z32 = z.float() if z.dtype != torch.float32 else z
    mu = z32.mean(dim=axes, keepdim=True)
    var = z32.var(dim=axes, keepdim=True, unbiased=False)
    rsigma = (var + eps).rsqrt()
    return mu, rsigma
