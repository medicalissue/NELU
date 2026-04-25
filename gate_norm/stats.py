"""Reduction statistics used by Gate Normalization.

We rescale the gate input by its root-mean-square: ``normed = z / rms(z)``,
where ``rms(z) = sqrt(mean(z²) + eps)``. Unlike LayerNorm we deliberately
*do not* subtract the mean — preserving the DC component is what makes the
outer ``z · g(...)`` retain ReLU-style "deactivate the negative side"
inductive bias. Statistics are computed in float32 regardless of input
dtype so AMP autocast doesn't underflow.
"""

from __future__ import annotations

import torch


def layer_stats(
    z: torch.Tensor, axes: tuple[int, ...], eps: float
) -> torch.Tensor:
    """Return ``1 / rms(z)`` over ``axes`` in float32 with ``keepdim=True``.

    Returning the reciprocal saves a divide on the hot forward path; the
    backward reuses the same value rather than recomputing.
    """
    z32 = z.float() if z.dtype != torch.float32 else z
    ms = z32.pow(2).mean(dim=axes, keepdim=True)
    return (ms + eps).rsqrt()
