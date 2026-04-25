"""Fused CUDA backend for Gate Normalization (currently disabled).

The custom op was written for the centered-and-learnable variant
``y = x · g(γ · (x - μ)/σ + β)`` and has not yet been ported to the
RMS-only form ``y = x · g(γ · x / rms(x))``. Until it is, every call
site in :mod:`gate_norm.core` falls through to the pure-PyTorch path,
which already trains via inductor at native-fp32-equivalent speed.

Set ``GATE_NORM_FORCE_PYTHON=1`` to silence any future kernel availability
check explicitly. The :func:`should_use_cuda` dispatcher already returns
``False`` here because :data:`gate_norm.core.GateNorm._CUDA_OP` is
``None`` for every shipped activation, so this file currently exposes no
public surface beyond a stub :func:`fused_forward` that raises if some
caller wires the kernel up by hand.
"""

from __future__ import annotations

import torch


def fused_forward(*args, **kwargs) -> torch.Tensor:  # pragma: no cover
    raise RuntimeError(
        "Fused CUDA kernel for the RMS-only Gate Normalization form has not "
        "been written yet. Use the native PyTorch path (the default)."
    )
