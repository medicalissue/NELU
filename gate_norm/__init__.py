"""Gate Normalization — scale-invariant self-gated activations.

    y = x · g(γ · x / rms(x))

where ``g`` is a pointwise gate (Gaussian CDF for :class:`NELU`, sigmoid for
:class:`NiLU`) and ``γ`` is a single learnable scalar initialized near zero.

Quick start
-----------

>>> import torch, torch.nn as nn
>>> from gate_norm import NELU, NiLU
>>> x = torch.randn(4, 128)
>>> NELU()(x).shape
torch.Size([4, 128])

For NCHW convolutional feature maps::

>>> x_conv = torch.randn(4, 64, 32, 32)
>>> NiLU(rms_mode="per_sample")(x_conv).shape
torch.Size([4, 64, 32, 32])
"""

from .activations import NELU, NiLU
from .core import GateNorm, gate_norm
from .functional import nelu, nilu
from .glu import NELUGLU, NiLUGLU, SwiGLU
from .logging import collect_gamma_stats

__all__ = [
    "GateNorm",
    "NELU",
    "NiLU",
    "NELUGLU",
    "NiLUGLU",
    "SwiGLU",
    "gate_norm",
    "nelu",
    "nilu",
    "collect_gamma_stats",
]

__version__ = "0.1.0"
