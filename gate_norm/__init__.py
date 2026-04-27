"""Gate Normalization — scale-invariant self-gated activations.

    y = x · g(γ · x / rms(x))

where ``g`` is a pointwise gate (Gaussian CDF for :class:`NELU`, sigmoid
for :class:`NiLU`) and ``γ`` is a single learnable scalar shared per
module.

Quick start
-----------

>>> import torch
>>> from gate_norm import NELU, NiLU
>>> x = torch.randn(4, 128)
>>> NELU()(x).shape
torch.Size([4, 128])

For NCHW convolutional feature maps::

>>> x_conv = torch.randn(4, 64, 32, 32)
>>> NiLU(norm_axes="sample")(x_conv).shape
torch.Size([4, 64, 32, 32])
"""

from .activations import NELU, NiLU
from .affine import NELU_AFF, NiLU_AFF, NELU_AFFCW, NiLU_AFFCW
from .core import GateNorm, gate_norm
from .functional import nelu, nilu
from .glu import NELUGLU, NiLUGLU, SwiGLU
from .ln_beta import NELU_LN, NiLU_LN
from .logging import collect_gamma_stats

__all__ = [
    "GateNorm",
    "NELU",
    "NiLU",
    "NELU_LN",
    "NiLU_LN",
    "NELU_AFF",
    "NiLU_AFF",
    "NELU_AFFCW",
    "NiLU_AFFCW",
    "NELUGLU",
    "NiLUGLU",
    "SwiGLU",
    "gate_norm",
    "nelu",
    "nilu",
    "collect_gamma_stats",
]

__version__ = "0.1.0"
