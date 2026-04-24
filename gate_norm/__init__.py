"""Gate Normalization — shift- and scale-invariant self-gated activations.

    y = x · g(γ · (x - μ(x)) / σ(x) + β)

where ``g`` is a pointwise gate (Gaussian CDF for :class:`NELU`, sigmoid for
:class:`NiLU`) and ``γ``, ``β`` are learnable scalars. ``γ`` is initialized
near zero and ``β`` at zero so the module recovers ``x · g(0)`` at init.

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

__version__ = "0.3.0"
