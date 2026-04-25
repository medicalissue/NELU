"""Gate Normalization — scale-invariant self-gated activations.

    y = x · g(γ · x / rms(x))

where ``g`` is a pointwise gate (Gaussian CDF for :class:`NELU`, sigmoid
for :class:`NiLU`) and ``γ`` is a non-learnable buffer scheduled by the
trainer (typically warmed up from 0 → 1 alongside the LR warmup, then
held at 1 for the rest of training).

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

To match the trainer's LR warmup, drive γ with :class:`GammaWarmup`::

>>> from gate_norm import GammaWarmup
>>> sched = GammaWarmup(model, warmup_steps=20 * steps_per_epoch)
>>> for step, batch in enumerate(loader):
...     sched.step(step)
...     # ... usual forward / backward / optimizer.step()
"""

from .activations import NELU, NiLU
from .core import GateNorm, gate_norm
from .functional import nelu, nilu
from .glu import NELUGLU, NiLUGLU, SwiGLU
from .logging import collect_gamma_stats
from .scheduler import GammaWarmup

__all__ = [
    "GateNorm",
    "GammaWarmup",
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

__version__ = "0.4.0"
