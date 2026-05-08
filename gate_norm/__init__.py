"""Gate Normalization — self-gated activations with channel-aware gating.

Default form (``NELU`` / ``NiLU``)::

    μ_c, var_c = pool over position axis (CNN: H×W; Transformer: T)
    z_norm     = (x − μ_c) / sqrt(var_c + ε)
    gate       = g(γ_c · z_norm + β_c)        # γ_c, β_c learnable per-channel
    y          = x · gate

The position axis is rank-dispatched: spatial ``(2, 3)`` for 4-D CNN
tensors, token ``(1,)`` for 3-D Transformer tensors. The per-channel
``γ_c, β_c`` are materialized lazily on the first forward.

Quick start
-----------

>>> import torch
>>> from gate_norm import NELU, NiLU
>>> NELU()(torch.randn(4, 128, 32, 32)).shape   # CNN  (B, C, H, W)
torch.Size([4, 128, 32, 32])
>>> NiLU()(torch.randn(4, 64, 256)).shape       # Tx   (B, T, C)
torch.Size([4, 64, 256])

Variants kept for ablation:

* :class:`NELU_RMS` / :class:`NiLU_RMS` — legacy RMS-only, scalar γ
  (the original paper draft form).
* :class:`NELU_AFF` / :class:`NiLU_AFF` — affine gate, no normalization
  (scalar γ, β).
* :class:`NELU_AFFCW` / :class:`NiLU_AFFCW` — affine gate, channel-wise
  γ, β; no normalization.
* :class:`xLN` — LayerNorm-as-multiplicative-gate.
"""

from .activations import NELU, NiLU, NELU_RMS, NiLU_RMS
from .affine import NELU_AFF, NiLU_AFF, NELU_AFFCW, NiLU_AFFCW
from .core import GateNorm, gate_norm
from .functional import nelu, nilu
from .glu import NELUGLU, NiLUGLU, SwiGLU
from .ln_beta import NELU_LN, NiLU_LN
from .logging import collect_gamma_stats
from .resact import ResActGELU, collect_resact_stats
from .xln import xLN

__all__ = [
    "GateNorm",
    "NELU",
    "NiLU",
    "NELU_RMS",
    "NiLU_RMS",
    "NELU_LN",
    "NiLU_LN",
    "NELU_AFF",
    "NiLU_AFF",
    "NELU_AFFCW",
    "NiLU_AFFCW",
    "NELUGLU",
    "NiLUGLU",
    "SwiGLU",
    "ResActGELU",
    "collect_resact_stats",
    "xLN",
    "gate_norm",
    "nelu",
    "nilu",
    "collect_gamma_stats",
]

__version__ = "0.1.0"
