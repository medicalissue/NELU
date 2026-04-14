"""RMS-gate-normalized activations and GLU blocks.

  Pointwise:
    NELU    = GELU + RMS gate normalization + softplus-learnable γ
    NiLU    = SiLU + RMS gate normalization + softplus-learnable γ

  GLU FFN blocks:
    SwiGLU  = baseline (LLaMA-style)
    NiLUGLU = SwiGLU with NiLU on the gate projection
    NELUGLU = SwiGLU with NELU on the gate projection

All share the same principle: dividing the gate argument by rms(z)
gives exact forward scale invariance, f(alpha z) = alpha f(z).
γ is parameterized as softplus(raw_γ) so it is always positive and
initialized near zero (near-linear activation at training start).
"""

from .activations import NELU, NiLU, nelu, nilu, collect_gamma_stats
from .glu import SwiGLU, NiLUGLU, NELUGLU

__all__ = [
    # pointwise
    "NELU", "nelu",
    "NiLU", "nilu",
    # diagnostics
    "collect_gamma_stats",
    # GLU FFN blocks
    "SwiGLU", "NiLUGLU", "NELUGLU",
]
