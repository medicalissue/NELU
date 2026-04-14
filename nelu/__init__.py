"""RMS-gate-normalized activations and GLU blocks.

  Pointwise:
    NELU    = GELU   + RMS gate normalization + learnable γ
    NiLU    = SiLU   + RMS gate normalization + learnable γ

  GLU FFN blocks:
    SwiGLU  = baseline (LLaMA-style)
    NiLUGLU = SwiGLU with NiLU on the gate projection
    NELUGLU = SwiGLU with NELU on the gate projection

All share the same principle: dividing the gate argument by rms(z)
gives exact forward scale invariance, f(alpha z) = alpha f(z).
The learnable γ is initialized small so the module starts near-linear
and training grows γ to recover the gated nonlinearity.
"""

from .activations import NELU, NiLU, nelu, nilu
from .glu import SwiGLU, NiLUGLU, NELUGLU

__all__ = [
    # pointwise (now with learnable γ)
    "NELU", "nelu",
    "NiLU", "nilu",
    # GLU FFN blocks
    "SwiGLU", "NiLUGLU", "NELUGLU",
]
