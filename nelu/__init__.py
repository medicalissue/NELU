"""RMS-gate-normalized activations and GLU blocks.

  Pointwise:
    NELU    = GELU + RMS gate normalization + learnable per-layer γ
    NiLU    = SiLU + RMS gate normalization + learnable per-layer γ

  GLU FFN blocks:
    SwiGLU  = baseline (LLaMA-style)
    NiLUGLU = SwiGLU with NiLU on the gate projection
    NELUGLU = SwiGLU with NELU on the gate projection

γ is a single learnable nn.Parameter scalar per NELU/NiLU module
(init 1e-4). Per-layer, not per-channel. See nelu/activations.py
docstring for the ablation evidence behind this choice.
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
