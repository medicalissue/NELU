"""RMS-gate-normalized activations and GLU blocks.

  Pointwise:
    NELU    = GELU + RMS gate normalization, γ scheduled (non-learnable)
    NiLU    = SiLU + RMS gate normalization, γ scheduled (non-learnable)

  GLU FFN blocks:
    SwiGLU  = baseline (LLaMA-style)
    NiLUGLU = SwiGLU with NiLU on the gate projection
    NELUGLU = SwiGLU with NELU on the gate projection

γ is a scalar non-persistent buffer updated each epoch by the training
loop via `set_gamma_all(model, gamma_schedule(...))`. Zero learnable
parameters added to the model (γ is NOT in state_dict).

See nelu/activations.py docstring for the CIFAR ablation that led to
this design (fixed γ with cosine warmup ≥ learnable γ, and
per-channel learnable collapses).
"""

from .activations import (
    NELU, NiLU, nelu, nilu,
    set_gamma_all, gamma_schedule, current_gamma,
)
from .glu import SwiGLU, NiLUGLU, NELUGLU

__all__ = [
    # pointwise
    "NELU", "nelu",
    "NiLU", "nilu",
    # training-loop helpers (scheduled gamma curriculum)
    "set_gamma_all", "gamma_schedule", "current_gamma",
    # GLU FFN blocks
    "SwiGLU", "NiLUGLU", "NELUGLU",
]
