"""RMS-gate-normalized activations and GLU blocks.

  Pointwise:
    NELU    = GELU   + RMS gate normalization
    NiLU    = SiLU   + RMS gate normalization

  GLU FFN blocks:
    SwiGLU  = baseline (LLaMA-style)
    NiLUGLU = SwiGLU with NiLU on the gate projection
    NELUGLU = SwiGLU with NELU on the gate projection

All share the same principle: dividing the gate argument by rms(z)
gives exact forward scale invariance, f(alpha z) = alpha f(z).
"""

from .activations import NELU, NiLU, NELU_GN, NiLU_GN, nelu, nilu
from .glu import SwiGLU, NiLUGLU, NELUGLU

try:
    from .cuda_kernel import NELUCUDA, nelu_cuda
except Exception:
    NELUCUDA = None
    nelu_cuda = None

try:
    from .nilu_cuda_kernel import NiLUCUDA, nilu_cuda
except Exception:
    NiLUCUDA = None
    nilu_cuda = None

__all__ = [
    # pointwise
    "NELU", "nelu",
    "NiLU", "nilu",
    # gate-normalized (non-expansive Jacobian)
    "NELU_GN", "NiLU_GN",
    # GLU FFN blocks
    "SwiGLU", "NiLUGLU", "NELUGLU",
    # Fused CUDA kernels
    "NELUCUDA", "nelu_cuda",
    "NiLUCUDA", "nilu_cuda",
]
