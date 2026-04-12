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

from .activations import (NELU, NiLU, NELU_SG, NiLU_SG,
                          NELU_Beta, NiLU_Beta,
                          NELU_Gamma, NiLU_Gamma,
                          NELU_Surr, NiLU_Surr, nelu, nilu)
from .glu import SwiGLU, NiLUGLU, NELUGLU

# NoSG CUDA kernels (backward has cross-term reduction)
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

# SG CUDA kernels (backward is purely element-wise, no cross-term)
try:
    from .cuda_kernel_sg import NELUCUDA_SG, nelu_cuda_sg
except Exception:
    NELUCUDA_SG = None
    nelu_cuda_sg = None

try:
    from .cuda_kernel_surr import NELUCUDA_Surr, nelu_cuda_surr
except Exception:
    NELUCUDA_Surr = None
    nelu_cuda_surr = None

try:
    from .nilu_cuda_kernel_sg import NiLUCUDA_SG, nilu_cuda_sg
except Exception:
    NiLUCUDA_SG = None
    nilu_cuda_sg = None

__all__ = [
    # pointwise (NoSG)
    "NELU", "nelu", "NiLU", "nilu",
    # pointwise (SG)
    "NELU_SG", "NiLU_SG",
    # GLU FFN blocks
    "SwiGLU", "NiLUGLU", "NELUGLU",
    # Fused CUDA — NoSG
    "NELUCUDA", "nelu_cuda",
    "NiLUCUDA", "nilu_cuda",
    # Fused CUDA — SG
    "NELUCUDA_SG", "nelu_cuda_sg",
    "NiLUCUDA_SG", "nilu_cuda_sg",
]
