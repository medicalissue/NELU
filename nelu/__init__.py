"""NELU: Normalized Gaussian Error Linear Unit."""

from .activations import NELU, nelu

try:
    from .cuda_kernel import NELUCUDA, nelu_cuda
except Exception:
    NELUCUDA = None
    nelu_cuda = None

__all__ = ["NELU", "nelu", "NELUCUDA", "nelu_cuda"]
