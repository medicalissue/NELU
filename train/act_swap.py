"""Recursively swap activation modules in any model.

The key operation for NELU/NiLU experiments: take a pretrained or
randomly-initialized model that uses nn.GELU or nn.SiLU and replace
every instance with the gate-normalized variant (NELU or NiLU).

This works on any model — timm, torchvision, HuggingFace — because it
operates on the module tree, not on specific attribute names.
"""

import torch.nn as nn

from nelu.activations import NELU, NiLU


def replace_activation(model: nn.Module, src_cls: type, target_cls: type,
                       **target_kwargs) -> int:
    """Recursively replace all instances of `src_cls` with `target_cls`.

    Args:
        model: The model to modify in-place.
        src_cls: Activation class to replace (e.g. nn.GELU, nn.SiLU).
        target_cls: Replacement class (e.g. NELU, NiLU).
        **target_kwargs: Keyword arguments forwarded to target_cls constructor
            (e.g. gamma_init=0.01, eps=1e-6).

    Returns:
        Number of modules replaced.

    Example::

        from nelu import NELU
        import timm

        model = timm.create_model("convnext_tiny", pretrained=False)
        n = replace_activation(model, nn.GELU, NELU)
        print(f"Replaced {n} GELU -> NELU")
    """
    count = 0
    for name, child in model.named_children():
        if isinstance(child, src_cls):
            setattr(model, name, target_cls(**target_kwargs))
            count += 1
        else:
            count += replace_activation(child, src_cls, target_cls, **target_kwargs)
    return count


def swap_gelu_to_nelu(model: nn.Module, **kwargs) -> int:
    """Convenience: replace all nn.GELU with NELU."""
    return replace_activation(model, nn.GELU, NELU, **kwargs)


def swap_silu_to_nilu(model: nn.Module, **kwargs) -> int:
    """Convenience: replace all nn.SiLU with NiLU."""
    return replace_activation(model, nn.SiLU, NiLU, **kwargs)
