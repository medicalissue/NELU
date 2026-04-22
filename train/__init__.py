"""Training entry points and utilities for the Gate Normalization paper.

Modules
-------
:mod:`train.imagenet`
    timm-based ImageNet-1k trainer.
:mod:`train.cifar`
    CIFAR-100 trainer.
:mod:`train.swap`
    Recursively replace :class:`torch.nn.GELU` / :class:`torch.nn.SiLU` with
    the Gate Normalization instances :class:`gate_norm.NELU` /
    :class:`gate_norm.NiLU`.
:mod:`train.diagnostics`
    Gate entropy, gate variance, and weight-norm probes.
"""
