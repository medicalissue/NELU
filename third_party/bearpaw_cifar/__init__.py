"""Minimal subset of bearpaw/pytorch-classification vendored for CIFAR-100.

chenyaofo's hub covers ResNet-20/32/44/56, VGG, MobileNetV2, ShuffleNetV2,
and a few transformer variants. The two gaps we fill here:

* :mod:`resnet`    — He-2015 CIFAR ResNet at depth 110 (chenyaofo only ships
                     depths ≤ 56).
* :mod:`densenet`  — DenseNet-BC (L=100, k=12).

See ``__about__.py`` for provenance.
"""

from . import densenet, resnet
from .__about__ import COMMIT_SHA, UPSTREAM

__all__ = ["densenet", "resnet", "COMMIT_SHA", "UPSTREAM"]
