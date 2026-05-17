""" NELU / NiLU drop-in activation pipelines for the GEM benchmark.

These reuse GEM's exact cross-entropy training pipeline (HPO via ASHA,
SGD + CosineAnnealingLR, balanced-accuracy eval, ciFAIR splits) and
change ONLY the activation function inside the backbone GEM builds.
This is the reviewer-defensible setup: every method (ReLU baseline =
``xent``, GELU/SiLU, NELU, NiLU) goes through the identical GEM
protocol; the sole variable is the activation.

The NELU/NiLU implementation and the audited ReLU->gate swap come from
the NELU research repo (``gate_norm`` + ``train.swap``); GEM must run
with that repo on PYTHONPATH (set by the launcher / orchestrator).

Architectures GEM builds for our pilot:
  * ``rn20`` -> torchvision BasicBlock-based CIFAR ResNet (rn_cifar.py);
    only the stem uses functional F.relu (1 site, left as ReLU — a
    single shallow layer, immaterial to the mechanism); all block
    activations are nn.ReLU modules and get swapped.
  * ``wrn-16-8`` -> nn.ReLU modules throughout (wrn_cifar.py) — fully
    swapped.

torchvision BasicBlock/Bottleneck reuse one ``self.relu`` at multiple
call sites with differing channel counts, which breaks per-channel
NELU gamma. We reuse the audited fix from the NELU repo
(``train.medmnist._split_block_relus`` + lazy gamma re-materialisation
via a dummy forward) so each call site gets a correctly-sized gate.
"""

import torch
from torch import nn

from gem.pipelines.xent import CrossEntropyClassifier


def _swap_to_gate(model: nn.Module, kind: str) -> nn.Module:
    """Swap every ReLU module in ``model`` to NELU or NiLU.

    Imports are deferred so GEM can import this module even when the
    NELU repo is absent (e.g. listing available pipelines); they only
    fire when a NELU/NiLU run is actually built.
    """
    from gate_norm import NELU, NiLU
    from train.swap import replace_activation_auto_axes
    from train.medmnist import _split_block_relus

    gate_cls = NELU if kind == "nelu" else NiLU

    # Give each torchvision residual block independent per-site ReLUs so
    # per-channel gamma is correctly sized (no-op for blocks GEM builds
    # that already use distinct modules; safe for BasicBlock/Bottleneck).
    _split_block_relus(model)

    n = replace_activation_auto_axes(
        model, (nn.ReLU,), gate_cls,
        activation_order="post", gamma_init=1.0,
    )

    # swap.py sizes gamma from the declaration-order adjacent conv, which
    # is wrong for torchvision Bottleneck (post-conv1 ReLU declared after
    # conv3). Reset to lazy and let one dummy forward materialise each
    # gate to its true input-channel count. 32x32 = CIFAR/ciFAIR input.
    for mod in model.modules():
        if type(mod).__name__ in ("NELU", "NiLU"):
            mod.gamma = nn.UninitializedParameter()
            mod.beta = nn.UninitializedParameter()
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(2, 3, 32, 32))
    model.train(was_training)
    print(f"[{kind}] swapped {n} ReLU -> {kind.upper()}")
    return model


class NELUClassifier(CrossEntropyClassifier):
    """Cross-entropy training with the activation swapped to NELU."""

    def create_model(self, arch, num_classes, input_channels, config={}):
        model = super().create_model(arch, num_classes, input_channels, config)
        return _swap_to_gate(model, "nelu")

    @staticmethod
    def get_pipe_name():
        return "nelu"


class NiLUClassifier(CrossEntropyClassifier):
    """Cross-entropy training with the activation swapped to NiLU."""

    def create_model(self, arch, num_classes, input_channels, config={}):
        model = super().create_model(arch, num_classes, input_channels, config)
        return _swap_to_gate(model, "nilu")

    @staticmethod
    def get_pipe_name():
        return "nilu"
