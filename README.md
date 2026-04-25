# Gate Normalization

*Scale-Invariant Self-Gated Activations.*

Gate Normalization replaces the standard self-gated activation
`x · g(x)` with an RMS-normalized variant

    y = x · g(γ · x / rms(x)),    rms(x) = sqrt(mean(x²) + eps),

where `g` is a pointwise squashing function (Gaussian CDF or sigmoid)
and `γ` is a non-learnable scalar driven by a warmup scheduler that
ramps γ from `0` to `1` over the LR warmup horizon. At step 0 the
activation is `y = 0.5 · x`, an exact linear identity that the
optimizer can absorb; by the end of warmup the activation has settled
into its production form. Applied to GELU and SiLU this yields two
drop-in replacements we call **NELU** and **NiLU**.

| Instance | Base activation | Gate function      |
|----------|-----------------|--------------------|
| `NELU`   | GELU            | `Φ(·)` (Gaussian)  |
| `NiLU`   | SiLU            | `σ(·)` (sigmoid)   |

## Install

```bash
git clone https://github.com/anonymous/gate-normalization.git
cd gate-normalization
pip install -e '.[train]'
```

Python ≥ 3.10 and PyTorch ≥ 2.1. The pure-PyTorch implementation runs
on CPU and CUDA out of the box; `torch.compile` (Inductor) fuses the
forward pass into a single reduction kernel.

## Usage

Gate Normalization layers are plain `nn.Module`s and can be constructed in
place of `nn.GELU()` / `nn.SiLU()`:

```python
import torch.nn as nn
from gate_norm import NELU, NiLU

# Drop-in for channels-last or transformer inputs (default).
act = NELU()

# For NCHW conv feature maps, reduce over (C, H, W).
act_conv = NiLU(norm_axes="sample")
```

For an existing timm / torchvision / HuggingFace model, the
`train.swap` helper replaces every instance on the module tree:

```python
import timm
from train.swap import apply_gate_normalization

model = timm.create_model("convnext_tiny", pretrained=False)
apply_gate_normalization(model, "nelu")  # swaps every GELU -> NELU
```

## Reproducing the paper

Eight ImageNet-1k recipes and one CIFAR-100 recipe are provided under
`configs/`. Each ImageNet config is ported verbatim from a reproduced
reference (MMPretrain for ConvNeXt / DeiT / Swin; timm's
`training_script.mdx` for EfficientNet) — see the header comment in each
file for the source URL and the reproduced Top-1 number.

### Single run (locally)

```bash
torchrun --nproc_per_node=8 -m train.imagenet \
    --config configs/imagenet/convnext_tiny.yaml \
    --activation nelu \
    --data-dir /data/imagenet \
    --output ./runs/convnext_tiny-nelu
```

Set `--activation gelu` (the default) to train the baseline, or `silu` /
`nilu` for the SiLU/NiLU pair.

### Single run (SkyPilot)

```bash
sky jobs launch -n convnext-tiny-nelu sky/train.yaml \
    --env CONFIG=configs/imagenet/convnext_tiny.yaml \
    --env ACTIVATION=nelu
```

Managed jobs recover automatically from spot preemption; see
[`sky/README.md`](sky/README.md).

### CIFAR-100

```bash
python -m train.cifar --config configs/cifar100.yaml --activation nelu
```

### Robustness evaluation

Once a checkpoint is trained, evaluate it on the standard robustness
benchmarks. `bash scripts/prepare_data.sh /data` fetches every dataset
except ImageNet-1k itself (which has to be downloaded manually from
image-net.org).

```bash
# ImageNet-C / A / R / O for an ImageNet checkpoint
python -m eval.imagenet_robustness \
    --model convnext_tiny --activation nelu \
    --checkpoint runs/convnext_tiny-nelu/last.pth.tar \
    --data-root /data

# CIFAR-100-C for a CIFAR-100 checkpoint
python -m eval.cifar_robustness \
    --model resnet20 --activation nelu \
    --checkpoint runs/resnet20-nelu/best.pt \
    --data-root /data
```

## Repository layout

```
gate_norm/        Library: GateNorm, NELU, NiLU, fused CUDA backend.
train/            Trainers: imagenet.py (timm-based), cifar.py.
eval/             Robustness eval: imagenet_robustness.py, cifar_robustness.py.
configs/          Recipe YAMLs, one per model.
sky/              SkyPilot task specs.
scripts/          Helper scripts (prepare_data.sh).
tests/            Unit tests.
docs/             Method derivation and reproduction notes.
```

## Citation

```bibtex
@article{gate_normalization_2026,
  title   = {Gate Normalization: Scale-Invariant Self-Gated Activations},
  author  = {Anonymous},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

Apache-2.0. See [`LICENSE`](LICENSE).
