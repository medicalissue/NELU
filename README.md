# NELU & NiLU: Gate-Normalized Activations

Gate normalization is a one-line modification to standard activations (GELU, SiLU) that makes the gating signal scale-invariant. Instead of `z * Phi(z)`, we compute `z * Phi(gamma * z / rms(z))`, where `rms(z)` is the channel-wise root mean square and `gamma` is a single learnable scalar per layer, initialized near zero. This makes the gate respond to the *direction* of activations rather than their magnitude, eliminating the implicit coupling between scale and gating that causes training instabilities in deep networks.

- **NELU** replaces GELU: `f(z) = z * Phi(gamma * z / rms(z))`
- **NiLU** replaces SiLU: `f(z) = z * sigma(gamma * z / rms(z))`

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/NELU.git
cd NELU
pip install torch timm pyyaml  # requires PyTorch >= 2.0

# Optional: build the fused CUDA kernel for ~2x throughput
python -c "from nelu.cuda_kernel import nelu_cuda"
```

## Quick start

Drop-in replacement for any model that uses GELU or SiLU:

```python
import torch.nn as nn
from nelu import NELU, NiLU

# Option 1: Direct use
act = NELU()  # or NiLU()
y = act(x)    # works with any shape (..., channels)

# Option 2: Swap activations in an existing model
from train.act_swap import swap_gelu_to_nelu, swap_silu_to_nilu
import timm

model = timm.create_model("convnext_tiny", pretrained=False)
n = swap_gelu_to_nelu(model)
print(f"Replaced {n} GELU -> NELU")

# Option 3: GLU variants for transformers
from nelu import NELUGLU, NiLUGLU
ffn = NELUGLU(dim=512, hidden_dim=1024)
```

## Reproducing experiments

### Setup

```bash
# Create environment
bash scripts/setup_env.sh

# Copy .env.example to .env and fill in your values first.
# Then prepare the datasets used by this repo:
#   imagenet, cifar-100-python, CIFAR-100-C,
#   ImageNet-C, imagenet-a, imagenet-r, imagenet-o
bash scripts/download_data.sh /data
```

### Single experiment

```bash
# Run one experiment
bash scripts/run_single.sh imagenet convnext_tiny nelu

# With custom gamma init
bash scripts/run_single.sh ablation convnext_tiny nelu --gamma_init 0.01
```

### Full reproduction (3 nodes)

The complete experiment suite is split across three nodes for parallel execution:

```bash
# Node 1: ViT-L x2, ViT-B GELU (~205 GPU-hours)
bash scripts/run_all.sh scripts/jobs_node1.txt

# Node 2: ConvNeXt-B/S x2, EfficientNet-B4 x2 (~153 GPU-hours)
bash scripts/run_all.sh scripts/jobs_node2.txt

# Node 3: ConvNeXt-T x2, EfficientNet-B0/B2 x2, ViT-B NELU, ablation, CIFAR-100 (~128 GPU-hours)
bash scripts/run_all.sh scripts/jobs_node3.txt
```

### AWS spot instances

```bash
# Launch 3 spot instances with the job queue
bash scripts/infra/launch_spot.sh 3 scripts/
```

Notes:
- Copy [.env.example](/Users/medicalissue/Desktop/NELU/.env.example) to `.env` and fill it in. The launcher, monitor, snapshot setup, and dataset prep scripts read `.env` automatically.
- Keep `.env` local only. `launch_spot.sh` excludes it from the uploaded code tarball.
- `launch_spot.sh` uploads the current local repo snapshot to `s3://.../code/nelu-code.tar.gz`, and instance user-data boots from that tarball.
- If `AMI` is unset, the launcher resolves the latest AWS Deep Learning Base GPU Ubuntu 22.04 AMI from the public SSM parameter store. Override with `AMI` or `AMI_SSM_PARAM` if needed.

## Repository structure

```
nelu/               # Activation library (NELU, NiLU, GLU variants, CUDA kernels)
train/              # Training utilities (activation swap, gamma logging, spot resilience)
configs/            # YAML configs for all experiments
scripts/            # Shell scripts for running experiments and infrastructure
patches/            # Patches for upstream training repos (ConvNeXt, DeiT)
tests/              # Unit tests
```

## Citation

```bibtex
@inproceedings{nelu2026,
  title     = {Gate Normalization: Scale-Invariant Self-Gated Activations},
  author    = {Anonymous},
  booktitle = {NeurIPS},
  year      = {2026}
}
```
