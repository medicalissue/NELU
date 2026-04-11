# External-repo patches

For bit-exact reproduction of the published timm/FB checkpoints we
train with the **upstream training scripts** and only swap activations.
That means we need to clone two upstream repos and apply minimal
patches:

| Patch                  | Upstream                                                              | Pinned commit                             |
| ---------------------- | --------------------------------------------------------------------- | ----------------------------------------- |
| `deit-train.patch`     | [facebookresearch/deit](https://github.com/facebookresearch/deit)      | `7e160fe43f0252d17191b71cbb5826254114ea5b` |
| `convnext-train.patch` | [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt) | `048efcea897d999aed302f2639b6270aedf8d4c8` |

What each patch does:

**`deit-train.patch`** (`main.py`, `augment.py`)
- adds `--act {gelu,silu,nelu,nilu}` argparse option
- recursively swaps `nn.GELU` for `NELU`/`NiLU` after `create_model`,
  before EMA / DDP / finetune-load
- adds `--torch-compile` flag (wraps model after DDP wrap, dynamo
  errors suppressed)
- alias `_pil_interp -> str_to_pil_interp` for newer timm compatibility

**`convnext-train.patch`** (`main.py`, `optim_factory.py`, `utils.py`,
`models/convnext.py`)
- adds `--act` and `--torch_compile` argparse options
- swaps GELU after `create_model`
- compatibility shims for newer timm/PyTorch:
  - `Nadam → NAdamLegacy`, `RAdam → RAdamLegacy`, `NovoGrad → optional`
  - `torch._six.inf → math.inf`
  - `ConvNeXt.__init__` accepts `**kwargs` so timm's `pretrained_cfg`
    doesn't break it

To apply, clone the upstream repos at the pinned commits and run:

    git -C /path/to/deit-train apply /path/to/ResAct/patches/deit-train.patch
    git -C /path/to/convnext-train apply /path/to/ResAct/patches/convnext-train.patch

Or just run `bash scripts/setup_h100.sh` from ResAct, which does all
of the above plus pip deps + cache clear + smoke test.
