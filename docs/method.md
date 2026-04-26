# Gate Normalization

This note derives Gate Normalization and fixes notation used throughout the
code base.

## Motivation

Self-gated activations — SiLU (`x·σ(x)`), GELU (`x·Φ(x)`), and the SwiGLU
family — share a common structure: a pointwise squashing function applied
to the same input that is being gated. Because the gate is a function of
`x` directly, its operating point depends on the magnitude of `x`. When
`x` saturates, every unit is either fully on or fully off; when `x`
shrinks toward zero, every unit behaves linearly.

The magnitude of `x` is itself controlled by whatever normalization sits
upstream (LayerNorm, BatchNorm, RMSNorm) and by the learned weights
feeding into the activation. Tying the gate's operating point to that
magnitude adds an implicit coupling between normalization and
nonlinearity that the optimizer must manage.

Gate Normalization decouples the two by rescaling the gate's input by
its root-mean-square, so the operating point is invariant to rescalings
of `x`:

    y = x · g(γ · x / rms(x)),    rms(x) = sqrt(mean(x²) + eps).

The gate input is scale-invariant with respect to `x`, and `γ` is a
single learnable scalar shared per module. We do *not* subtract the mean
and we do *not* add a learnable bias `β`: the outer multiplication by
the un-normalized `x` preserves the DC component, which is what keeps
the activation's "save positives, drop negatives" inductive bias intact.

## Learning γ

`γ` is a single learnable scalar per module, initialised to `γ_init`
(default 1) and driven by the optimizer alongside the rest of the
model. No reparameterization or positivity constraint is applied:
gradient flow naturally keeps `γ` positive because flipping `γ`
negative inverts the gate (`Φ(γ x̂)` becomes `1 − Φ(|γ| x̂)`) and
exchanges positive- and negative-half saturation, which yields no
useful loss signal. We empirically observe `γ` to stay in a narrow
positive range across all architectures we evaluate.

## Instances

Two concrete instances are studied in the paper:

* **NELU** — gate is the Gaussian CDF `Φ(·)`; the baseline is GELU.
* **NiLU** — gate is the sigmoid `σ(·)`; the baseline is SiLU.

Both keep the same parameter count as their baseline up to a single
shared scalar `γ_raw`. Both are exact drop-in replacements for
`nn.GELU()` / `nn.SiLU()`.

## Reduction axes

`rms(x)` is computed over axes that match the *mixing axes of the
preceding linear operation*, so the gate sees the same statistical
granularity that the upstream linear created:

* `"channel"` — trailing feature axis only. Matches channel-mixing
  linear ops (transformer FFN `Linear`, ConvNeXt pointwise, any
  channels-last activation at `(B, D)` / `(B, L, D)` / `(B, H, W, D)`).
* `"sample"`  — `(C, H, W)` for NCHW tensors. Matches blocks whose
  preceding linear mixes both channel and space (fused EfficientNet
  blocks, CIFAR-style BN-ReLU ResNets).

For depthwise or Squeeze-Excite blocks where neither alias is right,
pass the axis tuple directly (e.g. `(2, 3)` for spatial-only mixing
after a depthwise conv). `train.swap.apply_gate_normalization` does
this automatically for timm's EfficientNet MBConv / DepthwiseSeparable
/ EdgeResidual blocks — see `_MBCONV_POLICY` in `train/swap.py` for
the exact mapping per sub-attribute.

## GLU variants

For GLU-family feed-forward blocks (SwiGLU and relatives), Gate
Normalization is applied *only to the gate branch*:

    y = W_down( gate · g(γ · gate / rms(gate)) ⊙ up )

where `gate = W_gate(x)` and `up = W_up(x)`. The up-projection is left
intact. Parameter count matches SwiGLU up to the single `γ_raw` scalar.
The `NELUGLU` / `NiLUGLU` modules in `gate_norm.glu` implement this form.
