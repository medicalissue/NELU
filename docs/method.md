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
single non-learnable scalar shared per module. We do *not* subtract the
mean and we do *not* add a learnable bias `β`; both were tried in
earlier versions of the work (`v0.2-centered-learnable`) and gave
mixed results — see the appendix ablation.

`γ` is driven externally by a warmup scheduler
(:class:`gate_norm.GammaWarmup`) that ramps `γ` from `0` at step 0 to
`1` over the same number of optimizer steps that the LR warmup uses,
and then holds `γ = 1` for the rest of training. The ramp serves two
purposes:

* **At step 0 the activation is linear.** With `γ = 0` we get
  `y = x · g(0) = x · c` for the constant `c = g(0) = 0.5`, so every
  shipped activation has an exact identity-up-to-constant init that the
  optimizer can absorb into subsequent weights.
* **By the end of warmup `γ = 1`** and the activation has settled into
  its production form. Larger architectures (ConvNeXt-S, Swin-S, …)
  that were unstable at `γ = 1` from step 0 are stable when γ is
  introduced gradually.

Earlier learnable-`γ` variants developed two pathological attractors
that the buffer-only form sidesteps: small models collapsed `γ → 0`
combined with `β > 0` (gate becomes a constant ≈0.7, the activation
degenerates to a linear scaling); large models pushed `β` strongly
negative, killing roughly a third of the layers via near-zero gate
output. Fixing `γ` removes both.

## Instances

Two concrete instances are studied in the paper:

* **NELU** — gate is the Gaussian CDF `Φ(·)`; the baseline is GELU.
* **NiLU** — gate is the sigmoid `σ(·)`; the baseline is SiLU.

Both keep the *same parameter count as their baseline* — `γ` is a
non-learnable buffer, not a parameter. The drop-in replacement is
exact: at `γ = 0`, `y = x · g(0) = x · c` for the constant
`c = g(0) = 0.5`, and the module reduces to a linear rescaling that
the optimizer can absorb into subsequent weights.

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

## Initialization and scheduling

`γ` is initialized to `0` and ramped to `1` by
:class:`gate_norm.GammaWarmup` over the same number of optimizer steps
as the LR warmup. At step 0 the gate is flat at `g(0) = 0.5`, so the
module is `y = 0.5 · x` — a linear rescaling that the optimizer can
absorb into surrounding weights. After warmup `γ` is held at `1` and
the activation behaves as `y = x · g(x / rms(x))`.

The schedule is "linear" by default (matches `LinearLR`); "cosine" and
"constant" variants are available for ablation.

## GLU variants

For GLU-family feed-forward blocks (SwiGLU and relatives), Gate
Normalization is applied *only to the gate branch*:

    y = W_down( gate · g(γ · gate / rms(gate)) ⊙ up )

where `gate = W_gate(x)` and `up = W_up(x)`. The up-projection is left
intact. Parameter count is unchanged relative to SwiGLU. The
`NELUGLU` / `NiLUGLU` modules in `gate_norm.glu` implement this form.
