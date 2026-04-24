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

Gate Normalization decouples the two by standardizing the gate's input
— subtract the mean, divide by the standard deviation — and then adding
a learnable offset `β`, so the operating point is invariant to shifts
and rescalings of `x` *and* can be learned per layer:

    y = x · g(γ · (x - μ(x)) / σ(x) + β).

The gate input is shift- and scale-invariant with respect to `x`. Two
learnable scalars control how the normalized gate participates:

* `γ` (init `0`) sets the sensitivity of the gate to the normalized
  input. At `γ = 0` the gate is flat and the module is linear; training
  grows `|γ|` as needed.
* `β` (init `0`) shifts the operating point. At `β = 0` the gate sits at
  `t = 0`, matching the identity-at-init behavior of GELU/SiLU; a
  non-zero `β` lets the optimizer move the gate's decision boundary to
  wherever the upstream distribution actually lives.

Centering also neutralizes a bias added upstream: a constant DC offset
in `x` (e.g. from the preceding `Linear.bias`) cancels in `x - μ(x)`,
so the gate's decision boundary is decoupled from that degree of
freedom. What remains of the DC component is picked up by the learnable
`β`.

## Instances

Two concrete instances are studied in the paper:

* **NELU** — gate is the Gaussian CDF `Φ(·)`; the baseline is GELU.
* **NiLU** — gate is the sigmoid `σ(·)`; the baseline is SiLU.

Both keep the same parameter count as their baseline (the only new
parameter is the scalar `γ`). The drop-in replacement is exact: at
`γ = 0`, `y = x · g(0) = x · c` for the constant `c = g(0)`, and the
module reduces to a linear rescaling that the optimizer can absorb into
subsequent weights.

## Reduction axes

`μ(x)` and `σ(x)` are computed over axes that match the *mixing axes of
the preceding linear operation*, so the gate sees the same statistical
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

## Initialization

`γ` and `β` are both initialized to exactly `0`. At zero both scalars
the gate is flat at ``g(0)``, so the module is ``y = x · g(0)`` at
initialization — a linear rescaling that the optimizer can absorb into
surrounding weights. Training grows `|γ|` as the gate becomes useful
and shifts `β` to place the gate's decision boundary where the data
actually lives.

## GLU variants

For GLU-family feed-forward blocks (SwiGLU and relatives), Gate
Normalization is applied *only to the gate branch*:

    y = W_down( gate · g(γ · (gate - μ(gate)) / σ(gate) + β) ⊙ up )

where `gate = W_gate(x)` and `up = W_up(x)`. The up-projection is left
intact. Parameter count is unchanged relative to SwiGLU. The
`NELUGLU` / `NiLUGLU` modules in `gate_norm.glu` implement this form.
