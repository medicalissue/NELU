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

Gate Normalization decouples the two by rescaling the gate's input to a
unit-RMS regime:

    y = x · g(γ · x / rms(x)).

The gate input is now scale-invariant with respect to `x`. The learnable
scalar `γ` is a single parameter per layer that controls how strongly the
rescaled gate participates — at `γ = 0` the gate is saturated open and
the module is linear; as `γ` grows the gate becomes active.

## Instances

Two concrete instances are studied in the paper:

* **NELU** — gate is the Gaussian CDF `Φ(·)`; the baseline is GELU.
* **NiLU** — gate is the sigmoid `σ(·)`; the baseline is SiLU.

Both keep the same parameter count as their baseline (the only new
parameter is the scalar `γ`). The drop-in replacement is exact: at
`γ = 0`, `y = x · g(0) = x · c` for the constant `c = g(0)`, and the
module reduces to a linear rescaling that the optimizer can absorb into
subsequent weights.

## RMS reduction axes

`rms(x)` is computed over the axes that give *one statistic per spatially
local feature vector*:

* For transformer / channels-last inputs, the natural axis is the
  trailing feature axis. We refer to this as ``per_token``.
* For NCHW convolutional feature maps, the natural axes are
  `(C, H, W)` — each sample contributes a single statistic. We refer to
  this as ``per_sample``.

Custom tuples of axes may be passed to `GateNorm(rms_mode=...)` when
architecture-specific behavior is required.

## Initialization

`γ` is initialized to `1e-6`, matching the small-init conventions of
LayerScale. Near-zero init keeps the module close to a linear identity at
initialization, so subsequent layers do not need to absorb a large
magnitude shock.

## GLU variants

For GLU-family feed-forward blocks (SwiGLU and relatives), Gate
Normalization is applied *only to the gate branch*:

    y = W_down( x · g(γ · gate / rms(gate)) ⊙ up )

where `gate = W_gate(x)` and `up = W_up(x)`. The up-projection is left
intact. Parameter count is unchanged relative to SwiGLU. The
`NELUGLU` / `NiLUGLU` modules in `gate_norm.glu` implement this form.
