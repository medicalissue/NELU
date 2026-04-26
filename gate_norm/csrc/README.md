# CUDA kernels — RMS-only Gate Normalization

Implements the v0.4 form

```
rsigma[m] = 1 / sqrt(mean(x[m,:]²) + eps)
y[m,n]    = x[m,n] · g(γ · x[m,n] · rsigma[m])
```

with `g` selected at compile time (`GATE_PHI` for NELU, `GATE_SIGMOID`
for NiLU). γ is a Python float, threaded into the kernel as a constant —
the activation does not produce a γ gradient (the buffer is driven
externally by `gate_norm.GammaWarmup`).

## Layout

* `gate_norm_common.cuh` — warp/block reductions, gate functions,
  vectorized load/store, dynamic-smem cap query.
* `gate_norm.cu` — forward (`fwd_cached` / `fwd_twopass_*`), backward
  (`bwd_cached` / `bwd_twopass_*`), and the launchers that pick between
  the cached and two-pass variants based on the row's smem footprint.

The backward needs only one row-level reduction:

```
S[m]      = Σ_n  dy[m,n] · x[m,n]² · g'(t[m,n])
dx[m,n]  =  dy[m,n] · g(t[m,n])
          + γ · rsigma[m] · ( dy[m,n] · x[m,n] · g'(t[m,n])
                              - x[m,n] · rsigma[m]² / N · S[m] )
```

— simpler than v0.3, which centered the input (μ subtraction) and
needed two row reductions (R1, R2) plus per-feature γ/β atomicAdds.
There are no atomics anywhere in this version because γ is a buffer.

## When the cached path is used

`bwd_cached` stages both `x` and `dy` in shared memory (2·N floats). On
A10G/L4 with ~100 KB max dynamic smem, that's good for `N ≤ 12500`.
Above that, the two-pass variant streams `x` from HBM twice. CIFAR
ResNet's sample-axes reduction (`N = C·H·W`) hits this in the stem
(C=16, 32×32 → N=16384), so the two-pass path is exercised in
production.

## v0.3 archive

The previous "centered + learnable γ/β" kernel can be recovered from the
git tag `v0.2-centered-learnable`.
