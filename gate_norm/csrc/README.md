# CUDA kernels (currently dormant)

The kernels in this directory implement the centered-and-learnable variant
`y = x * g(γ * (x - μ) / σ + β)` from `gate_norm` v0.2 / v0.3. The current
package (`v0.4+`) computes `y = x * g(γ * x / rms(x))` with γ as a
non-learnable buffer driven by `gate_norm.GammaWarmup`, and the fused kernel
has not yet been ported to that simpler form.

`gate_norm.core.GateNorm._CUDA_OP` is `None` for every shipped activation,
so the dispatch in `gate_norm.dispatch` always returns `False` and the
PyTorch path is what runs. Inductor fuses that path well enough that we
have not measured a meaningful gap on production workloads.

Re-enable: rewrite `gate_norm.cu` (forward + backward) for the simpler RMS
form and set `_CUDA_OP = "nelu" | "nilu"` on the subclasses.

The git tag `v0.2-centered-learnable` preserves the last revision in which
this directory was active.
