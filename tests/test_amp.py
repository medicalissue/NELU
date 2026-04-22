"""AMP / mixed-precision behavior tests.

These guard the design decision that the statistics path (RMS, γ·z/ρ and
the gate function) always runs in float32, while the outer multiplication
by ``z`` preserves the caller's activation dtype so the autocast contract
is not broken.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gate_norm import NELU, NiLU, nelu, nilu


@pytest.mark.parametrize("act_cls", [NELU, NiLU])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gate_norm_preserves_input_dtype(act_cls, dtype) -> None:
    x = torch.randn(4, 16).to(dtype)
    act = act_cls(gamma_init=0.1)
    y = act(x)
    assert y.dtype == dtype, f"{act_cls.__name__}({dtype}) produced {y.dtype}"
    assert y.shape == x.shape


@pytest.mark.parametrize("fn", [nelu, nilu])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_functional_preserves_input_dtype(fn, dtype) -> None:
    x = torch.randn(4, 16).to(dtype)
    y = fn(x, gamma=0.1)
    assert y.dtype == dtype
    assert y.shape == x.shape


def test_rms_does_not_underflow_in_fp16() -> None:
    """With small-magnitude inputs, the fp32 internal path should still
    produce finite γ·z/ρ. If RMS were computed in fp16, small z² would
    underflow to zero and 1/ρ would be ±inf."""
    small = torch.full((1, 128), 1e-3, dtype=torch.float16)
    act = NELU(gamma_init=1e-3)
    y = act(small)
    assert torch.isfinite(y).all()


def test_matches_fp32_reference_within_tolerance() -> None:
    """Half-precision forward should agree with the fp32 reference up to
    dtype tolerance."""
    torch.manual_seed(0)
    x = torch.randn(8, 64)
    x_half = x.to(torch.float16)

    act = NELU(gamma_init=0.5)
    ref = act(x)
    out = act(x_half).float()
    assert torch.allclose(out, ref, atol=2e-3, rtol=2e-3)


def test_backward_still_runs_under_autocast() -> None:
    if not torch.cuda.is_available():
        pytest.skip("autocast-on-CPU bfloat16 path lacks a sqrt kernel")
    x = torch.randn(4, 16, requires_grad=True, device="cuda")
    act = NELU().to("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = act(x)
    y.sum().backward()
    assert act.gamma.grad is not None
    assert torch.isfinite(act.gamma.grad).all()
