"""Sanity + gradcheck for NELU/NiLU CUDA kernels (per-channel gamma).

Run on the A100/H100 box:
    cd ~/ResAct && python tests/test_nelu_cuda.py

Checks:
  1. CUDA forward matches Python reference (fp32, fp16, bf16).
  2. CUDA dz + dgamma match autograd through the Python reference.
  3. gradcheck (fp64) on a small tensor routes through the scalar
     fallback path and verifies the analytic backward.
  4. Shape sweep — 2D/3D, small/mid/large N, N%{2,4} aligned + unaligned.
"""

import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_cuda_ops():
    from nelu.activations import _nelu_py, _nilu_py  # python reference
    from nelu.cuda_kernel import nelu_cuda
    from nelu.nilu_cuda_kernel import nilu_cuda

    return _nelu_py, _nilu_py, nelu_cuda, nilu_cuda


def _allclose(a, b, rtol, atol, name):
    ok = torch.allclose(a, b, rtol=rtol, atol=atol)
    diff = (a - b).abs().max().item()
    print(f"  {'OK' if ok else 'FAIL'} {name}: max|diff|={diff:.3e}")
    return ok


def run_fwd_bwd_check(op_name, cuda_fn, py_fn, shape, dtype):
    torch.manual_seed(0)
    C = shape[-1]
    z = torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)
    gamma_val = 0.3
    gamma = torch.full((C,), gamma_val, device="cuda",
                       dtype=torch.float32, requires_grad=True)

    # Reference (Python autograd).
    z_ref = z.detach().clone().requires_grad_(True)
    g_ref = gamma.detach().clone().requires_grad_(True)
    y_ref = py_fn(z_ref, g_ref, 1e-6)
    loss_ref = y_ref.float().pow(2).mean()
    loss_ref.backward()

    # CUDA op.
    z_cu = z.detach().clone().requires_grad_(True)
    g_cu = gamma.detach().clone().requires_grad_(True)
    y_cu = cuda_fn(z_cu, g_cu, 1e-6)
    loss_cu = y_cu.float().pow(2).mean()
    loss_cu.backward()

    rtol = {torch.float32: 1e-4, torch.float16: 5e-2, torch.bfloat16: 5e-2}[dtype]
    atol = {torch.float32: 1e-5, torch.float16: 5e-3, torch.bfloat16: 5e-3}[dtype]

    tag = f"{op_name} {tuple(shape)} {dtype}"
    print(f"\n[{tag}]")
    ok = True
    ok &= _allclose(y_cu.float(), y_ref.float(), rtol, atol, "y")
    ok &= _allclose(z_cu.grad.float(), z_ref.grad.float(), rtol, atol, "dz")
    ok &= _allclose(g_cu.grad.float(), g_ref.grad.float(), 1e-3, 1e-4, "dgamma")
    return ok


def run_shape_sweep(nelu_cuda, nilu_cuda, _nelu_py, _nilu_py):
    shapes = [
        (8, 32),       # warp path
        (8, 64),       # small
        (8, 192),      # DeiT small hidden / unaligned? 192%4=0
        (8, 384),
        (8, 768),      # DeiT-B hidden (last dim)
        (4, 197, 768), # ViT-B seq
        (4, 196, 768),
        (2, 3072),     # GLU hidden
        (4, 96),       # ConvNeXt-T dim
        (4, 384),      # ConvNeXt-T stage-3 dim
        (2, 55),       # unaligned N
    ]
    all_ok = True
    for shape in shapes:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            all_ok &= run_fwd_bwd_check("NELU", nelu_cuda, _nelu_py, shape, dtype)
            all_ok &= run_fwd_bwd_check("NiLU", nilu_cuda, _nilu_py, shape, dtype)
    return all_ok


def run_gradcheck(nelu_cuda, nilu_cuda):
    print("\n[gradcheck fp64 — small tensor]")
    # gradcheck uses double; our kernel dispatches to scalar fallback.
    torch.manual_seed(1)
    z = torch.randn(3, 16, device="cuda", dtype=torch.float64, requires_grad=True)
    g = torch.randn(16, device="cuda", dtype=torch.float32, requires_grad=True) * 0.1 + 0.2

    def f_nelu(z_, g_):
        return nelu_cuda(z_, g_, 1e-6)

    ok1 = torch.autograd.gradcheck(f_nelu, (z, g), eps=1e-4, atol=1e-3, rtol=1e-2,
                                   fast_mode=True)
    print(f"  {'OK' if ok1 else 'FAIL'} NELU gradcheck")

    def f_nilu(z_, g_):
        return nilu_cuda(z_, g_, 1e-6)

    ok2 = torch.autograd.gradcheck(f_nilu, (z, g), eps=1e-4, atol=1e-3, rtol=1e-2,
                                   fast_mode=True)
    print(f"  {'OK' if ok2 else 'FAIL'} NiLU gradcheck")
    return ok1 and ok2


def test_cuda_kernels_end_to_end():
    import pytest

    if not torch.cuda.is_available() or not os.environ.get("CUDA_HOME"):
        pytest.skip("CUDA test requires a CUDA runtime with CUDA_HOME set")

    _nelu_py, _nilu_py, nelu_cuda, nilu_cuda = _load_cuda_ops()
    assert run_shape_sweep(nelu_cuda, nilu_cuda, _nelu_py, _nilu_py)
    assert run_gradcheck(nelu_cuda, nilu_cuda)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs CUDA"
    assert os.environ.get("CUDA_HOME"), "CUDA_HOME must be set"
    _nelu_py, _nilu_py, nelu_cuda, nilu_cuda = _load_cuda_ops()
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__}")

    ok_sweep = run_shape_sweep(nelu_cuda, nilu_cuda, _nelu_py, _nilu_py)
    try:
        ok_grad = run_gradcheck(nelu_cuda, nilu_cuda)
    except Exception as e:
        print(f"gradcheck error: {e}")
        ok_grad = False

    print("\n" + "=" * 50)
    print(f"shape sweep : {'PASS' if ok_sweep else 'FAIL'}")
    print(f"gradcheck   : {'PASS' if ok_grad else 'FAIL'}")
    sys.exit(0 if (ok_sweep and ok_grad) else 1)
