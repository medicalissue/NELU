"""CUDA kernel correctness + microbench for gate_norm.

Run this on a GPU worker to verify the fused kernel against the pure-PyTorch
path and measure HBM throughput.

    python -m tests.bench_gate_norm_cuda

Output: a table of (dtype, M, N) → max fwd/bwd abs error, and fwd/bwd
GB/s effective HBM bandwidth. Useful after the variance-algorithm swap or
any kernel edit.
"""

from __future__ import annotations

import os
import time

import torch

os.environ.setdefault("GATE_NORM_FORCE_PYTHON", "0")

from gate_norm import NELU, NiLU  # noqa: E402


def _run_one(kind: str, dtype: torch.dtype, M: int, N: int,
             warmup: int = 5, iters: int = 20):
    """One (dtype, M, N) microbench + numerical check."""
    torch.manual_seed(0)
    device = "cuda"
    cls = NELU if kind == "nelu" else NiLU
    act_kernel = cls(gamma_init=0.3, beta_init=0.2, eps=1e-6).to(device)
    act_python = cls(gamma_init=0.3, beta_init=0.2, eps=1e-6).to(device)
    # copy params so the two paths see identical γ, β
    act_python.gamma.data.copy_(act_kernel.gamma.data)
    act_python.beta.data.copy_(act_kernel.beta.data)

    x = torch.randn(M, N, device=device, dtype=dtype, requires_grad=True)
    xp = x.detach().clone().requires_grad_(True)

    # Kernel path
    y_k = act_kernel(x)
    # Python path — force disable CUDA dispatch
    os.environ["GATE_NORM_FORCE_PYTHON"] = "1"
    y_p = act_python(xp)
    os.environ["GATE_NORM_FORCE_PYTHON"] = "0"

    err_fwd = (y_k.float() - y_p.float()).abs().max().item()

    dy = torch.randn_like(y_k)
    y_k.backward(dy)
    dy_p = dy.detach().clone()
    y_p.backward(dy_p)

    err_dx = (x.grad.float() - xp.grad.float()).abs().max().item()
    err_dg = (act_kernel.gamma.grad - act_python.gamma.grad).abs().max().item()
    err_db = (act_kernel.beta.grad - act_python.beta.grad).abs().max().item()

    # Microbench — forward
    x_bench = torch.randn(M, N, device=device, dtype=dtype)
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = act_kernel(x_bench)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = act_kernel(x_bench)
    torch.cuda.synchronize()
    fwd_sec = (time.perf_counter() - t0) / iters

    # Microbench — backward
    x_bench.requires_grad_(True)
    dy_bench = torch.randn(M, N, device=device, dtype=dtype)
    for _ in range(warmup):
        y = act_kernel(x_bench)
        y.backward(dy_bench)
        x_bench.grad = None
        act_kernel.gamma.grad = None
        act_kernel.beta.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = act_kernel(x_bench)
        y.backward(dy_bench)
        x_bench.grad = None
        act_kernel.gamma.grad = None
        act_kernel.beta.grad = None
    torch.cuda.synchronize()
    bwd_sec = (time.perf_counter() - t0) / iters

    elem_bytes = dtype.itemsize
    fwd_bytes = 2 * M * N * elem_bytes  # read z + write y
    bwd_bytes = 3 * M * N * elem_bytes  # read z + read dy + write dz
    fwd_gbs = fwd_bytes / fwd_sec / 1e9
    bwd_gbs = (fwd_bytes + bwd_bytes) / bwd_sec / 1e9  # fwd is inside bwd loop

    return {
        "err_fwd": err_fwd, "err_dx": err_dx,
        "err_dg": err_dg, "err_db": err_db,
        "fwd_ms": fwd_sec * 1e3, "bwd_ms": bwd_sec * 1e3,
        "fwd_gbs": fwd_gbs, "bwd_gbs": bwd_gbs,
    }


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    shapes = [
        (8192,  256),
        (4096, 1024),
        (2048, 4096),
        ( 512, 8192),
    ]
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    for kind in ("nelu", "nilu"):
        print(f"\n=== {kind.upper()} ===")
        print(f"{'dtype':>10} {'M':>6} {'N':>6}   "
              f"{'err_y':>8} {'err_dx':>8} {'err_dγ':>8} {'err_dβ':>8}   "
              f"{'fwd ms':>8} {'bwd ms':>8}   {'fwd GB/s':>10} {'bwd GB/s':>10}")
        for dtype in dtypes:
            for M, N in shapes:
                r = _run_one(kind, dtype, M, N)
                print(f"{str(dtype).replace('torch.',''):>10} {M:>6} {N:>6}   "
                      f"{r['err_fwd']:>8.2e} {r['err_dx']:>8.2e} "
                      f"{r['err_dg']:>8.2e} {r['err_db']:>8.2e}   "
                      f"{r['fwd_ms']:>8.3f} {r['bwd_ms']:>8.3f}   "
                      f"{r['fwd_gbs']:>10.1f} {r['bwd_gbs']:>10.1f}")


if __name__ == "__main__":
    main()
