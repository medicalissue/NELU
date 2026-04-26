"""Throughput benchmark: pure PyTorch vs fused CUDA kernel.

Usage::

    python -m tests.bench_gate_norm_cuda                          # default suite
    python -m tests.bench_gate_norm_cuda --shape 32 1024 3072     # ad-hoc

Reports forward and backward wallclock for a few representative shapes
that show up in the paper's networks.

The benchmark builds the CUDA extension lazily on first call (30–60 s
the first time, cached afterwards). Subsequent runs reuse the cached
.so under ``~/.cache/torch_extensions/``.
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from gate_norm import NELU, NiLU


_SHAPES = [
    # (B, L, D)  — DeiT-Base FFN (B*L, 4*D)
    ("DeiT-Base FFN",   (32, 197, 3072)),
    # ConvNeXt-Tiny stage 3, channels-last (B, H, W, C)
    ("ConvNeXt-T s3",   (32, 14, 14, 384)),
    # Toy small/medium
    ("Small (4,128)",   (4, 128)),
    ("Medium (32,768)", (32, 768)),
    ("Large (8,8192)",  (8, 8192)),
    # Streaming tier
    ("Huge (4,32768)",  (4, 32768)),
]


def _bench_one(layer, x, n_warm=10, n_iter=100, do_backward=False):
    grad_out = torch.randn_like(x) if do_backward else None
    # warm
    for _ in range(n_warm):
        if do_backward:
            x_in = x.detach().clone().requires_grad_(True)
            y = layer(x_in)
            y.backward(grad_out)
        else:
            with torch.no_grad():
                _ = layer(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        if do_backward:
            x_in = x.detach().clone().requires_grad_(True)
            y = layer(x_in)
            y.backward(grad_out)
        else:
            with torch.no_grad():
                _ = layer(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1e3  # ms / iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs="+", type=int, default=None,
                    help="single ad-hoc shape to benchmark")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--kind", default="nelu", choices=["nelu", "nilu"])
    ap.add_argument("--axes", default="channel",
                    help="norm_axes: 'channel', 'sample', or comma-sep ints")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; nothing to benchmark.")
        return

    dtype = getattr(torch, args.dtype)
    cls = NELU if args.kind == "nelu" else NiLU

    if args.axes == "channel":
        norm_axes = "channel"
    elif args.axes == "sample":
        norm_axes = "sample"
    else:
        norm_axes = tuple(int(a) for a in args.axes.split(","))

    shapes = [(f"adhoc {tuple(args.shape)}", tuple(args.shape))] if args.shape else _SHAPES

    print(f"# Gate Normalization benchmark — {args.kind.upper()}, dtype={args.dtype}, axes={args.axes}")
    print(f"# {'name':<22} {'fwd-py(ms)':>12} {'fwd-cu(ms)':>12} {'bwd-py(ms)':>12} {'bwd-cu(ms)':>12} {'fwd×':>8} {'bwd×':>8}")

    for name, shape in shapes:
        x = torch.randn(*shape, dtype=dtype, device="cuda")
        layer = cls(norm_axes=norm_axes).cuda()

        # CUDA path
        os.environ.pop("GATE_NORM_FORCE_PYTHON", None)
        fwd_cu = _bench_one(layer, x, do_backward=False)
        bwd_cu = _bench_one(layer, x, do_backward=True)

        # Python path (same layer; dispatch reads env var on each call).
        os.environ["GATE_NORM_FORCE_PYTHON"] = "1"
        fwd_py = _bench_one(layer, x, do_backward=False)
        bwd_py = _bench_one(layer, x, do_backward=True)
        os.environ.pop("GATE_NORM_FORCE_PYTHON", None)

        fwd_x = fwd_py / max(fwd_cu, 1e-9)
        bwd_x = bwd_py / max(bwd_cu, 1e-9)
        print(f"  {name:<22} {fwd_py:>12.3f} {fwd_cu:>12.3f} {bwd_py:>12.3f} {bwd_cu:>12.3f} {fwd_x:>7.2f}× {bwd_x:>7.2f}×")


if __name__ == "__main__":
    main()
