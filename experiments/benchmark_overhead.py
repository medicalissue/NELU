"""
Wall-clock overhead benchmark: GELU vs NELU vs NELU (Triton).

Measures forward + backward time per activation on realistic tensor sizes.
Paper claims: "<1% overhead."

Usage:
    python experiments/benchmark_overhead.py
"""

import math
import time
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nelu import NELU
from nelu.cuda_kernel import NELUCUDA


def benchmark_activation(act_fn, z, n_warmup=50, n_iters=200):
    """Benchmark forward + backward time in ms."""
    # Warmup
    for _ in range(n_warmup):
        z_in = z.detach().requires_grad_(True)
        y = act_fn(z_in)
        y.sum().backward()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        z_in = z.detach().requires_grad_(True)
        y = act_fn(z_in)
        y.sum().backward()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1000  # ms
    return elapsed


def main():
    device = "cuda"
    torch.manual_seed(42)

    # Realistic sizes: (batch, seq_len, d_ff) for Transformer MLP
    sizes = [
        ("ViT-Tiny MLP", (64, 197, 768)),
        ("ViT-Base MLP", (32, 197, 3072)),
        ("GPT-2 MLP", (16, 1024, 3072)),
        ("LLaMA-7B MLP", (4, 2048, 11008)),
    ]

    activations = {
        "GELU": nn.GELU(),
        "ReLU": nn.ReLU(),
        "NELU (PyTorch)": NELU(),
        "NELU (CUDA)": NELUCUDA(),
    }

    print(f"{'Size':>20} | ", end="")
    for name in activations:
        print(f"{name:>16} ", end="")
    print()
    print("-" * (22 + 17 * len(activations)))

    for size_name, shape in sizes:
        z = torch.randn(shape, device=device, dtype=torch.float32)
        times = {}
        for act_name, act_fn in activations.items():
            t = benchmark_activation(act_fn, z)
            times[act_name] = t

        gelu_time = times["GELU"]
        print(f"{size_name:>20} | ", end="")
        for name in activations:
            overhead = (times[name] / gelu_time - 1) * 100
            print(f"{times[name]:7.3f}ms ({overhead:+5.1f}%) ", end="")
        print()

    # Also benchmark in float16
    print(f"\n{'--- float16 ---':>20}")
    for size_name, shape in sizes:
        z = torch.randn(shape, device=device, dtype=torch.float16)
        times = {}
        for act_name, act_fn in activations.items():
            t = benchmark_activation(act_fn, z)
            times[act_name] = t

        gelu_time = times["GELU"]
        print(f"{size_name:>20} | ", end="")
        for name in activations:
            overhead = (times[name] / gelu_time - 1) * 100
            print(f"{times[name]:7.3f}ms ({overhead:+5.1f}%) ", end="")
        print()


if __name__ == "__main__":
    main()
