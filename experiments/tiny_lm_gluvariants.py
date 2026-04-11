#!/usr/bin/env python3
"""Tiny LLaMA-style LM: SwiGLU vs NiLUGLU vs NELUGLU head-to-head.

Purpose
-------
Before committing to a full LLaMA-tiny run (which takes hours to days),
this script trains a small GPT-style model on FineWeb-Edu / WikiText /
OpenWebText (whichever is available) with either SwiGLU, NiLUGLU, or
NELUGLU as the FFN block. Goal: does the "RMS gate normalization"
principle transfer from pointwise activations to GLU blocks?

The experiment is intentionally cheap: ~4 layers, ~256 dim, 100 k
optimizer steps, single GPU, ~1 h wall-clock. With that budget, a
2-seed run already gives a meaningful signal.

Recipe (LLaMA-flavored, scaled down):
    optimizer : AdamW (betas=(0.9, 0.95), wd=0.1)
    lr        : 3e-4 peak, cosine to 10 % floor, 2k-step warmup
    batch     : 32 × seq_len 512 = 16384 tokens / step
    steps     : 100k  (~1.6B tokens)
    model     : n_layers=4, dim=256, n_heads=4, hidden_dim=688
    precision : bf16 AMP, gradient clipping 1.0
    seeds     : 42, 123

Usage:
    # Baseline
    python experiments/tiny_lm_gluvariants.py --ffn swiglu  --seed 42
    # Ours (sigmoid-gate version)
    python experiments/tiny_lm_gluvariants.py --ffn nilu_glu --seed 42
    # Ours (Gaussian-gate version)
    python experiments/tiny_lm_gluvariants.py --ffn nelu_glu --seed 42
    # Aggregate comparison after all runs
    python experiments/tiny_lm_gluvariants.py --aggregate-only

Status: SCAFFOLD — not yet implemented, design doc only.
Next steps:
  1. Wire up FineWeb-Edu streaming or pre-tokenized data loading.
  2. Implement the tiny GPT body (RMSNorm, rotary, causal attn).
  3. Use nelu.SwiGLU / NiLUGLU / NELUGLU for the FFN block
     (already available in nelu/glu.py).
  4. Log train_loss, val_loss, val_ppl to wandb.
  5. If either NiLUGLU or NELUGLU ≥ SwiGLU after 100k steps, graduate
     to a full LLaMA-tiny (6 L × 512 d × full 10B tokens) on H100×8.
  6. If both lose meaningfully, drop §4.3 from the paper and keep the
     framing to NELU + NiLU only.
"""

raise NotImplementedError(
    "This is a design scaffold. Implement before running.\n"
    "The SwiGLU / NiLUGLU / NELUGLU modules are already in nelu/glu.py "
    "and pass a forward+backward smoke test."
)
