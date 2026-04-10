#!/usr/bin/env python3
"""GPT-2 language model training — GELU vs NELU.

Trains GPT-2 Small/Medium/Large from scratch on FineWeb-Edu 10B tokens.
Both GELU and NELU trained with identical recipe — fair comparison.

Recipe (following llm.c / nanoGPT):
    AdamW, lr=6e-4 (scaled by model), wd=0.1, β=(0.9, 0.95),
    cosine LR → 10% floor, warmup 2000 steps, grad clip 1.0,
    seq_len 1024, FP16/BF16, ~10B tokens.

Models:
    gpt2-small  124M  12L  768d  12h
    gpt2-medium 355M  24L 1024d  16h
    gpt2-large  774M  36L 1280d  20h

Usage (H100×8):
    # GPT-2 Small — GELU and NELU (~1h each)
    torchrun --nproc_per_node=8 train_lm.py --size small --act gelu --wandb
    torchrun --nproc_per_node=8 train_lm.py --size small --act nelu --wandb

    # GPT-2 Medium (~3h each)
    torchrun --nproc_per_node=8 train_lm.py --size medium --act nelu --wandb

    # GPT-2 Large (~8h each)
    torchrun --nproc_per_node=8 train_lm.py --size large --act nelu --wandb

    # Eval only
    python train_lm.py --size small --act gelu --eval-only \
        --resume results/lm/small_gelu/best.pt
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torch.compile safety net (same as other scripts)
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 512
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nelu import NELU
from nelu import NELUCUDA
_NELU_CLS = NELUCUDA if NELUCUDA is not None else NELU

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# ── Model configs ────────────────────────────────────────────────

@dataclass
class GPTConfig:
    vocab_size: int = 50304  # padded for efficiency
    seq_len: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False

CONFIGS = {
    "small":  GPTConfig(n_layer=12, n_head=12, n_embd=768),
    "medium": GPTConfig(n_layer=24, n_head=16, n_embd=1024),
    "large":  GPTConfig(n_layer=36, n_head=20, n_embd=1280),
}

LR_MAP = {"small": 6e-4, "medium": 3e-4, "large": 2.5e-4}

# ── Model ────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg, act_cls):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.act = act_cls()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg, act_cls):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg, act_cls)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg, act_cls):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.seq_len, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg, act_cls) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters()) - self.wpe.weight.numel()


# ── Data ─────────────────────────────────────────────────────────

class FineWebDataset(IterableDataset):
    """Stream FineWeb-Edu from HuggingFace or load from pre-tokenized file."""

    def __init__(self, data_path, seq_len=1024):
        self.seq_len = seq_len
        self.data_path = data_path

    def __iter__(self):
        if self.data_path and os.path.exists(self.data_path):
            import numpy as np
            data = np.memmap(self.data_path, dtype=np.uint16, mode="r")
            n = len(data)
            while True:
                i = torch.randint(0, n - self.seq_len - 1, (1,)).item()
                x = torch.from_numpy(data[i:i+self.seq_len].astype(np.int64))
                y = torch.from_numpy(data[i+1:i+1+self.seq_len].astype(np.int64))
                yield x, y
        else:
            # Stream from HuggingFace
            from datasets import load_dataset
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            ds = load_dataset("HuggingFaceFW/fineweb-edu",
                              name="sample-10BT", split="train", streaming=True)
            buffer = []
            for example in ds:
                tokens = tokenizer(example["text"], truncation=False,
                                   add_special_tokens=False)["input_ids"]
                buffer.extend(tokens)
                while len(buffer) > self.seq_len + 1:
                    chunk = buffer[:self.seq_len + 1]
                    buffer = buffer[self.seq_len:]
                    yield (torch.tensor(chunk[:-1], dtype=torch.long),
                           torch.tensor(chunk[1:], dtype=torch.long))


# ── Training ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="small", choices=CONFIGS.keys())
    parser.add_argument("--act", default="nelu", choices=["gelu", "nelu"])
    parser.add_argument("--data", default=None,
                        help="Path to pre-tokenized data (.bin) or 'hf' for streaming")
    parser.add_argument("--total-tokens", type=float, default=10e9,
                        help="Total tokens to train on (default 10B)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Per-GPU micro batch size (sequences)")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="Auto-computed to reach ~0.5M tokens/step if not set")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Distributed
    args.distributed = int(os.environ.get("RANK", -1)) != -1
    if args.distributed:
        dist.init_process_group("nccl")
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        args.local_rank = args.rank = 0
        args.world_size = 1
        device = torch.device("cuda")
    args.is_main = args.rank == 0

    cfg = CONFIGS[args.size]
    act_cls = nn.GELU if args.act == "gelu" else _NELU_CLS
    peak_lr = LR_MAP[args.size]

    # Auto grad accumulation → ~0.5M tokens per step
    tokens_per_micro = args.batch_size * cfg.seq_len
    target_tokens_per_step = 524_288  # ~0.5M
    if args.grad_accum is None:
        args.grad_accum = max(1, target_tokens_per_step //
                              (tokens_per_micro * args.world_size))
    tokens_per_step = tokens_per_micro * args.grad_accum * args.world_size
    total_steps = int(args.total_tokens / tokens_per_step)
    warmup_steps = 2000

    if args.output_dir is None:
        args.output_dir = f"results/lm/{args.size}_{args.act}"
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed + args.rank)

    if args.is_main:
        n_params = GPT(cfg, act_cls).n_params()
        print(f"GPT-2 {args.size} + {args.act}: {n_params/1e6:.1f}M params")
        print(f"Batch: {args.batch_size}×{args.grad_accum}×{args.world_size} "
              f"= {tokens_per_step:,} tok/step")
        print(f"Steps: {total_steps:,} ({args.total_tokens/1e9:.0f}B tokens)")

    # Model
    model = GPT(cfg, act_cls).to(device)
    if args.compile:
        model = torch.compile(model)
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank])
    raw = model.module if args.distributed else model
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod

    # Eval only
    if args.eval_only:
        assert args.resume, "--resume required for --eval-only"
        ckpt = torch.load(args.resume, map_location=device)
        raw.load_state_dict(ckpt["model"])
        ds = FineWebDataset(args.data, cfg.seq_len)
        loader = DataLoader(ds, batch_size=args.batch_size)
        model.eval()
        losses = []
        for i, (x, y) in enumerate(loader):
            if i >= 100: break
            x, y = x.to(device), y.to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=args.amp):
                _, loss = model(x, y)
            losses.append(loss.item())
        avg = sum(losses) / len(losses)
        if args.is_main:
            print(f"Eval: loss={avg:.4f}, ppl={math.exp(avg):.2f}")
        return

    # Data
    dataset = FineWebDataset(args.data, cfg.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2,
                        pin_memory=True)
    data_iter = iter(loader)

    # Optimizer
    optimizer = optim.AdamW(raw.parameters(), lr=peak_lr,
                            betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    min_lr = peak_lr * 0.1

    def get_lr(step):
        if step < warmup_steps:
            return peak_lr * step / warmup_steps
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * t))

    # Wandb
    if args.wandb and args.is_main and HAS_WANDB:
        wandb.init(project="nelu", group="lm",
                   name=f"gpt2_{args.size}_{args.act}",
                   config=vars(args))

    # Resume — auto-detect last.pt unless --resume given
    start_step = 0
    best_loss = float("inf")
    last_path = f"{args.output_dir}/last.pt"
    resume_path = args.resume
    if resume_path is None and os.path.exists(last_path):
        resume_path = last_path
    if resume_path is not None and os.path.exists(resume_path):
        if args.is_main:
            print(f"  → resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        if "rng_torch" in ckpt:
            torch.set_rng_state(ckpt["rng_torch"].cpu())
            if torch.cuda.is_available() and "rng_cuda" in ckpt:
                torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng_cuda"]])
        if args.is_main:
            print(f"  → resumed: start_step={start_step}  best_loss={best_loss:.4f}")

    # Train
    model.train()
    t0 = time.time()
    running_loss = 0.0

    for step in range(start_step, total_steps):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Accumulation
        loss_accum = 0.0
        for micro in range(args.grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if args.distributed:
                model.require_backward_grad_sync = (micro == args.grad_accum - 1)

            with torch.amp.autocast("cuda", enabled=args.amp):
                _, loss = model(x, y)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            loss_accum += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss_accum

        # Log
        if step % 100 == 0 and step > 0 and args.is_main:
            avg = running_loss / 100
            tok_sec = (100 * tokens_per_step) / (time.time() - t0)
            t0 = time.time()
            running_loss = 0.0
            print(f"step {step}/{total_steps}  loss={avg:.4f}  "
                  f"ppl={math.exp(avg):.1f}  lr={lr:.2e}  "
                  f"tok/s={tok_sec/1e6:.2f}M")
            if args.wandb and HAS_WANDB:
                wandb.log({"step": step, "train_loss": avg,
                           "train_ppl": math.exp(avg), "lr": lr,
                           "tok_per_sec": tok_sec})

        # Eval
        if step % args.eval_interval == 0 and step > 0:
            model.eval()
            eval_losses = []
            for i in range(20):
                try:
                    ex, ey = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    ex, ey = next(data_iter)
                ex, ey = ex.to(device), ey.to(device)
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=args.amp):
                    _, el = model(ex, ey)
                eval_losses.append(el.item())
            val_loss = sum(eval_losses) / len(eval_losses)

            # Gate diagnostics
            gate_ent, binary_frac, w_norm = None, None, None
            try:
                pre_acts = []
                hooks = []
                for m in raw.modules():
                    if isinstance(m, (nn.GELU, NELU)) or (NELUCUDA and isinstance(m, NELUCUDA)):
                        def _h(mod, inp, out, s=pre_acts):
                            s.append(inp[0].detach().float())
                        hooks.append(m.register_forward_hook(_h))
                if hooks:
                    with torch.no_grad():
                        raw(ex[:2])  # tiny forward for hooks
                    for h in hooks:
                        h.remove()
                    if pre_acts:
                        gates = []
                        for z in pre_acts:
                            rms = z.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
                            g_nelu = 0.5*(1+torch.erf(z/(rms*math.sqrt(2))))
                            g_gelu = 0.5*(1+torch.erf(z/math.sqrt(2)))
                            g = g_nelu if args.act == "nelu" else g_gelu
                            gates.append(g.cpu())
                        gates = torch.cat([g.reshape(-1) for g in gates])
                        gc = gates.clamp(1e-7, 1-1e-7)
                        gate_ent = -(gc*gc.log()+(1-gc)*(1-gc).log()).mean().item()
                        binary_frac = ((gates<0.05)|(gates>0.95)).float().mean().item()
                w_norm = sum(p.pow(2).sum() for p in raw.parameters()
                             if p.dim()>=2).sqrt().item()
            except Exception:
                pass

            if args.is_main:
                diag_str = ""
                if gate_ent is not None:
                    diag_str = f"  gate_ent={gate_ent:.4f} binary={binary_frac:.1%} ||W||={w_norm:.1f}"
                print(f"  eval: loss={val_loss:.4f} ppl={math.exp(val_loss):.1f}{diag_str}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save({"model": raw.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scaler": scaler.state_dict() if args.amp else None,
                                "step": step, "best_loss": best_loss,
                                "config": vars(args)},
                               f"{args.output_dir}/best.pt")
                if args.wandb and HAS_WANDB:
                    log_d = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
                    if gate_ent is not None:
                        log_d.update({"gate/entropy": gate_ent,
                                      "gate/binary_frac": binary_frac,
                                      "weight_norm": w_norm})
                    wandb.log(log_d)
            model.train()

        # Periodic save: last.pt (full state, every 500 steps) +
        # named milestone every 5000 (for archival).
        save_now_last = (step % 500 == 0) and (step > start_step)
        save_now_milestone = (step % 5000 == 0) and (step > 0)
        if (save_now_last or save_now_milestone) and args.is_main:
            payload = {
                "model": raw.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if args.amp else None,
                "step": step,
                "best_loss": best_loss,
                "config": vars(args),
                "rng_torch": torch.get_rng_state(),
            }
            if torch.cuda.is_available():
                payload["rng_cuda"] = torch.cuda.get_rng_state_all()
            if save_now_last:
                tmp = f"{args.output_dir}/last.pt.tmp"
                torch.save(payload, tmp)
                os.replace(tmp, f"{args.output_dir}/last.pt")
            if save_now_milestone:
                torch.save(payload, f"{args.output_dir}/step_{step}.pt")

    # Final
    if args.is_main:
        torch.save({"model": raw.state_dict(), "step": total_steps,
                    "best_loss": best_loss},
                   f"{args.output_dir}/final.pt")
        result = {"size": args.size, "act": args.act,
                  "best_loss": best_loss, "best_ppl": math.exp(best_loss),
                  "total_steps": total_steps}
        with open(f"{args.output_dir}/result.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nDone. Best loss: {best_loss:.4f}, ppl: {math.exp(best_loss):.1f}")

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
