#!/usr/bin/env python3
"""Pre-tokenize FineWeb-Edu 10B sample for GPT-2 training.

Downloads from HuggingFace, tokenizes with GPT-2 tokenizer,
saves as memory-mapped numpy file for fast random access.

Usage:
    python scripts/prepare_fineweb.py --output data/fineweb-edu
"""

import argparse
import os
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/fineweb-edu")
    parser.add_argument("--shard-size", type=int, default=100_000_000,
                        help="Tokens per shard (default 100M)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    from datasets import load_dataset
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)

    print("Tokenizing FineWeb-Edu 10B...")
    all_tokens = []
    total = 0
    shard_idx = 0

    for example in tqdm(ds, desc="tokenizing"):
        tokens = tokenizer(example["text"], truncation=False,
                          add_special_tokens=False)["input_ids"]
        all_tokens.extend(tokens)
        total += len(tokens)

        # Save shards
        while len(all_tokens) >= args.shard_size:
            chunk = np.array(all_tokens[:args.shard_size], dtype=np.uint16)
            path = os.path.join(args.output, f"shard_{shard_idx:04d}.bin")
            chunk.tofile(path)
            print(f"  Saved {path} ({len(chunk):,} tokens)")
            all_tokens = all_tokens[args.shard_size:]
            shard_idx += 1

    # Save remainder
    if all_tokens:
        chunk = np.array(all_tokens, dtype=np.uint16)
        path = os.path.join(args.output, f"shard_{shard_idx:04d}.bin")
        chunk.tofile(path)

    # Concatenate all shards into one file
    print("\nConcatenating shards...")
    all_shards = sorted(f for f in os.listdir(args.output) if f.startswith("shard_"))
    with open(os.path.join(args.output, "train.bin"), "wb") as out:
        for shard in all_shards:
            data = np.fromfile(os.path.join(args.output, shard), dtype=np.uint16)
            data.tofile(out)

    # Clean up shards
    for shard in all_shards:
        os.remove(os.path.join(args.output, shard))

    total_file = np.memmap(os.path.join(args.output, "train.bin"),
                           dtype=np.uint16, mode="r")
    print(f"\nDone. {len(total_file):,} tokens saved to {args.output}/train.bin")
    print(f"File size: {os.path.getsize(os.path.join(args.output, 'train.bin')) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
