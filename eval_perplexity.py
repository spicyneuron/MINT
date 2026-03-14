#!/usr/bin/env python3
"""Perplexity evaluation on WikiText-2 for MINT quantized models.

Loads an MLX model and evaluates perplexity on WikiText-2 test split
with robust statistics (mean, median, trimmed mean).

Usage:
    python mint/eval_perplexity.py \
        --model /path/to/Model-MINT \
        --num-samples 256 \
        --output results/model-ppl.json
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def load_wikitext2(tokenizer, sequence_length: int, num_samples: int, seed: int):
    """Load and tokenize WikiText-2 test split."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Concatenate all text
    text = "\n\n".join([x["text"] for x in ds if x["text"].strip()])

    # Tokenize
    tokens = tokenizer.encode(text)
    tokens = mx.array(tokens)

    # Reshape into sequences
    n_tokens = (len(tokens) // sequence_length) * sequence_length
    tokens = tokens[:n_tokens].reshape(-1, sequence_length)

    # Sample with fixed seed
    rng = np.random.RandomState(seed)
    if num_samples > 0 and num_samples < tokens.shape[0]:
        indices = rng.choice(tokens.shape[0], size=num_samples, replace=False)
        indices.sort()
        tokens = tokens[mx.array(indices.tolist())]

    return tokens


def evaluate_perplexity(model, tokens, batch_size: int = 1):
    """Compute perplexity over tokenized sequences.

    Returns standard PPL (exp of mean loss) plus per-sequence losses
    for computing robust statistics (median, trimmed mean).
    """
    total_loss = 0.0
    total_tokens = 0
    num_batches = tokens.shape[0]
    per_seq_losses = []

    for i in range(0, num_batches, batch_size):
        batch = tokens[i : i + batch_size]
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)
        logits = logits.astype(mx.float32)

        # Per-sequence mean loss (for robust stats)
        seq_loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
        mx.eval(seq_loss)
        per_seq_losses.append(seq_loss.item())

        # Total loss for standard PPL
        loss = nn.losses.cross_entropy(logits, targets, reduction="sum")
        mx.eval(loss)

        total_loss += loss.item()
        total_tokens += targets.size

        if (i // batch_size) % 25 == 0:
            running_ppl = np.exp(total_loss / total_tokens)
            print(f"  Batch {i // batch_size + 1}/{(num_batches + batch_size - 1) // batch_size}, running PPL: {running_ppl:.4f}")

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    # Robust statistics
    seq_ppls = np.exp(np.array(per_seq_losses))
    median_ppl = float(np.median(seq_ppls))
    trim_n = max(1, int(0.05 * len(per_seq_losses)))
    sorted_losses = np.sort(per_seq_losses)
    trimmed_ppl = float(np.exp(sorted_losses[trim_n:-trim_n].mean()))
    outlier_count = int((seq_ppls > 100).sum())

    return perplexity, avg_loss, {
        "median_ppl": median_ppl,
        "trimmed_mean_ppl": trimmed_ppl,
        "outlier_sequences": outlier_count,
        "total_sequences": len(per_seq_losses),
    }


def main():
    parser = argparse.ArgumentParser(description="MINT perplexity evaluation on WikiText-2")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    from mlx_lm import load

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    # Count parameters
    total_params = sum(p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    print(f"Model parameters: {total_params / 1e9:.2f}B")

    print(f"Loading WikiText-2 test split (seq_len={args.sequence_length}, samples={args.num_samples}, seed={args.seed})")
    tokens = load_wikitext2(tokenizer, args.sequence_length, args.num_samples, args.seed)
    print(f"Evaluation data: {tokens.shape[0]} sequences x {tokens.shape[1]} tokens")

    # Get model size
    model_path = Path(args.model)
    model_size_bytes = sum(f.stat().st_size for f in model_path.glob("*.safetensors"))
    model_size_gb = model_size_bytes / (1024 ** 3)

    print("Evaluating perplexity...")
    start = time.time()
    perplexity, avg_loss, robust_stats = evaluate_perplexity(model, tokens, args.batch_size)
    elapsed = time.time() - start

    peak_mem = mx.metal.get_peak_memory() / (1024 ** 3)

    print(f"\n{'='*50}")
    print(f"Model: {args.model}")
    print(f"Perplexity (standard): {perplexity:.4f}")
    print(f"Perplexity (median):   {robust_stats['median_ppl']:.4f}")
    print(f"Perplexity (trimmed):  {robust_stats['trimmed_mean_ppl']:.4f}")
    print(f"Outlier sequences:     {robust_stats['outlier_sequences']}/{robust_stats['total_sequences']}")
    print(f"Avg loss: {avg_loss:.6f}")
    print(f"Model size: {model_size_gb:.2f} GB")
    print(f"Peak memory: {peak_mem:.2f} GB")
    print(f"Eval time: {elapsed:.1f}s")
    print(f"{'='*50}")

    result = {
        "model": args.model,
        "perplexity": float(perplexity),
        "median_ppl": robust_stats["median_ppl"],
        "trimmed_mean_ppl": robust_stats["trimmed_mean_ppl"],
        "outlier_sequences": robust_stats["outlier_sequences"],
        "avg_loss": float(avg_loss),
        "model_size_gb": float(model_size_gb),
        "peak_memory_gb": float(peak_mem),
        "eval_time_seconds": float(elapsed),
        "config": {
            "dataset": "wikitext-2-raw-v1",
            "split": "test",
            "sequence_length": args.sequence_length,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "batch_size": args.batch_size,
        },
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {args.output}")

    return result


if __name__ == "__main__":
    main()
