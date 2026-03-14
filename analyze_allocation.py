#!/usr/bin/env python3
"""Analyze a MINT allocation: bit distribution, group sizes, top tensors by loss.

Can also compare two allocations (e.g., two different budget points).

Usage:
    python mint/analyze_allocation.py \
        --allocation results/model-allocation-19gb.json

    python mint/analyze_allocation.py \
        --allocation results/model-allocation-19gb.json \
        --compare results/model-allocation-24gb.json \
        --ppl-a results/model-19gb-ppl.json \
        --ppl-b results/model-24gb-ppl.json
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mint.analyze")


def load_allocation(alloc_path: str):
    """Extract per-tensor decisions from allocation."""
    a = json.load(open(alloc_path))
    decisions = {}
    for tname, tinfo in a["allocations"].items():
        decisions[tname] = {
            "bits": tinfo["bits"],
            "group_size": tinfo["group_size"],
            "num_params": tinfo["num_params"],
            "nrmse": tinfo.get("nrmse", 0),
            "prior": tinfo.get("prior", 1),
            "loss": tinfo.get("loss", 0),
        }
    return decisions, a


def analyze_single(alloc_path: str, ppl_path: str = None):
    """Analyze a single allocation."""
    dec, alloc = load_allocation(alloc_path)

    print(f"\n{'='*70}")
    print(f"MINT Allocation Analysis: {Path(alloc_path).stem}")
    print(f"{'='*70}")

    print(f"\n--- Summary ---")
    print(f"  Budget:     {alloc['budget_gb']:.2f} GB")
    print(f"  Allocated:  {alloc['total_size_gb']:.2f} GB ({alloc['budget_utilization']:.1%})")
    print(f"  Avg bits:   {alloc['average_bits']:.2f}")
    print(f"  Total loss: {alloc['total_loss']:.6f}")
    print(f"  Tensors:    {alloc['num_tensors']}")

    # Bit distribution
    print(f"\n--- Bit Distribution ---")
    for b, info in sorted(alloc["bits_distribution"].items(), key=lambda x: int(x[0])):
        print(f"  {b:>2s}-bit: {info['params']:>15,} params ({info['percentage']:.1f}%)")

    # Group size distribution
    print(f"\n--- Group Size Distribution ---")
    gs_dist = Counter()
    for d in dec.values():
        if d["bits"] < 16:
            gs_dist[(d["bits"], d["group_size"])] += 1
    for (b, gs), count in sorted(gs_dist.items()):
        print(f"  {b}-bit g{gs}: {count} tensors")

    # Top tensors by loss
    print(f"\n--- Top 20 Tensors by Loss ---")
    sorted_tensors = sorted(
        [(t, d) for t, d in dec.items() if d.get("nrmse", 0) > 0],
        key=lambda x: x[1]["loss"],
        reverse=True,
    )
    print(f"  {'Tensor':<55s} {'Bits':>4s} {'GS':>3s} {'NRMSE':>8s} {'Prior':>5s} {'Loss':>8s}")
    for tname, d in sorted_tensors[:20]:
        short = tname[-55:] if len(tname) > 55 else tname
        print(
            f"  {short:<55s} {d['bits']:>4d} {d['group_size']:>3d} "
            f"{d['nrmse']:>8.5f} {d['prior']:>5.1f} {d['loss']:>8.5f}"
        )

    # PPL (if available)
    if ppl_path:
        try:
            ppl = json.load(open(ppl_path))
            print(f"\n--- Perplexity ---")
            print(f"  Standard: {ppl.get('perplexity', 0):.4f}")
            print(f"  Median:   {ppl.get('median_ppl', 0):.4f}")
            print(f"  Trimmed:  {ppl.get('trimmed_mean_ppl', 0):.4f}")
            print(f"  Size:     {ppl.get('model_size_gb', 0):.2f} GB")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"\n  (PPL skipped: {e})")

    print(f"\n{'='*70}\n")


def analyze_compare(path_a: str, path_b: str, ppl_a_path: str = None, ppl_b_path: str = None):
    """Compare two allocations."""
    dec_a, alloc_a = load_allocation(path_a)
    dec_b, alloc_b = load_allocation(path_b)

    name_a = Path(path_a).stem
    name_b = Path(path_b).stem

    print(f"\n{'='*70}")
    print(f"Allocation Comparison: {name_a} vs {name_b}")
    print(f"{'='*70}")

    print(f"\n--- Summary ---")
    print(f"  {'':>15s} {'A':>12s} {'B':>12s} {'Delta':>10s}")
    print(f"  {'Budget GB':>15s} {alloc_a['budget_gb']:>12.2f} {alloc_b['budget_gb']:>12.2f}")
    print(f"  {'Size GB':>15s} {alloc_a['total_size_gb']:>12.2f} {alloc_b['total_size_gb']:>12.2f} {alloc_b['total_size_gb']-alloc_a['total_size_gb']:>+10.2f}")
    print(f"  {'Avg bits':>15s} {alloc_a['average_bits']:>12.2f} {alloc_b['average_bits']:>12.2f} {alloc_b['average_bits']-alloc_a['average_bits']:>+10.2f}")
    print(f"  {'Total loss':>15s} {alloc_a['total_loss']:>12.6f} {alloc_b['total_loss']:>12.6f} {alloc_b['total_loss']-alloc_a['total_loss']:>+10.6f}")

    # PPL comparison
    if ppl_a_path and ppl_b_path:
        try:
            ppl_a = json.load(open(ppl_a_path))
            ppl_b = json.load(open(ppl_b_path))
            print(f"\n--- Perplexity ---")
            print(f"  {'':>15s} {'A':>12s} {'B':>12s} {'Delta':>10s}")
            for key, label in [("perplexity", "PPL (mean)"), ("median_ppl", "PPL (median)"), ("model_size_gb", "Size (GB)")]:
                va = ppl_a.get(key, 0)
                vb = ppl_b.get(key, 0)
                if va > 0:
                    print(f"  {label:>15s} {va:>12.3f} {vb:>12.3f} {(vb-va)/va*100:>+9.1f}%")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"\n  (PPL comparison skipped: {e})")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze MINT allocation")
    parser.add_argument("--allocation", required=True, help="Allocation JSON")
    parser.add_argument("--compare", help="Second allocation to compare against")
    parser.add_argument("--ppl-a", help="PPL JSON for first allocation")
    parser.add_argument("--ppl-b", help="PPL JSON for second allocation")
    args = parser.parse_args()

    if args.compare:
        analyze_compare(args.allocation, args.compare, args.ppl_a, args.ppl_b)
    else:
        analyze_single(args.allocation, args.ppl_a)


if __name__ == "__main__":
    main()
