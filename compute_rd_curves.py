#!/usr/bin/env python3
"""Compute rate-distortion curves + SQNR for every tensor in a model.

Loads each safetensor shard, computes NRMSE and SQNR at multiple
(bits, group_size) configurations, and saves the results to JSON.

Usage:
    python mint/compute_rd_curves.py \
        --model-dir /path/to/Model-BF16 \
        --output results/model-rd-curves.json
"""

import argparse
import json
import logging
import math
import re
import time
from pathlib import Path

import torch
from safetensors import safe_open

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mint.rd_curves")

# Configs to evaluate: (bits, group_size)
CONFIGS = [
    (2, 64),
    (3, 64),
    (4, 64),
    (4, 128),
    (8, 64),
    (8, 128),
]


def compute_nrmse(tensor: torch.Tensor, bits: int, group_size: int) -> float:
    """Compute NRMSE for a given (bits, group_size) config."""
    if tensor.dim() < 2:
        return 0.0

    # Handle 3D+ tensors: MoE expert tensors [num_experts, d_in, d_out] or vision patches
    if tensor.dim() > 2:
        t = tensor.reshape(-1, tensor.shape[-1]).float()
        rows, cols = t.shape
    else:
        t = tensor.float()
        rows, cols = t.shape

    # Pad columns to group_size multiple
    if cols % group_size != 0:
        pad = group_size - (cols % group_size)
        t = torch.nn.functional.pad(t, (0, pad))
        cols = t.shape[1]

    t_grouped = t.reshape(rows, cols // group_size, group_size)
    g_min = t_grouped.min(dim=-1, keepdim=True).values
    g_max = t_grouped.max(dim=-1, keepdim=True).values
    n_levels = (1 << bits) - 1
    scale = ((g_max - g_min) / n_levels).clamp(min=1e-12)

    quantized = torch.round((t_grouped - g_min) / scale).clamp(0, n_levels)
    dequantized = quantized * scale + g_min

    diff = dequantized - t_grouped
    rmse = diff.pow(2).mean().sqrt().item()
    rms_orig = t_grouped.pow(2).mean().sqrt().item()

    if rms_orig < 1e-12:
        return 0.0
    return rmse / rms_orig


def compute_sqnr(tensor: torch.Tensor, bits: int, group_size: int) -> float:
    """Compute SQNR in dB for a given (bits, group_size) config."""
    if tensor.dim() < 2:
        return float("inf")

    # Handle 3D MoE expert tensors [num_experts, d_in, d_out]
    if tensor.dim() == 3:
        sqnrs = [compute_sqnr(tensor[i], bits, group_size) for i in range(tensor.shape[0])]
        return min(sqnrs)  # worst-case (lowest SQNR) across experts

    t = tensor.float()
    rows, cols = t.shape

    if cols % group_size != 0:
        pad = group_size - (cols % group_size)
        t = torch.nn.functional.pad(t, (0, pad))
        cols = t.shape[1]

    t_grouped = t.reshape(rows, cols // group_size, group_size)
    g_min = t_grouped.min(dim=-1, keepdim=True).values
    g_max = t_grouped.max(dim=-1, keepdim=True).values
    n_levels = (1 << bits) - 1
    scale = ((g_max - g_min) / n_levels).clamp(min=1e-12)

    quantized = torch.round((t_grouped - g_min) / scale).clamp(0, n_levels)
    dequantized = quantized * scale + g_min

    signal_power = t_grouped.pow(2).mean().item()
    noise_power = (dequantized - t_grouped).pow(2).mean().item()

    if noise_power < 1e-20:
        return 100.0  # effectively noiseless
    return 10 * math.log10(signal_power / noise_power)


def extract_layer_idx(name: str):
    """Extract layer index from tensor name, or None."""
    m = re.search(r"layers\.(\d+)", name)
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser(description="Compute rate-distortion curves")
    parser.add_argument("--model-dir", required=True, help="Path to BF16 model directory")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Read total_layers from config.json
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    model_name = config_path.parent.name
    total_layers = config.get("num_hidden_layers", 48)

    shard_files = sorted(model_dir.glob("model-*.safetensors"))
    if not shard_files:
        shard_files = sorted(model_dir.glob("*.safetensors"))
    logger.info(f"Found {len(shard_files)} shards")

    all_tensors = {}
    t0 = time.time()
    processed = 0
    skipped = 0

    for shard_file in shard_files:
        logger.info(f"Processing {shard_file.name}...")
        with safe_open(str(shard_file), framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                shape = tuple(tensor.shape)
                num_params = tensor.numel()

                # Skip 1D tensors (norms, biases)
                if tensor.dim() < 2:
                    all_tensors[name] = {
                        "shape": shape,
                        "num_params": num_params,
                        "layer_idx": extract_layer_idx(name),
                        "is_1d": True,
                        "rd_curve": {},
                        "sqnr": {},
                    }
                    skipped += 1
                    continue

                rd_curve = {}
                sqnr_values = {}

                for bits, gs in CONFIGS:
                    cfg_key = f"{bits}_{gs}"
                    rd_curve[cfg_key] = compute_nrmse(tensor, bits, gs)
                    sqnr_values[cfg_key] = compute_sqnr(tensor, bits, gs)

                # 16-bit reference
                rd_curve["16_0"] = 0.0
                sqnr_values["16_0"] = 100.0

                all_tensors[name] = {
                    "shape": shape,
                    "num_params": num_params,
                    "layer_idx": extract_layer_idx(name),
                    "is_1d": False,
                    "rd_curve": rd_curve,
                    "sqnr": sqnr_values,
                }

                processed += 1
                if processed % 500 == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        f"  {processed} tensors processed, {skipped} skipped, "
                        f"{elapsed:.0f}s elapsed"
                    )

                # Free memory
                del tensor

    elapsed = time.time() - t0
    logger.info(
        f"Done: {processed} 2D tensors + {skipped} 1D tensors "
        f"in {elapsed:.1f}s ({elapsed/60:.1f} min)"
    )

    output = {
        "model": model_name,
        "total_layers": total_layers,
        "configs": [list(c) for c in CONFIGS],
        "num_2d_tensors": processed,
        "num_1d_tensors": skipped,
        "compute_time_seconds": elapsed,
        "tensors": all_tensors,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f)
    logger.info(f"Saved to {args.output} ({Path(args.output).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
