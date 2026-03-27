#!/usr/bin/env python3
"""Compute rate-distortion curves + SQNR for every tensor in a model.

GPU-accelerated version using MLX on Apple Silicon. Computes the same
NRMSE and SQNR metrics as compute_rd_curves.py but runs quantization
simulations on the GPU via MLX for significantly faster throughput.

Falls back to CPU (NumPy) on non-Apple-Silicon platforms.

Output is identical to compute_rd_curves.py — the allocator, manifest
builder, and converter work unchanged.

Usage:
    python mint/compute_rd_curves_gpu.py \\
        --model-dir /path/to/Model-BF16 \\
        --output results/model-rd-curves.json
"""

import argparse
import json
import logging
import math
import re
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mint.rd_curves_gpu")

# Configs to evaluate: (bits, group_size)
CONFIGS = [
    (2, 32),
    (2, 64),
    (3, 32),
    (3, 64),
    (4, 32),
    (4, 64),
    (4, 128),
    (5, 32),
    (5, 64),
    (6, 32),
    (6, 64),
    (8, 64),
    (8, 128),
]

# Try MLX first, fall back to NumPy
try:
    import mlx.core as mx

    GPU_AVAILABLE = True
    logger.info("Using MLX GPU acceleration")
except ImportError:
    GPU_AVAILABLE = False
    logger.info("MLX not available, using NumPy CPU fallback")


# ── GPU (MLX) implementations ──────────────────────────────────────────────


def _simulate_quant_mlx(t: mx.array, bits: int, group_size: int):
    """Simulate group-wise RTN quantization, return (signal_power, noise_power).

    Runs on GPU via MLX. Computes both metrics in a single pass to avoid
    redundant quantization between NRMSE and SQNR.
    """
    rows, cols = t.shape

    # Pad columns to group_size multiple
    if cols % group_size != 0:
        pad = group_size - (cols % group_size)
        t = mx.pad(t, [(0, 0), (0, pad)])
        cols = t.shape[1]

    t_grouped = t.reshape(rows, cols // group_size, group_size)
    g_min = t_grouped.min(axis=-1, keepdims=True)
    g_max = t_grouped.max(axis=-1, keepdims=True)
    n_levels = (1 << bits) - 1
    scale = mx.maximum((g_max - g_min) / n_levels, 1e-12)

    quantized = mx.clip(mx.round((t_grouped - g_min) / scale), 0, n_levels)
    dequantized = quantized * scale + g_min

    noise = dequantized - t_grouped
    signal_power = mx.mean(t_grouped * t_grouped).item()
    noise_power = mx.mean(noise * noise).item()

    return signal_power, noise_power


def compute_rd_mlx(
    tensor_np: np.ndarray, bits: int, group_size: int
) -> tuple[float, float]:
    """Compute NRMSE and SQNR for a 2D tensor using MLX GPU.

    Returns (nrmse, sqnr_db).
    """
    t = mx.array(tensor_np, dtype=mx.float32)
    signal_power, noise_power = _simulate_quant_mlx(t, bits, group_size)

    # NRMSE
    rms_orig = math.sqrt(signal_power) if signal_power > 0 else 0.0
    rmse = math.sqrt(noise_power) if noise_power > 0 else 0.0
    nrmse = rmse / rms_orig if rms_orig > 1e-12 else 0.0

    # SQNR
    if noise_power < 1e-20:
        sqnr_db = 100.0
    else:
        sqnr_db = 10 * math.log10(signal_power / noise_power)

    return nrmse, sqnr_db


def compute_metrics_2d_mlx(tensor_np: np.ndarray) -> tuple[dict, dict]:
    """Compute RD curve and SQNR at all configs for a 2D tensor using MLX."""
    rd_curve = {}
    sqnr_values = {}

    for bits, gs in CONFIGS:
        cfg_key = f"{bits}_{gs}"
        nrmse, sqnr_db = compute_rd_mlx(tensor_np, bits, gs)
        rd_curve[cfg_key] = nrmse
        sqnr_values[cfg_key] = sqnr_db

    # 16-bit reference
    rd_curve["16_0"] = 0.0
    sqnr_values["16_0"] = 100.0

    return rd_curve, sqnr_values


def compute_metrics_3d_mlx(tensor_np: np.ndarray) -> tuple[dict, dict]:
    """Compute RD curve and SQNR for a 3D MoE tensor [num_experts, d_in, d_out].

    Uses worst-case NRMSE (max) and worst-case SQNR (min) across experts.
    """
    rd_curve = {f"{b}_{g}": 0.0 for b, g in CONFIGS}
    sqnr_values = {f"{b}_{g}": float("inf") for b, g in CONFIGS}

    for i in range(tensor_np.shape[0]):
        expert_rd, expert_sqnr = compute_metrics_2d_mlx(tensor_np[i])
        for key in expert_rd:
            if key == "16_0":
                continue
            rd_curve[key] = max(rd_curve[key], expert_rd[key])
            sqnr_values[key] = min(sqnr_values[key], expert_sqnr[key])

    rd_curve["16_0"] = 0.0
    sqnr_values["16_0"] = 100.0

    return rd_curve, sqnr_values


# ── CPU (NumPy) implementations ────────────────────────────────────────────


def _simulate_quant_np(t: np.ndarray, bits: int, group_size: int):
    """Simulate group-wise RTN quantization using NumPy (CPU fallback)."""
    rows, cols = t.shape

    if cols % group_size != 0:
        pad = group_size - (cols % group_size)
        t = np.pad(t, [(0, 0), (0, pad)])
        cols = t.shape[1]

    t_grouped = t.reshape(rows, cols // group_size, group_size)
    g_min = t_grouped.min(axis=-1, keepdims=True)
    g_max = t_grouped.max(axis=-1, keepdims=True)
    n_levels = (1 << bits) - 1
    scale = np.maximum((g_max - g_min) / n_levels, 1e-12)

    quantized = np.clip(np.round((t_grouped - g_min) / scale), 0, n_levels)
    dequantized = quantized * scale + g_min

    noise = dequantized - t_grouped
    signal_power = float(np.mean(t_grouped * t_grouped))
    noise_power = float(np.mean(noise * noise))

    return signal_power, noise_power


def compute_rd_np(
    tensor_np: np.ndarray, bits: int, group_size: int
) -> tuple[float, float]:
    """Compute NRMSE and SQNR for a 2D tensor using NumPy."""
    signal_power, noise_power = _simulate_quant_np(tensor_np, bits, group_size)

    rms_orig = math.sqrt(signal_power) if signal_power > 0 else 0.0
    rmse = math.sqrt(noise_power) if noise_power > 0 else 0.0
    nrmse = rmse / rms_orig if rms_orig > 1e-12 else 0.0

    if noise_power < 1e-20:
        sqnr_db = 100.0
    else:
        sqnr_db = 10 * math.log10(signal_power / noise_power)

    return nrmse, sqnr_db


def compute_metrics_2d_np(tensor_np: np.ndarray) -> tuple[dict, dict]:
    """Compute RD curve and SQNR at all configs for a 2D tensor using NumPy."""
    rd_curve = {}
    sqnr_values = {}

    for bits, gs in CONFIGS:
        cfg_key = f"{bits}_{gs}"
        nrmse, sqnr_db = compute_rd_np(tensor_np, bits, gs)
        rd_curve[cfg_key] = nrmse
        sqnr_values[cfg_key] = sqnr_db

    rd_curve["16_0"] = 0.0
    sqnr_values["16_0"] = 100.0

    return rd_curve, sqnr_values


def compute_metrics_3d_np(tensor_np: np.ndarray) -> tuple[dict, dict]:
    """Compute RD curve and SQNR for a 3D MoE tensor using NumPy."""
    rd_curve = {f"{b}_{g}": 0.0 for b, g in CONFIGS}
    sqnr_values = {f"{b}_{g}": float("inf") for b, g in CONFIGS}

    for i in range(tensor_np.shape[0]):
        expert_rd, expert_sqnr = compute_metrics_2d_np(tensor_np[i])
        for key in expert_rd:
            if key == "16_0":
                continue
            rd_curve[key] = max(rd_curve[key], expert_rd[key])
            sqnr_values[key] = min(sqnr_values[key], expert_sqnr[key])

    rd_curve["16_0"] = 0.0
    sqnr_values["16_0"] = 100.0

    return rd_curve, sqnr_values


# ── Dispatch ────────────────────────────────────────────────────────────────


def compute_metrics(tensor_np: np.ndarray) -> tuple[dict, dict]:
    """Compute RD curve and SQNR for a tensor, dispatching to GPU or CPU.

    Handles 2D, 3D (MoE), and >3D (vision) tensors. Returns (rd_curve, sqnr).
    """
    ndim = tensor_np.ndim

    # >3D tensors (e.g., vision patch_embed): flatten to 2D
    if ndim > 3:
        tensor_np = tensor_np.reshape(tensor_np.shape[0], -1)
        ndim = 2

    if GPU_AVAILABLE:
        if ndim == 3:
            return compute_metrics_3d_mlx(tensor_np)
        return compute_metrics_2d_mlx(tensor_np)
    else:
        if ndim == 3:
            return compute_metrics_3d_np(tensor_np)
        return compute_metrics_2d_np(tensor_np)


# ── FP8 dequantization ─────────────────────────────────────────────────────


def dequant_fp8(weight_np: np.ndarray, scale_inv_np: np.ndarray) -> np.ndarray:
    """Dequantize FP8 weight using block-wise scale_inv (128x128 blocks)."""
    block_size = 128
    w = weight_np.astype(np.float32)
    si = scale_inv_np.astype(np.float32)

    # Expand 2D block scales: [r/bs, c/bs] -> [r, c]
    si_expanded = np.repeat(np.repeat(si, block_size, axis=0), block_size, axis=1)
    si_expanded = si_expanded[: w.shape[0], : w.shape[1]]

    return w * si_expanded


# ── Utilities ───────────────────────────────────────────────────────────────


def extract_layer_idx(name: str):
    """Extract layer index from tensor name, or None."""
    m = re.search(r"layers\.(\d+)", name)
    return int(m.group(1)) if m else None


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compute rate-distortion curves (GPU-accelerated)"
    )
    parser.add_argument("--model-dir", required=True, help="Path to model directory")
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
    logger.info(f"Backend: {'MLX GPU' if GPU_AVAILABLE else 'NumPy CPU'}")

    all_tensors = {}
    t0 = time.time()
    processed = 0
    skipped = 0

    for shard_file in shard_files:
        logger.info(f"Processing {shard_file.name}...")

        # Load shard via safetensors. Prefer NumPy (no torch dependency)
        # but fall back to PyTorch for dtypes NumPy can't handle (bfloat16).
        use_pt = False
        try:
            sf = safe_open(str(shard_file), framework="numpy", device="cpu")
            # Probe first key to catch bfloat16 errors early
            first_key = list(sf.keys())[0]
            _ = sf.get_tensor(first_key)
        except (TypeError, Exception):
            sf = safe_open(str(shard_file), framework="pt", device="cpu")
            use_pt = True

        keys = list(sf.keys())

        for name in keys:
            # Skip FP8 scale tensors — consumed during dequant
            if name.endswith("weight_scale_inv"):
                skipped += 1
                continue

            tensor_raw = sf.get_tensor(name)

            # Convert to NumPy if loaded via PyTorch
            if hasattr(tensor_raw, "numpy"):
                tensor_np = tensor_raw.float().numpy()
            else:
                tensor_np = tensor_raw.astype(np.float32) if tensor_raw.dtype != np.float32 else tensor_raw

            # FP8 block dequantization
            scale_name = name + "_scale_inv"
            if scale_name in keys:
                scale_raw = sf.get_tensor(scale_name)
                if hasattr(scale_raw, "numpy"):
                    scale_np = scale_raw.float().numpy()
                else:
                    scale_np = scale_raw.astype(np.float32)
                tensor_np = dequant_fp8(tensor_np, scale_np)
                del scale_raw, scale_np
                logger.debug(f"  Dequantized FP8: {name}")

            shape = tuple(tensor_np.shape)
            num_params = tensor_np.size

            # Skip 1D tensors (norms, biases)
            if tensor_np.ndim < 2:
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

            rd_curve, sqnr_values = compute_metrics(tensor_np)

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
            del tensor_raw, tensor_np

        # Ensure GPU work is flushed between shards
        if GPU_AVAILABLE:
            mx.eval(mx.array([0]))

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
