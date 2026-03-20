#!/usr/bin/env python3
"""Build a MINT manifest from allocation + model safetensors.

Reads the model's safetensor shards directly to get tensor shapes/dtypes,
and maps each tensor to its allocation decision.

Usage:
    python mint/build_manifest.py \
        --allocation results/model-allocation-19gb.json \
        --model-dir /path/to/Model-BF16 \
        --output results/model-manifest-19gb.json
"""

import argparse
import json
import logging
from pathlib import Path

from safetensors import safe_open

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mint.build_manifest")


def main():
    parser = argparse.ArgumentParser(description="Build manifest from allocation + model dir")
    parser.add_argument("--allocation", required=True, help="Allocation JSON from allocator.py")
    parser.add_argument("--model-dir", required=True, help="Path to BF16 model directory")
    parser.add_argument("--output", required=True, help="Output manifest path")
    args = parser.parse_args()

    allocation = json.load(open(args.allocation))
    model_dir = Path(args.model_dir)
    alloc_map = allocation["allocations"]

    # Read model config
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    model_name = config_path.parent.name
    total_layers = config.get("num_hidden_layers", 48)

    # Build shard structure from safetensor files
    shard_files = sorted(model_dir.glob("model-*.safetensors"))
    if not shard_files:
        shard_files = sorted(model_dir.glob("*.safetensors"))

    shards = {}
    for sf in shard_files:
        tensors = {}
        with safe_open(str(sf), framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                shape = list(t.shape)
                dtype = str(t.dtype)
                num_params = t.numel()

                tensor_info = {
                    "shape": shape,
                    "dtype": dtype,
                    "num_params": num_params,
                }

                if key in alloc_map:
                    a = alloc_map[key]
                    tensor_info["decision"] = {
                        "bits": a["bits"],
                        "group_size": a["group_size"],
                        "reason": allocation.get("selection_method", allocation.get("solver", "allocation_selected")),
                        "nrmse": a["nrmse"],
                        "prior": a["prior"],
                    }
                else:
                    # Default: 16-bit (unquantized)
                    tensor_info["decision"] = {
                        "bits": 16,
                        "group_size": 0,
                        "reason": "not_in_allocation",
                    }
                    logger.warning(f"Tensor {key} not in allocation, defaulting to 16-bit")

                tensors[key] = tensor_info
                del t

        shards[sf.name] = {
            "file": sf.name,
            "tensors": tensors,
        }

    manifest = {
        "model": model_name,
        "config": {
            "source_bits": 16,
            "optimizer": allocation.get("solver", "knapsack_greedy"),
            "budget_gb": allocation.get("budget_gb", allocation["total_size_gb"]),
            "sqnr_floor_db": allocation["sqnr_floor_db"],
        },
        "total_layers": total_layers,
        "total_tensors": sum(len(s["tensors"]) for s in shards.values()),
        "shards": shards,
        "summary": {
            "total_params": allocation["total_params"],
            "bits_distribution": allocation["bits_distribution"],
            "estimated_size_gb": allocation["total_size_gb"],
            "average_bits": allocation["average_bits"],
            "solver": allocation["solver"],
            "solver_runtime_ms": allocation["solver_runtime_ms"],
            "total_loss": allocation["total_loss"],
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest written to {args.output}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Budget: {allocation['budget_gb']:.2f} GB")
    logger.info(f"  Estimated size: {allocation['total_size_gb']:.2f} GB")
    logger.info(f"  Avg bits: {allocation['average_bits']:.2f}")
    logger.info(f"  Tensors: {manifest['total_tensors']}")


if __name__ == "__main__":
    main()
