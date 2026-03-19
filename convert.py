#!/usr/bin/env python3
"""Convert a model to MLX format using MINT mixed-precision allocation.

Uses the MINT bridge (bridge.py) which reads per-tensor (bits, group_size)
from the manifest to create an MLX quant_predicate.

Usage:
    python mint/convert.py \
        --hf-path /path/to/Model-BF16 \
        --mlx-path /path/to/Model-MINT \
        --manifest results/model-manifest-19gb.json
"""

import argparse
import collections
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("mint.convert")


def summarize_manifest_families(manifest: dict) -> tuple[collections.Counter, collections.Counter]:
    counts = collections.Counter()
    params = collections.Counter()
    for shard_data in manifest["shards"].values():
        for name, info in shard_data["tensors"].items():
            if name.startswith("model.language_model.") or name.startswith("lm_head."):
                family = "language_model"
            elif name.startswith("model.visual."):
                family = "visual"
            elif name.startswith("mtp."):
                family = "mtp"
            else:
                family = name.split(".", 1)[0]
            counts[family] += 1
            params[family] += info.get("num_params", 0)
    return counts, params


def summarize_output_families(index_data: dict) -> collections.Counter:
    counts = collections.Counter()
    for name in index_data["weight_map"]:
        if name.startswith("language_model."):
            family = "language_model"
        else:
            family = name.split(".", 1)[0]
        counts[family] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Convert model to MLX with MINT mixed-precision"
    )
    parser.add_argument("--hf-path", required=True, help="Path to BF16 model")
    parser.add_argument("--mlx-path", required=True, help="Output path")
    parser.add_argument("--manifest", required=True, help="MINT manifest JSON")
    parser.add_argument("--default-bits", type=int, default=4)
    parser.add_argument("--default-group-size", type=int, default=64)
    parser.add_argument("--dtype", default=None, choices=["float16", "bfloat16"])
    args = parser.parse_args()

    # Import bridge from same directory
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    from bridge import load_manifest, create_knapsack_predicate

    manifest = load_manifest(Path(args.manifest))
    predicate = create_knapsack_predicate(manifest)

    output_dir = Path(args.mlx_path)
    if output_dir.exists():
        logger.info(f"Removing existing output: {output_dir}")
        import shutil
        shutil.rmtree(str(output_dir))

    logger.info(f"Converting {args.hf_path} -> {args.mlx_path}")
    logger.info(f"Manifest: {args.manifest}")
    logger.info(f"Default: {args.default_bits}-bit g{args.default_group_size}")

    from mlx_lm import convert

    start = time.time()
    convert(
        hf_path=args.hf_path,
        mlx_path=args.mlx_path,
        quantize=True,
        q_bits=args.default_bits,
        q_group_size=args.default_group_size,
        quant_predicate=predicate,
        dtype=args.dtype,
    )
    elapsed = time.time() - start
    logger.info(f"Conversion completed in {elapsed / 60:.1f} minutes")

    # Verify
    total_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    logger.info(f"Output size: {total_bytes / (1024**3):.1f} GB")

    index_path = output_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index_data = json.load(f)
        manifest_families, manifest_params = summarize_manifest_families(manifest)
        output_families = summarize_output_families(index_data)
        missing_families = [family for family in sorted(manifest_families) if family not in output_families]
        if missing_families:
            details = ", ".join(
                f"{family} ({manifest_families[family]} tensors, {manifest_params[family]:,} params)"
                for family in missing_families
            )
            logger.warning(
                "Manifest includes tensor families not present in the MLX output index: %s",
                details,
            )

    # Show config quantization info
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        q = config.get("quantization", {})
        default_bits = q.get("bits")
        default_group_size = q.get("group_size")
        override_count = sum(1 for k, v in q.items() if isinstance(v, dict) and "bits" in v)
        if default_bits is not None and default_group_size is not None:
            logger.info(f"Default quantization: {default_bits}-bit g{default_group_size}")
        logger.info(f"Explicit module overrides: {override_count}")
        from collections import Counter
        bits_dist = Counter()
        for k, v in q.items():
            if isinstance(v, dict) and "bits" in v:
                bits_dist[v["bits"]] += 1
        if bits_dist:
            logger.info(f"Override module distribution: {dict(sorted(bits_dist.items()))}")


if __name__ == "__main__":
    main()
