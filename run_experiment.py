#!/usr/bin/env python3
"""Orchestrator: convert model using MINT allocation, then evaluate PPL.

Combines Steps 4 and 5 into a single command (assumes manifest already built).

Usage:
    python mint/run_experiment.py \
        --manifest results/model-manifest-19gb.json \
        --model-dir /path/to/Model-BF16 \
        --output-dir /path/to/Model-MINT-19gb \
        --eval-output results/model-mint-19gb-ppl.json \
        --tag "model-19gb"
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mint.run")


def convert_model(manifest_path: str, model_dir: str, output_dir: str):
    """Convert model using MINT bridge predicate."""
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    from bridge import load_manifest, create_knapsack_predicate

    logger.info(f"Loading manifest: {manifest_path}")
    manifest = load_manifest(Path(manifest_path))

    logger.info(f"Creating MINT predicate...")
    predicate = create_knapsack_predicate(manifest)

    logger.info(f"Converting model: {model_dir} -> {output_dir}")
    from mlx_lm import convert

    t0 = time.time()
    convert(
        model_dir,
        mlx_path=output_dir,
        quantize=True,
        quant_predicate=predicate,
    )
    elapsed = time.time() - t0
    logger.info(f"Conversion done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Report actual output size
    out_path = Path(output_dir)
    total_bytes = sum(f.stat().st_size for f in out_path.glob("*.safetensors"))
    logger.info(f"Output model size: {total_bytes / (1024**3):.2f} GB")

    return total_bytes


def evaluate_ppl(model_path: str, output_path: str):
    """Evaluate perplexity using eval_perplexity.py."""
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    from eval_perplexity import main as eval_main

    # Build args for eval_perplexity
    sys.argv = [
        "eval_perplexity.py",
        "--model", model_path,
        "--output", output_path,
        "--num-samples", "128",
        "--sequence-length", "2048",
    ]

    logger.info(f"Evaluating PPL: {model_path}")
    eval_main()
    logger.info(f"PPL results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run MINT experiment (convert + eval)")
    parser.add_argument("--manifest", required=True, help="MINT manifest JSON")
    parser.add_argument("--model-dir", required=True, help="BF16 model directory")
    parser.add_argument("--output-dir", required=True, help="Quantized model output dir")
    parser.add_argument("--eval-output", required=True, help="PPL results JSON output")
    parser.add_argument("--tag", default="mint", help="Experiment tag for logging")
    parser.add_argument("--skip-convert", action="store_true", help="Skip conversion, use existing model")
    parser.add_argument("--skip-eval", action="store_true", help="Skip eval, only convert")
    args = parser.parse_args()

    logger.info(f"=== MINT Experiment: {args.tag} ===")

    if not args.skip_convert:
        model_bytes = convert_model(args.manifest, args.model_dir, args.output_dir)
        gc.collect()
    else:
        out_path = Path(args.output_dir)
        model_bytes = sum(f.stat().st_size for f in out_path.glob("*.safetensors"))
        logger.info(f"Skipping conversion, existing model: {model_bytes / (1024**3):.2f} GB")

    if not args.skip_eval:
        evaluate_ppl(args.output_dir, args.eval_output)

        # Print results
        results = json.load(open(args.eval_output))
        print(f"\n{'='*60}")
        print(f"MINT Experiment Results: {args.tag}")
        print(f"{'='*60}")
        print(f"Model size:    {results.get('model_size_gb', model_bytes/(1024**3)):.2f} GB")
        print(f"PPL (mean):    {results.get('perplexity', 'N/A')}")
        print(f"PPL (median):  {results.get('median_ppl', 'N/A')}")
        print(f"PPL (trimmed): {results.get('trimmed_mean_ppl', 'N/A')}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
