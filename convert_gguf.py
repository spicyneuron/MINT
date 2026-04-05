#!/usr/bin/env python3
"""Convert a BF16 model to GGUF with MINT per-tensor quantization allocation.

Two-step process:
  1. BF16 safetensors → F16 GGUF (via llama.cpp's convert_hf_to_gguf.py)
  2. F16 GGUF → mixed-quant GGUF (via llama-quantize --tensor-type-file)

Usage:
    python mint/convert_gguf.py \
        --model-dir /path/to/Model-BF16 \
        --allocation results/model-allocation-19gb.json \
        --output /path/to/Model-MINT.gguf \
        --llama-cpp /path/to/llama.cpp

    # Or with a manifest (will extract allocation from it):
    python mint/convert_gguf.py \
        --model-dir /path/to/Model-BF16 \
        --manifest results/model-manifest-19gb.json \
        --output /path/to/Model-MINT.gguf
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mint.convert_gguf")

# MINT (bits, group_size) → GGUF quant type mapping
# NOTE: Use ggml_type names (Q4_K, Q3_K), NOT ftype names (Q4_K_M, Q3_K_M).
# The _M/_S/_L suffixes are quantization modes, not tensor types.
MINT_TO_GGUF = {
    (2, 32): "Q2_K",
    (2, 64): "Q2_K",
    (3, 32): "Q3_K",
    (3, 64): "Q3_K",
    (4, 32): "Q4_K",
    (4, 64): "Q4_K",
    (4, 128): "Q4_K",
    (5, 32): "Q5_K",
    (5, 64): "Q5_K",
    (6, 32): "Q6_K",
    (6, 64): "Q6_K",
    (8, 64): "Q8_0",
    (8, 128): "Q8_0",
    (16, 0): "F16",
}

# HF tensor name → GGUF tensor name patterns
# These cover the major architectures (Llama, Qwen, Mistral, Mixtral)
HF_TO_GGUF_PATTERNS = [
    # Embeddings
    ("model.embed_tokens.weight", "token_embd.weight"),
    ("lm_head.weight", "output.weight"),
    ("model.norm.weight", "output_norm.weight"),
    # Per-layer attention
    ("model.layers.{N}.self_attn.q_proj", "blk.{N}.attn_q"),
    ("model.layers.{N}.self_attn.k_proj", "blk.{N}.attn_k"),
    ("model.layers.{N}.self_attn.v_proj", "blk.{N}.attn_v"),
    ("model.layers.{N}.self_attn.o_proj", "blk.{N}.attn_output"),
    ("model.layers.{N}.self_attn.qkv_proj", "blk.{N}.attn_qkv"),
    # Per-layer FFN (dense models)
    ("model.layers.{N}.mlp.gate_proj", "blk.{N}.ffn_gate"),
    ("model.layers.{N}.mlp.up_proj", "blk.{N}.ffn_up"),
    ("model.layers.{N}.mlp.down_proj", "blk.{N}.ffn_down"),
    # Per-layer FFN (MoE gate/router)
    ("model.layers.{N}.mlp.gate", "blk.{N}.ffn_gate_inp"),
    # Per-layer norms
    ("model.layers.{N}.input_layernorm", "blk.{N}.attn_norm"),
    ("model.layers.{N}.post_attention_layernorm", "blk.{N}.ffn_norm"),
    # Mixtral expert pattern
    ("model.layers.{N}.block_sparse_moe.experts.{E}.w1", "blk.{N}.ffn_gate_exps.{E}"),
    ("model.layers.{N}.block_sparse_moe.experts.{E}.w2", "blk.{N}.ffn_down_exps.{E}"),
    ("model.layers.{N}.block_sparse_moe.experts.{E}.w3", "blk.{N}.ffn_up_exps.{E}"),
    ("model.layers.{N}.block_sparse_moe.gate", "blk.{N}.ffn_gate_inp"),
    # Qwen/Llama-4 MoE expert pattern
    ("model.layers.{N}.mlp.experts.{E}.gate_proj", "blk.{N}.ffn_gate_exps.{E}"),
    ("model.layers.{N}.mlp.experts.{E}.up_proj", "blk.{N}.ffn_up_exps.{E}"),
    ("model.layers.{N}.mlp.experts.{E}.down_proj", "blk.{N}.ffn_down_exps.{E}"),
    # Shared expert (Qwen MoE)
    ("model.layers.{N}.mlp.shared_expert.gate_proj", "blk.{N}.ffn_gate_shexp"),
    ("model.layers.{N}.mlp.shared_expert.up_proj", "blk.{N}.ffn_up_shexp"),
    ("model.layers.{N}.mlp.shared_expert.down_proj", "blk.{N}.ffn_down_shexp"),
    ("model.layers.{N}.mlp.shared_expert_gate", "blk.{N}.ffn_gate_shexp_inp"),
]


def hf_name_to_gguf(hf_name: str) -> str:
    """Convert HF tensor name to GGUF tensor name."""
    import re

    # Strip .weight suffix for matching
    base = hf_name.replace(".weight", "")

    for hf_pattern, gguf_pattern in HF_TO_GGUF_PATTERNS:
        hf_pat = hf_pattern.replace(".weight", "")
        # Build regex from pattern
        regex = hf_pat.replace(".", r"\.")
        regex = regex.replace("{N}", r"(\d+)")
        regex = regex.replace("{E}", r"(\d+)")
        regex = "^" + regex + "$"

        m = re.match(regex, base)
        if m:
            result = gguf_pattern
            groups = list(m.groups())
            # Replace {N} and {E} with captured numbers
            if "{N}" in result and groups:
                result = result.replace("{N}", groups.pop(0))
            if "{E}" in result and groups:
                result = result.replace("{E}", groups.pop(0))
            return result + ".weight"

    return None


def load_allocation(path: str) -> dict:
    """Load allocation JSON and extract per-tensor decisions."""
    data = json.load(open(path))

    decisions = {}
    if "allocations" in data:
        # MINT allocator output format
        for tname, info in data["allocations"].items():
            decisions[tname] = {
                "bits": info["bits"],
                "group_size": info.get("group_size", 0),
            }
    elif "tensors" in data:
        # V2 manifest format
        for tname, info in data["tensors"].items():
            dec = info.get("decision", {})
            decisions[tname] = {
                "bits": dec.get("bits", 16),
                "group_size": dec.get("group_size", 0),
            }
    elif "shards" in data:
        # V1 manifest format
        for shard in data["shards"].values():
            for tname, info in shard["tensors"].items():
                dec = info.get("decision", {})
                decisions[tname] = {
                    "bits": dec.get("bits", 16),
                    "group_size": dec.get("group_size", 64),
                }
    else:
        raise ValueError(f"Unrecognized allocation format in {path}")

    return decisions


def mint_bits_to_gguf_type(bits: int, group_size: int) -> str:
    """Map MINT (bits, group_size) to GGUF quantization type."""
    key = (bits, group_size)
    if key in MINT_TO_GGUF:
        return MINT_TO_GGUF[key]

    # Fallback by bits only
    if bits <= 2:
        return "Q2_K"
    elif bits <= 3:
        return "Q3_K"
    elif bits <= 4:
        return "Q4_K"
    elif bits <= 6:
        return "Q6_K"
    elif bits <= 8:
        return "Q8_0"
    else:
        return "F16"


def build_tensor_type_spec(decisions: dict) -> dict:
    """Build GGUF tensor_name → quant_type mapping from MINT decisions."""
    spec = {}
    unmapped = []

    for hf_name, dec in decisions.items():
        gguf_name = hf_name_to_gguf(hf_name)
        if gguf_name is None:
            unmapped.append(hf_name)
            continue

        gguf_type = mint_bits_to_gguf_type(dec["bits"], dec["group_size"])
        spec[gguf_name] = gguf_type

    if unmapped:
        logger.warning(f"{len(unmapped)} tensors could not be mapped to GGUF names:")
        for name in unmapped[:10]:
            logger.warning(f"  {name}")
        if len(unmapped) > 10:
            logger.warning(f"  ... and {len(unmapped) - 10} more")

    return spec


def write_tensor_type_file(spec: dict, path: str):
    """Write tensor type spec file for llama-quantize."""
    with open(path, "w") as f:
        for tensor_name, quant_type in sorted(spec.items()):
            f.write(f"{tensor_name}={quant_type}\n")
    logger.info(f"Wrote {len(spec)} tensor type overrides to {path}")


def find_llama_cpp():
    """Find llama.cpp installation."""
    # Check known locations
    candidates = [
        Path("/Users/macuser/code/llama.cpp"),
        Path.home() / "llama.cpp",
        Path("/opt/homebrew/bin"),
    ]

    convert_script = None
    quantize_bin = None

    for p in candidates:
        if (p / "convert_hf_to_gguf.py").exists():
            convert_script = p / "convert_hf_to_gguf.py"
        if (p / "llama-quantize").exists():
            quantize_bin = p / "llama-quantize"

    # Also check PATH
    import shutil
    if quantize_bin is None:
        quantize_bin = shutil.which("llama-quantize")

    return convert_script, quantize_bin


def main():
    parser = argparse.ArgumentParser(description="Convert BF16 model to GGUF with MINT allocation")
    parser.add_argument("--model-dir", required=True, help="Path to BF16 model directory")
    parser.add_argument("--allocation", help="MINT allocation JSON")
    parser.add_argument("--manifest", help="MINT manifest JSON (alternative to --allocation)")
    parser.add_argument("--output", required=True, help="Output GGUF file path")
    parser.add_argument("--llama-cpp", help="Path to llama.cpp directory")
    parser.add_argument("--default-type", default="Q2_K",
                        help="Base quant type (default Q2_K to ensure all overrides apply; "
                             "Q4_K_M causes silent override failures due to llama-quantize bug)")
    parser.add_argument("--keep-f16", action="store_true", help="Keep intermediate F16 GGUF file")
    args = parser.parse_args()

    if not args.allocation and not args.manifest:
        parser.error("One of --allocation or --manifest is required")

    alloc_path = args.allocation or args.manifest

    # Find llama.cpp tools
    if args.llama_cpp:
        convert_script = Path(args.llama_cpp) / "convert_hf_to_gguf.py"
        quantize_bin = Path(args.llama_cpp) / "llama-quantize"
        if not quantize_bin.exists():
            import shutil
            quantize_bin = shutil.which("llama-quantize")
    else:
        convert_script, quantize_bin = find_llama_cpp()

    if convert_script is None or not Path(convert_script).exists():
        logger.error("Cannot find convert_hf_to_gguf.py. Install llama.cpp or pass --llama-cpp")
        sys.exit(1)
    if quantize_bin is None:
        logger.error("Cannot find llama-quantize binary. Install llama.cpp or brew install llama.cpp")
        sys.exit(1)

    logger.info(f"Using convert script: {convert_script}")
    logger.info(f"Using quantize binary: {quantize_bin}")

    # Load MINT allocation
    logger.info(f"Loading allocation: {alloc_path}")
    decisions = load_allocation(alloc_path)
    logger.info(f"Loaded {len(decisions)} tensor decisions")

    # Build tensor type spec
    spec = build_tensor_type_spec(decisions)

    # Print distribution
    from collections import Counter
    type_counts = Counter(spec.values())
    logger.info("GGUF type distribution:")
    for qtype, count in sorted(type_counts.items()):
        logger.info(f"  {qtype}: {count} tensors")

    # Step 1: Convert BF16 → F16 GGUF
    output_path = Path(args.output)
    f16_path = output_path.with_suffix(".f16.gguf")

    if f16_path.exists():
        logger.info(f"F16 GGUF already exists: {f16_path}")
    else:
        logger.info(f"Converting BF16 → F16 GGUF: {f16_path}")
        cmd = [
            sys.executable, str(convert_script),
            "--outfile", str(f16_path),
            "--outtype", "f16",
            str(args.model_dir),
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            logger.error(f"convert_hf_to_gguf.py failed with code {result.returncode}")
            sys.exit(1)

    # Step 2: Write tensor type file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        spec_file = f.name
        for tensor_name, quant_type in sorted(spec.items()):
            f.write(f"{tensor_name}={quant_type}\n")

    logger.info(f"Tensor type spec: {spec_file}")

    # Step 3: Run llama-quantize
    logger.info(f"Quantizing with MINT allocation → {output_path}")
    cmd = [
        str(quantize_bin),
        "--tensor-type-file", spec_file,
        str(f16_path),
        str(output_path),
        args.default_type,
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    # Cleanup
    os.unlink(spec_file)
    if not args.keep_f16 and f16_path.exists():
        logger.info(f"Removing intermediate F16 GGUF: {f16_path}")
        f16_path.unlink()

    if result.returncode != 0:
        logger.error(f"llama-quantize failed with code {result.returncode}")
        sys.exit(1)

    # Report
    if output_path.exists():
        size_gb = output_path.stat().st_size / (1024**3)
        logger.info(f"Done! Output: {output_path} ({size_gb:.2f} GB)")
    else:
        logger.error("Output file not created")
        sys.exit(1)


if __name__ == "__main__":
    main()
