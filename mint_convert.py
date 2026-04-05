#!/usr/bin/env python3
"""MINT Convert — Convert MINT models to MLX or GGUF format.

Handles all architectures including MoE models with fused expert tensors.
Automatically splits MLX switch_mlp tensors back to HuggingFace format
for GGUF conversion.

Usage:
    # Default: MLX format (no conversion needed, just validates)
    python mint_convert.py --model models/my-mint-model

    # GGUF format for Ollama/llama.cpp
    python mint_convert.py --model models/my-mint-model --format gguf --output model.gguf

    # GGUF with custom quant type
    python mint_convert.py --model models/my-mint-model --format gguf --output model.gguf --quant Q4_K_M
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import safetensors.torch as st


# ── Dequantization ───────────────────────────────────────────────────────────

def dequantize_tensor(packed_weight, scales, biases, bits, group_size):
    """Dequantize MLX packed weights to float. Works for 2D and 3D tensors."""
    elems_per_int = 32 // bits
    mask = (1 << bits) - 1

    orig_shape = list(packed_weight.shape)
    packed = packed_weight.to(torch.int64)

    # Unpack: extract sub-values from each uint32
    parts = []
    for i in range(elems_per_int):
        parts.append((packed >> (i * bits)) & mask)
    unpacked = torch.stack(parts, dim=-1).reshape(*orig_shape[:-1], -1).float()

    # Apply affine dequant per group: float = scale * int_val + bias
    s = scales.float().repeat_interleave(group_size, dim=-1)
    b = biases.float().repeat_interleave(group_size, dim=-1)
    out_dim = unpacked.shape[-1]
    s = s[..., :out_dim]
    b = b[..., :out_dim]

    return (s * unpacked + b).to(torch.bfloat16)


def get_tensor_config(name, quant_config, default_bits, default_gs):
    """Get (bits, group_size) for a tensor from MINT config."""
    base = name
    for suffix in [".weight", ".scales", ".biases"]:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    if base in quant_config and isinstance(quant_config[base], dict):
        return quant_config[base].get("bits", default_bits), \
               quant_config[base].get("group_size", default_gs)
    return default_bits, default_gs


# ── MoE Tensor Splitting ─────────────────────────────────────────────────────

# MLX fused format: switch_mlp.{proj}.weight [num_experts, out, packed_in]
# HF individual format: experts.{E}.{proj}.weight [out, in]

MOE_PROJ_MAP = {
    "switch_mlp.gate_proj": "experts.{E}.gate_proj",
    "switch_mlp.up_proj": "experts.{E}.up_proj",
    "switch_mlp.down_proj": "experts.{E}.down_proj",
}


def is_moe_fused_tensor(name):
    """Check if this is an MLX fused MoE expert tensor."""
    return "switch_mlp." in name and name.endswith(".weight")


def split_moe_tensor(name, dequantized_3d):
    """Split a fused [num_experts, out, in] tensor into individual expert tensors.

    Returns list of (new_name, tensor_2d) pairs.
    """
    num_experts = dequantized_3d.shape[0]
    results = []

    # Find which projection this is
    for mlx_pattern, hf_pattern in MOE_PROJ_MAP.items():
        if mlx_pattern in name:
            base = name.replace(mlx_pattern + ".weight", "")
            for e in range(num_experts):
                new_name = base + hf_pattern.format(E=e) + ".weight"
                results.append((new_name, dequantized_3d[e]))
            return results

    # Fallback: return as-is
    return [(name, dequantized_3d)]


# ── Main Convert ─────────────────────────────────────────────────────────────

def convert_to_bf16_hf(model_path, output_dir):
    """Convert MINT MLX model to standard BF16 HuggingFace format.

    Handles:
    - 2D quantized tensors: dequantize to BF16
    - 3D fused MoE tensors: dequantize and split into per-expert 2D tensors
    - Non-quantized tensors: pass through
    - Config: strip quantization fields
    """
    with open(os.path.join(model_path, "config.json")) as f:
        config = json.load(f)

    quant_config = config.get("quantization", config.get("quantization_config", {}))
    default_bits = quant_config.get("bits", 4)
    default_gs = quant_config.get("group_size", 64)

    print(f"MINT: {default_bits}-bit, gs={default_gs}")

    # Find safetensor shards
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            shard_files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shard_files = sorted(f for f in os.listdir(model_path) if f.endswith(".safetensors"))

    os.makedirs(output_dir, exist_ok=True)

    total_dequant = 0
    total_split = 0
    total_pass = 0
    new_weight_map = {}
    all_output_tensors = {}

    for shard_file in shard_files:
        print(f"\nProcessing {shard_file}...")
        tensors = st.load_file(os.path.join(model_path, shard_file))

        # Identify quantized bases
        scales_map = {}
        biases_map = {}
        for n in tensors:
            if n.endswith(".scales"):
                base = n.replace(".scales", ".weight")
                scales_map[base] = n
            elif n.endswith(".biases"):
                base = n.replace(".biases", ".weight")
                biases_map[base] = n

        output_tensors = {}

        for name in sorted(tensors.keys()):
            if name.endswith(".scales") or name.endswith(".biases"):
                continue

            t = tensors[name]

            if name in scales_map and name in biases_map:
                # Quantized tensor
                s = tensors[scales_map[name]]
                b = tensors[biases_map[name]]
                bits, gs = get_tensor_config(name, quant_config, default_bits, default_gs)

                dequant = dequantize_tensor(t, s, b, bits, gs)

                if dequant.ndim == 3 and is_moe_fused_tensor(name):
                    # Split fused MoE into individual experts
                    expert_tensors = split_moe_tensor(name, dequant)
                    for new_name, expert_t in expert_tensors:
                        output_tensors[new_name] = expert_t
                    total_split += len(expert_tensors)
                    print(f"  Split {name} [{list(t.shape)}] → {len(expert_tensors)} expert tensors")
                elif dequant.ndim == 3:
                    # Non-MoE 3D tensor (e.g. embed_q) — keep as-is
                    output_tensors[name] = dequant
                    total_dequant += 1
                else:
                    output_tensors[name] = dequant
                    total_dequant += 1
            else:
                output_tensors[name] = t
                total_pass += 1

        # Strip vision/visual tensors (not needed for text-only GGUF)
        output_tensors = {k: v for k, v in output_tensors.items()
                          if not any(x in k.lower() for x in ["vision_tower", "visual", "image_"])}

        # Normalize tensor name prefixes:
        # MLX Qwen3.5 uses "language_model.model.layers" but HF expects "model.layers"
        renamed = {}
        for k, v in output_tensors.items():
            new_k = k
            if new_k.startswith("language_model.model."):
                new_k = "model." + new_k[len("language_model.model."):]
            elif new_k.startswith("language_model."):
                new_k = new_k[len("language_model."):]
            renamed[new_k] = v
        output_tensors = renamed

        # Accumulate all tensors — we'll write properly-numbered shards at the end
        all_output_tensors.update(output_tensors)
        print(f"  Processed {len(output_tensors)} tensors")

        for n in output_tensors:
            new_weight_map[n] = shard_file

    # Write all tensors into properly-numbered shards
    MAX_TENSORS_PER_SHARD = 500
    items = sorted(all_output_tensors.items())
    total_shards = max(1, math.ceil(len(items) / MAX_TENSORS_PER_SHARD))
    print(f"\nWriting {len(items)} tensors to {total_shards} shards...")

    for shard_idx in range(total_shards):
        chunk_start = shard_idx * MAX_TENSORS_PER_SHARD
        chunk = dict(items[chunk_start:chunk_start + MAX_TENSORS_PER_SHARD])
        shard_name = f"model-{shard_idx+1:05d}-of-{total_shards:05d}.safetensors"
        out_path = os.path.join(output_dir, shard_name)
        st.save_file(chunk, out_path)
        for n in chunk:
            new_weight_map[n] = shard_name
        print(f"  {shard_name}: {len(chunk)} tensors")

    # Always write index (we may have created new shard files)
    new_index = {"metadata": {"total_size": 0}, "weight_map": new_weight_map}
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)

    # Clean config
    clean_config = {k: v for k, v in config.items() if "quant" not in k.lower()}
    # Fix model type if needed (language_model prefix)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(clean_config, f, indent=2)

    # Copy non-safetensor files
    for fname in os.listdir(model_path):
        if fname.endswith(".safetensors") or fname == "config.json" or fname == "model.safetensors.index.json":
            continue
        src = os.path.join(model_path, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    print(f"\nDequantized: {total_dequant}, Split MoE: {total_split}, Passthrough: {total_pass}")
    return output_dir


def convert_to_gguf(model_path, output_gguf, quant_type="Q4_1"):
    """Full pipeline: MINT MLX → BF16 HF → F16 GGUF → quantized GGUF."""
    # Use NAS for temp if available (large models need lots of temp space)
    tmp_base = "/Volumes/large/SWAN/tmp" if os.path.exists("/Volumes/large/SWAN") else None
    if tmp_base:
        os.makedirs(tmp_base, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="mint_convert_", dir=tmp_base)

    try:
        # Step 1: Dequantize + split MoE → BF16 HF
        print("Step 1/3: Dequantize MINT → BF16 (with MoE expert splitting)")
        bf16_dir = os.path.join(tmpdir, "bf16")
        convert_to_bf16_hf(model_path, bf16_dir)

        # Step 2: BF16 → F16 GGUF
        print("\nStep 2/3: BF16 → F16 GGUF")
        f16_gguf = os.path.join(tmpdir, "model.f16.gguf")
        convert_script = "/opt/homebrew/Cellar/llama.cpp/8240/libexec/convert_hf_to_gguf.py"
        if not os.path.exists(convert_script):
            convert_script = shutil.which("convert_hf_to_gguf.py") or "convert_hf_to_gguf.py"

        result = subprocess.run(
            [sys.executable, convert_script, "--outfile", f16_gguf, "--outtype", "f16", bf16_dir],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"GGUF conversion failed:\n{result.stderr[-1000:]}")
            sys.exit(1)
        print(f"  F16 GGUF: {os.path.getsize(f16_gguf) / 1e9:.1f} GB")

        # Step 3: F16 → quantized GGUF
        print(f"\nStep 3/3: F16 → {quant_type} GGUF")
        quantize_bin = "/opt/homebrew/bin/llama-quantize"
        if not os.path.exists(quantize_bin):
            quantize_bin = shutil.which("llama-quantize") or "llama-quantize"

        os.makedirs(os.path.dirname(output_gguf) or ".", exist_ok=True)
        result = subprocess.run(
            [quantize_bin, f16_gguf, output_gguf, quant_type],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"Quantization failed:\n{result.stderr[-1000:]}")
            sys.exit(1)

        if os.path.exists(output_gguf):
            size_gb = os.path.getsize(output_gguf) / (1024**3)
            print(f"\nOutput: {output_gguf} ({size_gb:.2f} GB)")
        else:
            print("ERROR: Output not created")
            sys.exit(1)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="MINT Convert — Convert MINT models between formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate MLX model (default format)
  python mint_convert.py --model models/my-mint-model

  # Convert to GGUF for Ollama
  python mint_convert.py --model models/my-mint-model --format gguf --output model.gguf

  # Convert to GGUF with Q4_K_M quantization
  python mint_convert.py --model models/my-mint-model --format gguf -o model.gguf --quant Q4_K_M

  # Convert to standard BF16 HuggingFace format
  python mint_convert.py --model models/my-mint-model --format hf --output models/my-model-bf16
""",
    )
    parser.add_argument("--model", required=True, help="Path to MINT MLX model")
    parser.add_argument("--format", choices=["mlx", "gguf", "hf"], default="mlx",
                        help="Output format: mlx (default), gguf, or hf (BF16 HuggingFace)")
    parser.add_argument("--output", "-o", help="Output path (GGUF file or HF directory)")
    parser.add_argument("--quant", default="Q4_1",
                        help="GGUF quantization type (default: Q4_1). Options: Q4_1, Q4_K_M, Q5_K_M, Q8_0")

    args = parser.parse_args()

    if args.format == "mlx":
        print(f"Model: {args.model}")
        print("Format: MLX (native — no conversion needed)")
        print("Use with: mlx_lm.generate or mlx_lm.server")
        # Just validate
        config_path = os.path.join(args.model, "config.json")
        if not os.path.exists(config_path):
            print("ERROR: config.json not found")
            sys.exit(1)
        config = json.load(open(config_path))
        quant = config.get("quantization", config.get("quantization_config", {}))
        if quant:
            print(f"MINT config: {quant.get('bits')}-bit, gs={quant.get('group_size')}")
        print("OK — model is ready for MLX inference")

    elif args.format == "gguf":
        if not args.output:
            name = os.path.basename(args.model)
            args.output = f"{name}.gguf"
        t0 = time.time()
        convert_to_gguf(args.model, args.output, args.quant)
        print(f"Completed in {time.time() - t0:.0f}s")
        print(f"\nUsage:")
        print(f"  ollama create mymodel -f <(echo 'FROM {os.path.abspath(args.output)}')")
        print(f"  ollama run mymodel")

    elif args.format == "hf":
        if not args.output:
            args.output = args.model + "-bf16"
        t0 = time.time()
        convert_to_bf16_hf(args.model, args.output)
        print(f"Completed in {time.time() - t0:.0f}s")
        print(f"\nUsage:")
        print(f"  from transformers import AutoModelForCausalLM")
        print(f"  model = AutoModelForCausalLM.from_pretrained('{args.output}')")


if __name__ == "__main__":
    main()
