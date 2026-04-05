#!/usr/bin/env python3
"""Dequantize MINT MLX models to standard BF16 HuggingFace format.

Reads MLX quantized safetensors (packed uint32 + scales + biases)
and writes standard BF16 safetensors that any framework can load.

Usage:
    python mint_dequantize.py \
        --input models/glm-4.7-flash-mint-4bit \
        --output models/glm-4.7-flash-mint-dequant
"""

import argparse
import json
import os
import shutil

import torch
import safetensors.torch as st


def get_tensor_quant_config(tensor_name: str, quant_config: dict) -> tuple:
    """Get (bits, group_size) for a tensor from the quantization config."""
    # Strip .weight/.scales/.biases suffix to get the layer name
    layer_name = tensor_name
    for suffix in [".weight", ".scales", ".biases"]:
        if layer_name.endswith(suffix):
            layer_name = layer_name[:-len(suffix)]
            break

    # Check per-tensor override
    if layer_name in quant_config:
        override = quant_config[layer_name]
        return override.get("bits", quant_config.get("bits", 4)), \
               override.get("group_size", quant_config.get("group_size", 64))

    # Default
    return quant_config.get("bits", 4), quant_config.get("group_size", 64)


def dequantize_tensor(packed_weight: torch.Tensor, scales: torch.Tensor,
                      biases: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Dequantize MLX packed uint32 weights to bfloat16.

    MLX affine quantization: float_val = scale * (int_val - bias)

    Packing: each uint32 holds (32 // bits) values, packed LSB first.
    """
    elems_per_int = 32 // bits
    mask = (1 << bits) - 1

    # packed_weight shape: [out_features, packed_cols] or [d1, d2, packed_cols]
    # We need to unpack the last dimension
    orig_shape = list(packed_weight.shape)
    packed_cols = orig_shape[-1]
    out_cols = packed_cols * elems_per_int

    # Flatten all but last dim for processing
    flat_shape = (-1, packed_cols)
    packed = packed_weight.reshape(flat_shape).to(torch.int64)

    # Unpack: extract each sub-value from the uint32
    unpacked_parts = []
    for i in range(elems_per_int):
        val = (packed >> (i * bits)) & mask
        unpacked_parts.append(val)

    # Stack and interleave: shape [rows, packed_cols * elems_per_int]
    unpacked = torch.stack(unpacked_parts, dim=-1).reshape(-1, out_cols)

    # Apply affine dequantization per group
    # scales/biases shape: [rows, num_groups] or [d1, d2, num_groups]
    scales_flat = scales.reshape(-1, scales.shape[-1]).to(torch.float32)
    biases_flat = biases.reshape(-1, biases.shape[-1]).to(torch.float32)
    unpacked_f = unpacked.to(torch.float32)

    # Expand scales/biases to match unpacked columns
    num_groups = scales_flat.shape[-1]
    # Each group covers group_size elements
    scales_expanded = scales_flat.repeat_interleave(group_size, dim=-1)[:, :out_cols]
    biases_expanded = biases_flat.repeat_interleave(group_size, dim=-1)[:, :out_cols]

    # Dequantize: float = scale * int_val + bias  (MLX affine mode)
    dequantized = scales_expanded * unpacked_f + biases_expanded

    # Reshape back to original dims (replacing packed_cols with out_cols)
    out_shape = orig_shape[:-1] + [out_cols]
    return dequantized.reshape(out_shape).to(torch.bfloat16)


def dequantize_model(input_dir: str, output_dir: str):
    """Dequantize all shards in a MINT MLX model."""
    os.makedirs(output_dir, exist_ok=True)

    # Load config
    config_path = os.path.join(input_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    quant_config = config.get("quantization", {})
    default_bits = quant_config.get("bits", 4)
    default_gs = quant_config.get("group_size", 64)
    print(f"Default quantization: {default_bits}-bit, group_size={default_gs}")

    # Find safetensor shards
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".safetensors"))

    # Process each shard
    new_weight_map = {}
    total_dequantized = 0
    total_passthrough = 0

    for shard_file in shard_files:
        shard_path = os.path.join(input_dir, shard_file)
        print(f"\nProcessing {shard_file}...")
        tensors = st.load_file(shard_path)

        # Group tensors: find (weight, scales, biases) triples
        weight_names = [n for n in tensors if n.endswith(".weight")]
        scales_names = {n.replace(".scales", ".weight"): n for n in tensors if n.endswith(".scales")}
        biases_names = {n.replace(".biases", ".weight"): n for n in tensors if n.endswith(".biases")}

        output_tensors = {}

        for wname in weight_names:
            w = tensors[wname]

            if wname in scales_names and wname in biases_names:
                # Quantized tensor — dequantize
                s = tensors[scales_names[wname]]
                b = tensors[biases_names[wname]]
                layer = wname.replace(".weight", "")
                bits, gs = get_tensor_quant_config(wname, quant_config)

                dequant = dequantize_tensor(w, s, b, bits, gs)
                output_tensors[wname] = dequant
                total_dequantized += 1
                print(f"  Dequantized {wname}: {list(w.shape)} uint32 ({bits}-bit) -> {list(dequant.shape)} bf16")
            else:
                # Non-quantized tensor — pass through
                output_tensors[wname] = w
                total_passthrough += 1

        # Also include non-weight tensors (norms, biases, etc.) that aren't scales/biases
        for name, t in tensors.items():
            if not name.endswith(".weight") and not name.endswith(".scales") and not name.endswith(".biases"):
                output_tensors[name] = t
                total_passthrough += 1

        # Save output shard
        out_shard = os.path.join(output_dir, shard_file)
        st.save_file(output_tensors, out_shard)
        print(f"  Saved {len(output_tensors)} tensors to {shard_file}")

        for name in output_tensors:
            new_weight_map[name] = shard_file

    # Write new index
    if os.path.exists(index_path):
        with open(index_path) as f:
            orig_index = json.load(f)
        new_index = {
            "metadata": orig_index.get("metadata", {}),
            "weight_map": new_weight_map,
        }
        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(new_index, f, indent=2)

    # Copy config (remove quantization section)
    new_config = {k: v for k, v in config.items() if k != "quantization"}
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(new_config, f, indent=2)

    # Copy other files
    for fname in os.listdir(input_dir):
        if fname.endswith(".safetensors") or fname == "config.json" or fname == "model.safetensors.index.json":
            continue
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    print(f"\nDone! Dequantized {total_dequantized} tensors, passed through {total_passthrough}")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Dequantize MINT MLX model to BF16")
    parser.add_argument("--input", required=True, help="MINT MLX model directory")
    parser.add_argument("--output", required=True, help="Output BF16 model directory")
    args = parser.parse_args()
    dequantize_model(args.input, args.output)


if __name__ == "__main__":
    main()
