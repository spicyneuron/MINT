"""MINT bridge: manifest -> MLX quant predicate with per-tensor group_size.

Reads the MINT allocation's per-tensor (bits, group_size) from the manifest
and returns the exact dict MLX expects from a quant_predicate function.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("mint.bridge")


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    with open(manifest_path) as f:
        return json.load(f)


def build_module_lookup(manifest: Dict[str, Any]) -> Dict[str, Dict]:
    """Build lookup from MLX module path to (bits, group_size) dict."""
    lookup = {}

    for shard_data in manifest["shards"].values():
        for tensor_name, info in shard_data["tensors"].items():
            decision = info["decision"]
            bits = decision["bits"]
            group_size = decision.get("group_size", 128 if bits == 4 else 64 if bits in (2, 3, 8) else 0)

            cfg = {"bits": bits, "group_size": group_size}
            if tensor_name.endswith(".weight"):
                module_name = tensor_name.rsplit(".", 1)[0]
                lookup[module_name] = cfg
                lookup[tensor_name] = cfg
            elif tensor_name.endswith(".bias"):
                module_name = tensor_name.rsplit(".", 1)[0]
                lookup[tensor_name] = cfg
            else:
                module_name = tensor_name
                lookup[module_name] = cfg
                lookup[tensor_name] = cfg

            # Handle fused gate_up_proj -> separate gate_proj + up_proj
            # BF16 models store fused experts.gate_up_proj but MLX splits
            # them into experts.gate_proj and experts.up_proj during loading
            if "gate_up_proj" in module_name:
                gate_name = module_name.replace("gate_up_proj", "gate_proj")
                up_name = module_name.replace("gate_up_proj", "up_proj")
                lookup[gate_name] = cfg
                lookup[up_name] = cfg
                if tensor_name != module_name:
                    lookup[tensor_name.replace("gate_up_proj", "gate_proj")] = cfg
                    lookup[tensor_name.replace("gate_up_proj", "up_proj")] = cfg

    # Aggregate MoE experts -> SwitchLinear
    from collections import Counter
    moe_pattern = re.compile(r"(.+)\.experts\.(\d+)\.(.+)")
    moe_groups: Dict[str, list] = {}

    for module_name, cfg in list(lookup.items()):
        m = moe_pattern.match(module_name)
        if m:
            prefix, expert_idx, proj = m.groups()
            switch_name = f"{prefix}.switch_mlp.{proj}"
            moe_groups.setdefault(switch_name, []).append(cfg)

    # Mixtral-style w1/w2/w3 -> gate_proj/down_proj/up_proj mapping
    PROJ_ALIASES = {
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
    }

    for switch_name, cfgs in moe_groups.items():
        # Mode of bits across experts
        bits_counter = Counter(c["bits"] for c in cfgs)
        mode_bits = bits_counter.most_common(1)[0][0]
        # Mode of group_size for that bit level
        gs_counter = Counter(c["group_size"] for c in cfgs if c["bits"] == mode_bits)
        mode_gs = gs_counter.most_common(1)[0][0]
        cfg_dict = {"bits": mode_bits, "group_size": mode_gs}
        lookup[switch_name] = cfg_dict
        logger.debug(f"MoE aggregate: {switch_name} -> {mode_bits}-bit g{mode_gs}")

        # Add aliases: switch_mlp.w1 -> switch_mlp.gate_proj, etc.
        for old_name, new_name in PROJ_ALIASES.items():
            if switch_name.endswith(f".{old_name}"):
                alias = switch_name[:-len(old_name)] + new_name
                lookup[alias] = cfg_dict
                logger.debug(f"  alias: {alias}")
            elif switch_name.endswith(f".{new_name}"):
                alias = switch_name[:-len(new_name)] + old_name
                lookup[alias] = cfg_dict

    if moe_groups:
        logger.info(f"Mapped {len(moe_groups)} MoE expert groups to SwitchLinear modules")

    # Handle packed MoE expert tensors (no individual expert index).
    # Some models store experts as 3D tensors: experts.gate_up_proj [num_experts, d_in, d_out]
    # MLX loads these as SwitchLinear: switch_mlp.gate_proj, switch_mlp.up_proj
    packed_expert_pattern = re.compile(r"(.+)\.experts\.(gate_up_proj|down_proj)$")
    packed_count = 0
    packed_entries = {}
    for key, cfg in list(lookup.items()):
        m = packed_expert_pattern.match(key)
        if m:
            prefix, proj = m.groups()
            if proj == "gate_up_proj":
                packed_entries[f"{prefix}.switch_mlp.gate_proj"] = cfg
                packed_entries[f"{prefix}.switch_mlp.gate_proj.weight"] = cfg
                packed_entries[f"{prefix}.switch_mlp.up_proj"] = cfg
                packed_entries[f"{prefix}.switch_mlp.up_proj.weight"] = cfg
                packed_count += 2
            elif proj == "down_proj":
                packed_entries[f"{prefix}.switch_mlp.down_proj"] = cfg
                packed_entries[f"{prefix}.switch_mlp.down_proj.weight"] = cfg
                packed_count += 1
    if packed_entries:
        lookup.update(packed_entries)
        logger.info(f"Mapped {packed_count} packed expert tensors to SwitchLinear modules")

    # Handle MLX sanitize remapping: some models remap safetensor keys during load.
    # E.g., Qwen3.5 MoE: "model.language_model.X" -> "language_model.model.X"
    # Add remapped variants so the predicate can match either naming convention.
    remapped = {}
    for key, cfg in lookup.items():
        if key.startswith("model.language_model."):
            alt = "language_model.model." + key[len("model.language_model."):]
            if alt not in lookup:
                remapped[alt] = cfg
    if remapped:
        lookup.update(remapped)
        logger.info(f"Added {len(remapped)} MLX-sanitize remapped keys (model.language_model -> language_model.model)")

    return lookup


def create_knapsack_predicate(manifest: Dict[str, Any]):
    """Create MLX quant_predicate with per-tensor (bits, group_size) from MINT allocation."""
    lookup = build_module_lookup(manifest)
    source_bits = manifest.get("config", {}).get("source_bits", 16)

    # Summary stats
    from collections import Counter
    tensor_counter = Counter()
    param_counter = Counter()
    for shard_data in manifest["shards"].values():
        for tensor_name, info in shard_data["tensors"].items():
            bits = info["decision"]["bits"]
            tensor_counter[bits] += 1
            param_counter[bits] += info.get("num_params", 0)
    total_tensors = sum(tensor_counter.values())
    total_params = sum(param_counter.values())
    logger.info(f"MINT predicate (source: {source_bits}-bit)")
    logger.info("Manifest tensor counts:")
    for b in sorted(tensor_counter):
        logger.info(f"  {b:2d}-bit: {tensor_counter[b]} tensors ({tensor_counter[b]/total_tensors*100:.1f}%)")
    if total_params:
        logger.info("Manifest parameter counts:")
        for b in sorted(param_counter):
            logger.info(f"  {b:2d}-bit: {param_counter[b]:,} params ({param_counter[b]/total_params*100:.1f}%)")

    def knapsack_predicate(path: str, module: Any, config: Any = None):
        """MLX quant predicate with per-tensor bits and group_size."""
        name = path

        # Never quantize 1D modules
        if hasattr(module, "weight"):
            w = module.weight
            if hasattr(w, "shape") and len(w.shape) <= 1:
                return False

        name_lower = name.lower()

        # Skip norms, embeddings, heads, routers
        if any(kw in name_lower for kw in ["layernorm", "layer_norm", "rmsnorm"]):
            return False
        if "embed_tokens" in name_lower or "lm_head" in name_lower:
            return False
        if name_lower.endswith("router") or name_lower.endswith("gate"):
            return False
        if re.search(r"model\.norm$", name):
            return False

        # Vision components
        vision_patterns = [
            "vision_model", "visual_encoder", "vision_encoder",
            "multi_modal_projector", "aligner", "image_newline",
        ]
        if any(p in name_lower for p in vision_patterns):
            return False

        # Look up MINT decision — try multiple naming conventions
        cfg = lookup.get(name)
        if cfg is None:
            cfg = lookup.get(name + ".weight")
        if cfg is None:
            cfg = lookup.get("language_model." + name)
        if cfg is None:
            cfg = lookup.get("language_model." + name + ".weight")
        if cfg is None:
            cfg = lookup.get("model." + name)
        if cfg is None:
            cfg = lookup.get("model." + name + ".weight")
        # Reverse MLX sanitize: language_model.model.X -> model.language_model.X
        if cfg is None and name.startswith("language_model.model."):
            orig = "model.language_model." + name[len("language_model.model."):]
            cfg = lookup.get(orig)
            if cfg is None:
                cfg = lookup.get(orig + ".weight")

        if cfg is None:
            logger.warning(f"No manifest entry for '{name}', using default 4-bit")
            return True  # default 4-bit

        bits = cfg["bits"]
        group_size = cfg["group_size"]

        if bits >= 16:
            return False

        if bits == 8:
            if source_bits == 8:
                return False
            return {"bits": 8, "group_size": group_size}

        if bits == 2:
            return {"bits": 2, "group_size": group_size}

        if bits == 3:
            return {"bits": 3, "group_size": group_size}

        # 4-bit with explicit per-tensor group_size
        return {"bits": 4, "group_size": group_size}

    return knapsack_predicate
