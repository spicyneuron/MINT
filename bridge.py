"""MINT bridge: manifest -> MLX quant predicate with per-tensor group_size.

Reads the MINT allocation's per-tensor (bits, group_size) from the manifest
and returns the exact dict MLX expects from a quant_predicate function.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from tensor_aliases import tensor_aliases

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

            cfg_dict = {"bits": bits, "group_size": group_size}
            for alias in tensor_aliases(tensor_name):
                lookup[alias] = cfg_dict

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

    for switch_name, cfgs in moe_groups.items():
        # Mode of bits across experts
        bits_counter = Counter(c["bits"] for c in cfgs)
        mode_bits = bits_counter.most_common(1)[0][0]
        # Mode of group_size for that bit level
        gs_counter = Counter(c["group_size"] for c in cfgs if c["bits"] == mode_bits)
        mode_gs = gs_counter.most_common(1)[0][0]
        cfg_dict = {"bits": mode_bits, "group_size": mode_gs}
        for alias in tensor_aliases(switch_name):
            lookup[alias] = cfg_dict
        logger.debug(f"MoE aggregate: {switch_name} -> {mode_bits}-bit g{mode_gs}")

    if moe_groups:
        logger.info(f"Mapped {len(moe_groups)} MoE expert groups to SwitchLinear modules")

    return lookup


def create_knapsack_predicate(manifest: Dict[str, Any]):
    """Create MLX quant_predicate with per-tensor (bits, group_size) from MINT allocation."""
    lookup = build_module_lookup(manifest)
    source_bits = manifest.get("config", {}).get("source_bits", 16)

    # Summary stats
    from collections import Counter
    bits_counter = Counter()
    for shard_data in manifest["shards"].values():
        for tensor_name, info in shard_data["tensors"].items():
            bits_counter[info["decision"]["bits"]] += 1
    total = sum(bits_counter.values())
    logger.info(f"MINT predicate (source: {source_bits}-bit):")
    for b in sorted(bits_counter):
        logger.info(f"  {b:2d}-bit: {bits_counter[b]} tensors ({bits_counter[b]/total*100:.1f}%)")

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
