#!/usr/bin/env python3
"""Knapsack allocator: greedy MCKP solver with soft protection priors.

Reads rate-distortion curves from compute_rd_curves.py output and solves
the Multiple-Choice Knapsack Problem to find the optimal (bits, group_size)
per tensor under a byte budget.

Usage:
    python mint/allocator.py \
        --rd-curves results/model-rd-curves.json \
        --budget-gb 19.0 \
        --output results/model-allocation-19gb.json
"""

import argparse
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mint.allocator")

# SQNR floor: configs below this are vetoed (too much distortion)
SQNR_FLOOR_DB = 9.0

# Soft protection priors — multipliers on loss
DEFAULT_PRIORS = {
    "embedding": float("inf"),  # hard protect (bridge keeps at 16-bit)
    "lm_head": float("inf"),    # hard protect (bridge keeps at 16-bit)
    "layernorm": float("inf"),  # hard protect
    "router": float("inf"),     # hard protect (bridge keeps at 16-bit)
    "first_layer": 3.0,
    "last_layer": 2.0,
    "default": 1.0,
}

# Valid quantization configs to consider
VALID_CONFIGS = [
    (2, 32), (2, 64),
    (3, 64),
    (4, 32), (4, 64), (4, 128),
    (8, 64), (8, 128),
    (16, 0),
]


def estimate_size(num_params: int, bits: int, group_size: int) -> int:
    """Estimate storage size in bytes for a tensor at given (bits, group_size)."""
    if bits >= 16:
        return num_params * 2  # BF16/FP16

    weight_bytes = int(math.ceil(num_params * bits / 8))
    if group_size > 0:
        num_groups = int(math.ceil(num_params / group_size))
        scale_bytes = num_groups * 2  # float16 scales
        bias_bytes = num_groups * 2   # float16 biases/zeros
    else:
        scale_bytes = 0
        bias_bytes = 0
    return weight_bytes + scale_bytes + bias_bytes


def estimate_weight_bytes(num_params: int, bits: int) -> int:
    """Estimate weight storage bytes without scale/bias metadata."""
    if bits >= 16:
        return num_params * 2
    return int(math.ceil(num_params * bits / 8))


def parse_speed_bias(value: str) -> float:
    """Argparse type for speed bias."""
    speed_bias = float(value)
    if speed_bias < 0.0 or speed_bias > 1.0:
        raise argparse.ArgumentTypeError("--speed-bias must be between 0.0 and 1.0")
    return speed_bias


def extract_num_experts_per_tok(config: Dict) -> Optional[int]:
    """Read num_experts_per_tok from a model config dict if present."""
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and text_config.get("num_experts_per_tok") is not None:
        return text_config.get("num_experts_per_tok")
    return config.get("num_experts_per_tok")


def load_model_config(model_dir: Optional[str]) -> Dict:
    """Load minimal model metadata from config.json when a model dir is provided."""
    if not model_dir:
        return {}

    config_path = Path(model_dir) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    return {
        "num_experts_per_tok": extract_num_experts_per_tok(config),
    }


def compute_prior(
    tensor_name: str,
    layer_idx: Optional[int],
    total_layers: int,
) -> float:
    """Compute soft protection prior for a tensor."""
    name_lower = tensor_name.lower()

    # Hard protect: norms
    if any(kw in name_lower for kw in ["layernorm", "layer_norm", "rmsnorm"]):
        return float("inf")
    if re.search(r"model\.norm\.weight$", tensor_name):
        return float("inf")

    # Embeddings
    if "embed_tokens" in name_lower:
        return DEFAULT_PRIORS["embedding"]

    # LM head
    if "lm_head" in name_lower:
        return DEFAULT_PRIORS["lm_head"]

    # Router/gate (MoE)
    if name_lower.endswith("router.weight") or name_lower.endswith("gate.weight"):
        return DEFAULT_PRIORS["router"]

    # Positional priors
    if layer_idx is not None and total_layers > 0:
        if layer_idx == 0:
            return DEFAULT_PRIORS["first_layer"]
        if layer_idx == total_layers - 1:
            return DEFAULT_PRIORS["last_layer"]

    return DEFAULT_PRIORS["default"]


def _clamp_active_factor(value: float) -> float:
    """Clamp runtime activity to a valid fractional range."""
    return min(max(value, 1e-12), 1.0)


def _compress_active_factor(value: float) -> float:
    """Compress sparse activity discounts so speed bias remains stable."""
    return math.sqrt(_clamp_active_factor(value))


def classify_runtime(
    tensor_name: str,
    shape: List[int],
    model_config: Optional[Dict] = None,
    expert_members: Optional[Dict] = None,
) -> Tuple[float, Optional[str]]:
    """Infer a runtime family and active factor for allocator re-ranking."""
    model_config = model_config or {}

    if ".linear_attn." in tensor_name:
        return 1.0, "linear_attn"
    if ".self_attn." in tensor_name:
        return 1.0, "self_attn"
    if ".shared_expert." in tensor_name:
        return 1.0, "shared_expert"
    if ".mlp.experts." not in tensor_name:
        return 1.0, None

    num_experts_per_tok = model_config.get("num_experts_per_tok")
    if expert_members and tensor_name in expert_members:
        num_experts = len(expert_members[tensor_name])
    elif shape:
        num_experts = int(shape[0])
    else:
        num_experts = None

    if (
        num_experts_per_tok is None
        or num_experts is None
        or num_experts <= 0
        or num_experts_per_tok <= 0
    ):
        return 1.0, "moe_experts_fallback"

    active_factor = _clamp_active_factor(num_experts_per_tok / num_experts)
    return active_factor, "moe_experts"


def _group_moe_experts(rd_data: Dict, moe_aggregation: str = "weighted_mean") -> Tuple[Dict, Dict]:
    """Group MoE expert tensors by (layer, projection) for joint allocation.

    MLX's SwitchLinear requires all experts in a layer to share the same
    quantization. This groups experts and aggregates NRMSE across experts
    for each config, and sums params for size estimation.

    Args:
        rd_data: Rate-distortion data with per-tensor curves.
        moe_aggregation: How to aggregate NRMSE across experts in a group.
            "weighted_mean" (default): parameter-weighted mean, consistent
                with the additive global objective.
            "max": worst-case (maximum) across experts (conservative).

    Returns:
        grouped_tensors: dict mapping group_name -> merged tensor data
        expert_members: dict mapping group_name -> list of original tensor names
    """
    moe_pattern = re.compile(r"(.+)\.experts\.(\d+)\.(.+)")
    groups: Dict[str, List[Tuple[str, Dict]]] = {}
    non_expert = {}

    for name, tdata in rd_data["tensors"].items():
        m = moe_pattern.match(name)
        if m:
            prefix, expert_idx, suffix = m.groups()
            group_key = f"{prefix}.experts.*.{suffix}"
            groups.setdefault(group_key, []).append((name, tdata))
        else:
            non_expert[name] = tdata

    grouped_tensors = dict(non_expert)
    expert_members = {}

    for group_key, members in groups.items():
        # Merge expert group: aggregated NRMSE, min SQNR, sum params
        total_params = sum(m[1]["num_params"] for m in members)
        layer_idx = members[0][1]["layer_idx"]
        is_1d = members[0][1]["is_1d"]

        merged_rd = {}
        merged_sqnr = {}
        if not is_1d:
            all_cfg_keys = set()
            for _, tdata in members:
                all_cfg_keys.update(tdata["rd_curve"].keys())
                all_cfg_keys.update(tdata["sqnr"].keys())

            for cfg_key in all_cfg_keys:
                nrmses = [m[1]["rd_curve"].get(cfg_key, 1.0) for m in members if not m[1]["is_1d"]]
                sqnrs = [m[1]["sqnr"].get(cfg_key, 0.0) for m in members if not m[1]["is_1d"]]
                if nrmses:
                    if moe_aggregation == "weighted_mean":
                        # Parameter-weighted mean: consistent with additive global loss
                        params = [m[1]["num_params"] for m in members if not m[1]["is_1d"]]
                        total_p = sum(params)
                        merged_rd[cfg_key] = sum(n * p for n, p in zip(nrmses, params)) / total_p if total_p > 0 else max(nrmses)
                    else:
                        merged_rd[cfg_key] = max(nrmses)  # worst-case
                if sqnrs:
                    merged_sqnr[cfg_key] = min(sqnrs)  # worst-case

        grouped_tensors[group_key] = {
            "shape": list(members[0][1]["shape"]),
            "num_params": total_params,
            "layer_idx": layer_idx,
            "is_1d": is_1d,
            "rd_curve": merged_rd,
            "sqnr": merged_sqnr,
        }
        expert_members[group_key] = [m[0] for m in members]

    if expert_members:
        num_groups = len(expert_members)
        num_experts = sum(len(v) for v in expert_members.values())
        logger.info(f"Grouped {num_experts} expert tensors into {num_groups} groups")

    return grouped_tensors, expert_members


def build_tensor_specs(
    rd_data: Dict,
    total_layers: int,
    model_config: Optional[Dict] = None,
    moe_aggregation: str = "weighted_mean",
) -> Tuple[List[Dict], Dict]:
    """Build tensor specs for the allocator from RD curve data.

    Returns:
        specs: list of tensor specs for the allocator
        expert_members: dict mapping group names to member tensor names
    """
    # Group MoE experts for joint allocation
    grouped_tensors, expert_members = _group_moe_experts(rd_data, moe_aggregation)
    model_config = model_config or rd_data.get("model_config", {})

    specs = []

    for name, tdata in grouped_tensors.items():
        num_params = tdata["num_params"]
        layer_idx = tdata["layer_idx"]
        shape = list(tdata.get("shape", []))
        prior = compute_prior(name, layer_idx, total_layers)
        active_factor, runtime_family = classify_runtime(
            name,
            shape,
            model_config=model_config,
            expert_members=expert_members,
        )

        # Hard-protected tensors: only 16-bit
        if prior == float("inf") or tdata["is_1d"]:
            specs.append({
                "name": name,
                "shape": shape,
                "num_params": num_params,
                "valid_configs": [(16, 0)],
                "rd_curve": {(16, 0): 0.0},
                "prior": 1.0,  # doesn't matter, only one config
                "alpha": 1.0,
                "layer_idx": layer_idx,
                "active_factor": active_factor,
                "runtime_family": runtime_family,
            })
            continue

        # Build valid configs by SQNR veto
        valid_configs = []
        rd_curve = {}

        for bits, gs in VALID_CONFIGS:
            cfg_key = f"{bits}_{gs}"
            if bits >= 16:
                valid_configs.append((bits, gs))
                rd_curve[(bits, gs)] = 0.0
                continue

            sqnr = tdata["sqnr"].get(cfg_key, 0.0)
            nrmse = tdata["rd_curve"].get(cfg_key, 1.0)

            if sqnr >= SQNR_FLOOR_DB:
                valid_configs.append((bits, gs))
                rd_curve[(bits, gs)] = nrmse
            # else: vetoed by SQNR floor

        # Must have at least one valid config
        if not valid_configs:
            valid_configs = [(16, 0)]
            rd_curve = {(16, 0): 0.0}

        specs.append({
            "name": name,
            "shape": shape,
            "num_params": num_params,
            "valid_configs": valid_configs,
            "rd_curve": rd_curve,
            "prior": prior,
            "alpha": 1.0,
            "layer_idx": layer_idx,
            "active_factor": active_factor,
            "runtime_family": runtime_family,
        })

    return specs, expert_members


def compute_min_safe_size(tensor_specs: List[Dict]) -> int:
    """Compute the minimum model size (bytes) with all tensors at cheapest SQNR-safe config."""
    def config_sort_key(cfg):
        return (cfg[0], -cfg[1])

    total = 0
    for spec in tensor_specs:
        configs = sorted(spec["valid_configs"], key=config_sort_key)
        lowest = configs[0]
        total += estimate_size(spec["num_params"], lowest[0], lowest[1])
    return total


def allocate_greedy(
    tensor_specs: List[Dict],
    budget_bytes: int,
    expert_members: Optional[Dict] = None,
    speed_bias: float = 0.0,
) -> Dict:
    """Greedy efficiency-ordered MCKP allocator.

    1. Start all tensors at lowest valid (bits, group_size)
    2. Build upgrade options sorted by efficiency = loss_reduction / effective_cost
    3. Greedily apply best upgrades until budget exhausted
    """
    t0 = time.time()

    def config_sort_key(cfg):
        return (cfg[0], -cfg[1])

    # Initialize at cheapest config
    current = {}
    current_size = 0
    spec_by_name = {spec["name"]: spec for spec in tensor_specs}

    for spec in tensor_specs:
        name = spec["name"]
        configs = sorted(spec["valid_configs"], key=config_sort_key)
        lowest = configs[0]
        current[name] = lowest
        current_size += estimate_size(spec["num_params"], lowest[0], lowest[1])

    min_safe_bytes = current_size
    logger.info(f"Minimum safe size: {min_safe_bytes / (1024**3):.2f} GB (SQNR floor {SQNR_FLOOR_DB} dB)")
    logger.info(f"Initial allocation: {current_size / (1024**3):.2f} GB (all at minimum)")

    # Build upgrade options
    upgrades = []
    for spec in tensor_specs:
        name = spec["name"]
        configs = sorted(spec["valid_configs"], key=config_sort_key)
        rd_curve = spec["rd_curve"]
        prior = spec["prior"]
        alpha = spec["alpha"]
        active_factor = spec["active_factor"]
        runtime_factor = _compress_active_factor(active_factor)

        for i in range(len(configs) - 1):
            lo_cfg = configs[i]
            for j in range(i + 1, len(configs)):
                hi_cfg = configs[j]

                lo_loss = prior * alpha * rd_curve.get(lo_cfg, 0.0)
                hi_loss = prior * alpha * rd_curve.get(hi_cfg, 0.0)
                loss_reduction = lo_loss - hi_loss

                if loss_reduction <= 0:
                    continue

                lo_size = estimate_size(spec["num_params"], lo_cfg[0], lo_cfg[1])
                hi_size = estimate_size(spec["num_params"], hi_cfg[0], hi_cfg[1])
                size_increase = hi_size - lo_size
                lo_weight = estimate_weight_bytes(spec["num_params"], lo_cfg[0])
                hi_weight = estimate_weight_bytes(spec["num_params"], hi_cfg[0])
                weight_increase = max(hi_weight - lo_weight, 0)
                if weight_increase > 0:
                    runtime_increase = min(
                        size_increase,
                        runtime_factor * size_increase,
                    )
                else:
                    runtime_increase = 0.0

                effective_cost = size_increase + speed_bias * runtime_increase

                if effective_cost <= 0:
                    efficiency = float("inf")
                else:
                    efficiency = loss_reduction / effective_cost

                upgrades.append({
                    "name": name,
                    "from_cfg": lo_cfg,
                    "to_cfg": hi_cfg,
                    "loss_reduction": loss_reduction,
                    "size_increase": size_increase,
                    "runtime_increase": runtime_increase,
                    "effective_cost": effective_cost,
                    "efficiency": efficiency,
                })

    upgrades.sort(key=lambda u: u["efficiency"], reverse=True)
    logger.info(f"Built {len(upgrades)} upgrade options")

    # Greedy application
    applied = 0
    for upgrade in upgrades:
        name = upgrade["name"]
        if current[name] != upgrade["from_cfg"]:
            continue

        spec = spec_by_name[name]
        old_size = estimate_size(spec["num_params"], upgrade["from_cfg"][0], upgrade["from_cfg"][1])
        new_size = estimate_size(spec["num_params"], upgrade["to_cfg"][0], upgrade["to_cfg"][1])
        delta = new_size - old_size

        if current_size + delta <= budget_bytes:
            current[name] = upgrade["to_cfg"]
            current_size += delta
            applied += 1

    elapsed_ms = (time.time() - t0) * 1000
    logger.info(
        f"Applied {applied} upgrades in {elapsed_ms:.0f}ms, "
        f"final size: {current_size / (1024**3):.2f} GB / "
        f"{budget_bytes / (1024**3):.2f} GB budget "
        f"({current_size / budget_bytes:.1%} util)"
    )

    # Build result — expand expert groups back to individual tensors
    if expert_members is None:
        expert_members = {}
    allocations = {}
    total_loss = 0.0
    bits_dist = {}
    runtime_proxy_bytes = 0.0

    for spec in tensor_specs:
        name = spec["name"]
        bits, group_size = current[name]
        nrmse = spec["rd_curve"].get((bits, group_size), 0.0)
        loss = spec["prior"] * spec["alpha"] * nrmse
        size = estimate_size(spec["num_params"], bits, group_size)
        runtime_proxy_bytes += spec["active_factor"] * estimate_weight_bytes(spec["num_params"], bits)

        if name in expert_members:
            # Expand group back to individual expert tensors
            member_names = expert_members[name]
            per_expert_params = spec["num_params"] // len(member_names)
            per_expert_size = estimate_size(per_expert_params, bits, group_size)
            for member_name in member_names:
                alloc = {
                    "bits": bits,
                    "group_size": group_size,
                    "size_bytes": per_expert_size,
                    "nrmse": nrmse,
                    "loss": loss / len(member_names),
                    "prior": spec["prior"],
                    "num_params": per_expert_params,
                    "layer_idx": spec["layer_idx"],
                }
                if spec["runtime_family"] is not None:
                    alloc["active_factor"] = spec["active_factor"]
                    alloc["runtime_family"] = spec["runtime_family"]
                allocations[member_name] = alloc
        else:
            alloc = {
                "bits": bits,
                "group_size": group_size,
                "size_bytes": size,
                "nrmse": nrmse,
                "loss": loss,
                "prior": spec["prior"],
                "num_params": spec["num_params"],
                "layer_idx": spec["layer_idx"],
            }
            if spec["runtime_family"] is not None:
                alloc["active_factor"] = spec["active_factor"]
                alloc["runtime_family"] = spec["runtime_family"]
            allocations[name] = alloc
        total_loss += loss
        bits_dist[bits] = bits_dist.get(bits, 0) + spec["num_params"]

    total_params = sum(spec["num_params"] for spec in tensor_specs)
    avg_bits = sum(b * c for b, c in bits_dist.items()) / total_params if total_params else 0

    return {
        "budget_bytes": budget_bytes,
        "budget_gb": budget_bytes / (1024**3),
        "total_size_bytes": current_size,
        "total_size_gb": current_size / (1024**3),
        "runtime_proxy_bytes": runtime_proxy_bytes,
        "runtime_proxy_gb": runtime_proxy_bytes / (1024**3),
        "min_safe_size_bytes": min_safe_bytes,
        "min_safe_size_gb": min_safe_bytes / (1024**3),
        "budget_utilization": current_size / budget_bytes if budget_bytes else 0,
        "total_loss": total_loss,
        "total_params": total_params,
        "average_bits": avg_bits,
        "speed_bias": speed_bias,
        "objective": "size_only" if speed_bias == 0.0 else "size_runtime_tax",
        "bits_distribution": {
            str(b): {
                "params": c,
                "percentage": c / total_params * 100 if total_params else 0,
            }
            for b, c in sorted(bits_dist.items())
        },
        "solver": "greedy",
        "solver_runtime_ms": elapsed_ms,
        "sqnr_floor_db": SQNR_FLOOR_DB,
        "num_tensors": len(allocations),
        "allocations": allocations,
    }


def main():
    parser = argparse.ArgumentParser(description="MINT knapsack allocator")
    parser.add_argument("--rd-curves", required=True, help="RD curves JSON from compute_rd_curves.py")
    parser.add_argument("--output", required=True, help="Output allocation JSON")
    parser.add_argument(
        "--model-dir",
        help="Optional HF model dir. If set, allocator reads config.json for runtime metadata",
    )
    parser.add_argument(
        "--speed-bias",
        type=parse_speed_bias,
        default=0.0,
        help="Blend between size-only and runtime-aware upgrade ranking [0.0, 1.0]",
    )

    budget_group = parser.add_mutually_exclusive_group(required=True)
    budget_group.add_argument("--budget-gb", type=float, help="Target budget in GB")
    budget_group.add_argument(
        "--min-safe", action="store_true",
        help="Produce the smallest possible model that respects the SQNR safety floor"
    )
    args = parser.parse_args()

    rd_data = json.load(open(args.rd_curves))
    total_layers = rd_data.get("total_layers", 48)
    model_config = load_model_config(args.model_dir)

    logger.info(f"Loaded RD curves: {rd_data['num_2d_tensors']} 2D + {rd_data['num_1d_tensors']} 1D tensors")
    if args.model_dir:
        logger.info(
            "Loaded model config from %s (num_experts_per_tok=%s)",
            args.model_dir,
            model_config.get("num_experts_per_tok"),
        )

    specs, expert_members = build_tensor_specs(
        rd_data,
        total_layers,
        model_config=model_config,
    )
    min_safe_bytes = compute_min_safe_size(specs)
    min_safe_gb = min_safe_bytes / (1024**3)

    if args.min_safe:
        budget_bytes = min_safe_bytes
        logger.info(f"--min-safe: targeting minimum safe size {min_safe_gb:.2f} GB")
    else:
        budget_bytes = int(args.budget_gb * 1024**3)
        if budget_bytes < min_safe_bytes:
            logger.warning(
                f"Requested budget {args.budget_gb:.2f} GB is below the minimum safe "
                f"size of {min_safe_gb:.2f} GB (SQNR floor {SQNR_FLOOR_DB} dB). "
                f"Using minimum safe size instead."
            )
            budget_bytes = min_safe_bytes

    result = allocate_greedy(
        specs,
        budget_bytes,
        expert_members,
        speed_bias=args.speed_bias,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"MINT Allocation Summary")
    print(f"{'='*60}")
    print(f"Min safe:   {result['min_safe_size_gb']:.2f} GB (SQNR floor {SQNR_FLOOR_DB} dB)")
    print(f"Budget:     {result['budget_gb']:.2f} GB")
    print(f"Allocated:  {result['total_size_gb']:.2f} GB ({result['budget_utilization']:.1%})")
    print(f"Objective:  {result['objective']} (speed_bias={result['speed_bias']:.2f})")
    print(f"Runtime:    {result['runtime_proxy_gb']:.2f} GB active-weight proxy")
    print(f"Avg bits:   {result['average_bits']:.2f}")
    print(f"Total loss: {result['total_loss']:.6f}")
    print(f"Solver:     {result['solver']} ({result['solver_runtime_ms']:.0f}ms)")
    print(f"\nBit distribution:")
    for b, info in sorted(result["bits_distribution"].items(), key=lambda x: int(x[0])):
        print(f"  {b:>2s}-bit: {info['params']:>15,} params ({info['percentage']:.1f}%)")
    print(f"{'='*60}\n")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved allocation to {args.output}")


if __name__ == "__main__":
    main()
