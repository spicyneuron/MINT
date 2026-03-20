#!/usr/bin/env python3
"""Pareto frontier allocator with MLX-live text scope filtering."""

import argparse
import importlib
import inspect
import json
import logging
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from allocator import SQNR_FLOOR_DB, build_tensor_specs, estimate_size
from tensor_aliases import tensor_aliases

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mint.allocator_frontier")

_GB = 1024 ** 3
_PACKED_EXPERT_PATTERN = re.compile(r".+\.experts\.(gate_up_proj|down_proj)$")
_GROUPED_EXPERT_PATTERN = re.compile(r".+\.experts\.\*\..+")
_MOE_TOP_K_KEYS = (
    "num_experts_per_tok",
    "num_experts_per_token",
    "num_local_experts_per_tok",
    "moe_top_k",
    "num_selected_experts",
)


def _config_views(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    yield config
    for key in ("text_config", "language_config", "llm_config"):
        value = config.get(key)
        if isinstance(value, dict):
            yield value


def _find_first_number(config: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[float]:
    for view in _config_views(config):
        for key in keys:
            value = view.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
    return None


def _import_mlx_load():
    errors = []
    for module_name in ("mlx_lm", "mlx_lm.utils"):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")
            continue

        load_fn = getattr(module, "load", None)
        if callable(load_fn):
            return load_fn
        errors.append(f"{module_name}: missing load()")

    joined = "; ".join(errors) if errors else "no mlx_lm modules found"
    raise ImportError(f"MLX LM is required for live-module discovery ({joined})")


def collect_live_module_paths(model: Any) -> Set[str]:
    """Collect instantiated MLX module paths from a loaded model."""
    names = set()

    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        for item in named_modules():
            if isinstance(item, tuple) and item:
                name = item[0]
                if isinstance(name, str) and name:
                    names.add(name)
        if names:
            return names

    apply_to_modules = getattr(model, "apply_to_modules", None)
    if callable(apply_to_modules):
        def collect(*args):
            for arg in args:
                if isinstance(arg, str) and arg:
                    names.add(arg)

        apply_to_modules(collect)
        if names:
            return names

    raise RuntimeError("Loaded MLX model does not expose named_modules() or apply_to_modules()")


def load_live_module_paths(model_dir: Path) -> Set[str]:
    """Load the model through MLX LM and return instantiated module paths."""
    load_fn = _import_mlx_load()
    kwargs = {}
    try:
        signature = inspect.signature(load_fn)
    except (TypeError, ValueError):
        signature = None

    if signature and "lazy" in signature.parameters:
        kwargs["lazy"] = True

    loaded = load_fn(str(model_dir), **kwargs)
    model = loaded[0] if isinstance(loaded, tuple) else loaded
    live_modules = collect_live_module_paths(model)
    logger.info(f"Loaded {len(live_modules)} live MLX modules from {model_dir}")
    return live_modules


def filter_rd_by_live_modules(
    rd_data: Dict[str, Any],
    live_modules: Set[str],
) -> Tuple[Dict[str, Any], Dict[str, int], int]:
    """Keep only tensors whose aliases intersect the live MLX module set."""
    filtered_tensors = {}
    excluded_reason_counts = Counter()

    for name, tensor_data in rd_data["tensors"].items():
        if tensor_aliases(name).intersection(live_modules):
            filtered_tensors[name] = tensor_data
            continue
        excluded_reason_counts["not_live_in_mlxlm"] += 1

    if not filtered_tensors:
        raise ValueError("No RD tensors survived MLX live-module filtering")

    filtered = dict(rd_data)
    filtered["tensors"] = filtered_tensors
    filtered["num_1d_tensors"] = sum(1 for t in filtered_tensors.values() if t["is_1d"])
    filtered["num_2d_tensors"] = sum(1 for t in filtered_tensors.values() if not t["is_1d"])
    return filtered, dict(excluded_reason_counts), sum(excluded_reason_counts.values())


def _has_moe_specs(tensor_specs: List[Dict[str, Any]]) -> bool:
    for spec in tensor_specs:
        if _GROUPED_EXPERT_PATTERN.match(spec["name"]) or _PACKED_EXPERT_PATTERN.match(spec["name"]):
            return True
    return False


def resolve_moe_top_k(config: Dict[str, Any], tensor_specs: List[Dict[str, Any]]) -> float:
    """Resolve num_experts_per_tok only when the filtered scope includes MoE tensors."""
    if not _has_moe_specs(tensor_specs):
        return 1.0

    value = _find_first_number(config, _MOE_TOP_K_KEYS)
    if value is None:
        keys = ", ".join(_MOE_TOP_K_KEYS)
        raise ValueError(f"MoE tensors are in scope but config.json is missing one of: {keys}")
    return value


def infer_num_experts(
    spec_name: str,
    expert_members: Dict[str, List[str]],
    tensor_meta: Dict[str, Dict[str, Any]],
) -> Optional[int]:
    if spec_name in expert_members:
        return len(expert_members[spec_name])

    if not _PACKED_EXPERT_PATTERN.match(spec_name):
        return None

    tensor_data = tensor_meta.get(spec_name)
    shape = tensor_data.get("shape", []) if tensor_data else []
    if shape:
        return int(shape[0])
    return None


def compute_active_factor(
    spec_name: str,
    expert_members: Dict[str, List[str]],
    tensor_meta: Dict[str, Dict[str, Any]],
    moe_top_k: float,
) -> float:
    num_experts = infer_num_experts(spec_name, expert_members, tensor_meta)
    if not num_experts:
        return 1.0
    return min(moe_top_k, float(num_experts)) / float(num_experts)


def build_objective_tables(
    tensor_specs: List[Dict[str, Any]],
    expert_members: Dict[str, List[str]],
    tensor_meta: Dict[str, Dict[str, Any]],
    moe_top_k: float,
) -> List[Dict[str, Any]]:
    """Build per-tensor raw + normalized objectives for frontier search."""
    tables = []

    for spec in tensor_specs:
        active_factor = compute_active_factor(spec["name"], expert_members, tensor_meta, moe_top_k)
        configs = []

        for bits, group_size in spec["valid_configs"]:
            loss = spec["prior"] * spec["alpha"] * spec["rd_curve"].get((bits, group_size), 0.0)
            size = estimate_size(spec["num_params"], bits, group_size)
            runtime = size * active_factor
            configs.append({
                "cfg": (bits, group_size),
                "loss": loss,
                "size": size,
                "runtime": runtime,
            })

        losses = [cfg["loss"] for cfg in configs]
        sizes = [cfg["size"] for cfg in configs]
        runtimes = [cfg["runtime"] for cfg in configs]

        loss_min, loss_max = min(losses), max(losses)
        size_min, size_max = min(sizes), max(sizes)
        runtime_min, runtime_max = min(runtimes), max(runtimes)

        for cfg in configs:
            cfg["norm_loss"] = 0.0 if loss_max == loss_min else (cfg["loss"] - loss_min) / (loss_max - loss_min)
            cfg["norm_size"] = 0.0 if size_max == size_min else (cfg["size"] - size_min) / (size_max - size_min)
            cfg["norm_runtime"] = 0.0 if runtime_max == runtime_min else (cfg["runtime"] - runtime_min) / (runtime_max - runtime_min)
            cfg["active_factor"] = active_factor

        tables.append({
            "name": spec["name"],
            "configs": configs,
        })

    return tables


def generate_weight_triples(step: float = 0.05) -> Iterable[Tuple[float, float, float]]:
    units = int(round(1.0 / step))
    for q_units in range(units + 1):
        for m_units in range(units - q_units + 1):
            s_units = units - q_units - m_units
            yield (q_units / units, m_units / units, s_units / units)


def select_local_configs(
    objective_tables: List[Dict[str, Any]],
    weights: Tuple[float, float, float],
) -> Dict[str, Dict[str, Any]]:
    wq, wm, ws = weights
    selection = {}

    for table in objective_tables:
        best = min(
            table["configs"],
            key=lambda cfg: (
                wq * cfg["norm_loss"] + wm * cfg["norm_size"] + ws * cfg["norm_runtime"],
                cfg["runtime"],
                cfg["size"],
                cfg["loss"],
                cfg["cfg"][0],
                cfg["cfg"][1],
            ),
        )
        selection[table["name"]] = best

    return selection


def selection_signature(selection: Dict[str, Dict[str, Any]]) -> Tuple[Tuple[str, int, int], ...]:
    return tuple(
        (name, cfg["cfg"][0], cfg["cfg"][1])
        for name, cfg in sorted(selection.items())
    )


def expand_allocations(
    tensor_specs: List[Dict[str, Any]],
    selection: Dict[str, Dict[str, Any]],
    expert_members: Dict[str, List[str]],
    tensor_meta: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, int], float, int, float, float]:
    allocations = {}
    bits_dist = Counter()
    total_loss = 0.0
    total_size = 0
    total_runtime = 0.0

    for spec in tensor_specs:
        chosen = selection[spec["name"]]
        bits, group_size = chosen["cfg"]
        nrmse = spec["rd_curve"].get((bits, group_size), 0.0)
        total_loss += chosen["loss"]
        total_size += chosen["size"]
        total_runtime += chosen["runtime"]
        bits_dist[bits] += spec["num_params"]

        if spec["name"] in expert_members:
            member_names = expert_members[spec["name"]]
            total_member_params = sum(tensor_meta[name]["num_params"] for name in member_names)
            for member_name in member_names:
                member_params = tensor_meta[member_name]["num_params"]
                member_weight = member_params / total_member_params if total_member_params else 0.0
                allocations[member_name] = {
                    "bits": bits,
                    "group_size": group_size,
                    "size_bytes": estimate_size(member_params, bits, group_size),
                    "nrmse": nrmse,
                    "loss": chosen["loss"] * member_weight,
                    "prior": spec["prior"],
                    "num_params": member_params,
                    "layer_idx": tensor_meta[member_name]["layer_idx"],
                }
        else:
            allocations[spec["name"]] = {
                "bits": bits,
                "group_size": group_size,
                "size_bytes": chosen["size"],
                "nrmse": nrmse,
                "loss": chosen["loss"],
                "prior": spec["prior"],
                "num_params": spec["num_params"],
                "layer_idx": spec["layer_idx"],
            }

    total_params = sum(bits_dist.values())
    average_bits = sum(bits * params for bits, params in bits_dist.items()) / total_params if total_params else 0.0
    return allocations, dict(bits_dist), total_loss, total_size, total_runtime, average_bits


def frontier_summary(candidate: Dict[str, Any], selected_signature: Tuple[Tuple[str, int, int], ...]) -> Dict[str, Any]:
    return {
        "selected": candidate["signature"] == selected_signature,
        "total_loss": candidate["total_loss"],
        "total_size_bytes": candidate["total_size_bytes"],
        "total_size_gb": candidate["total_size_bytes"] / _GB,
        "runtime_proxy_bytes": candidate["runtime_proxy_bytes"],
        "runtime_proxy_gb": candidate["runtime_proxy_bytes"] / _GB,
        "average_bits": candidate["average_bits"],
        "num_tensors": candidate["num_tensors"],
    }


def pareto_prune(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    frontier = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            no_worse = (
                other["total_loss"] <= candidate["total_loss"]
                and other["total_size_bytes"] <= candidate["total_size_bytes"]
                and other["runtime_proxy_bytes"] <= candidate["runtime_proxy_bytes"]
            )
            strictly_better = (
                other["total_loss"] < candidate["total_loss"]
                or other["total_size_bytes"] < candidate["total_size_bytes"]
                or other["runtime_proxy_bytes"] < candidate["runtime_proxy_bytes"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)

    frontier.sort(
        key=lambda candidate: (
            candidate["total_size_bytes"],
            candidate["runtime_proxy_bytes"],
            candidate["total_loss"],
        )
    )
    return frontier


def choose_closest_to_ideal(frontier: List[Dict[str, Any]]) -> Dict[str, Any]:
    mins = {
        "total_loss": min(candidate["total_loss"] for candidate in frontier),
        "total_size_bytes": min(candidate["total_size_bytes"] for candidate in frontier),
        "runtime_proxy_bytes": min(candidate["runtime_proxy_bytes"] for candidate in frontier),
    }
    maxs = {
        "total_loss": max(candidate["total_loss"] for candidate in frontier),
        "total_size_bytes": max(candidate["total_size_bytes"] for candidate in frontier),
        "runtime_proxy_bytes": max(candidate["runtime_proxy_bytes"] for candidate in frontier),
    }

    def distance(candidate: Dict[str, Any]) -> float:
        total = 0.0
        for key in ("total_loss", "total_size_bytes", "runtime_proxy_bytes"):
            span = maxs[key] - mins[key]
            if span == 0:
                continue
            total += ((candidate[key] - mins[key]) / span) ** 2
        return math.sqrt(total)

    return min(
        frontier,
        key=lambda candidate: (
            distance(candidate),
            candidate["runtime_proxy_bytes"],
            candidate["total_size_bytes"],
            candidate["total_loss"],
        ),
    )


def search_frontier(
    tensor_specs: List[Dict[str, Any]],
    objective_tables: List[Dict[str, Any]],
    expert_members: Dict[str, List[str]],
    tensor_meta: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    deduped = {}

    for weights in generate_weight_triples():
        selection = select_local_configs(objective_tables, weights)
        signature = selection_signature(selection)
        if signature in deduped:
            continue

        allocations, bits_dist, total_loss, total_size, total_runtime, average_bits = expand_allocations(
            tensor_specs,
            selection,
            expert_members,
            tensor_meta,
        )
        deduped[signature] = {
            "signature": signature,
            "weights": weights,
            "allocations": allocations,
            "bits_distribution": bits_dist,
            "total_loss": total_loss,
            "total_size_bytes": total_size,
            "runtime_proxy_bytes": total_runtime,
            "average_bits": average_bits,
            "num_tensors": len(allocations),
        }

    frontier = pareto_prune(list(deduped.values()))
    if not frontier:
        raise ValueError("Frontier search produced no candidate allocations")

    selected = choose_closest_to_ideal(frontier)
    return frontier, selected


def build_result(
    selected: Dict[str, Any],
    frontier: List[Dict[str, Any]],
    excluded_reason_counts: Dict[str, int],
) -> Dict[str, Any]:
    total_params = sum(selected["bits_distribution"].values())
    frontier_points = [frontier_summary(candidate, selected["signature"]) for candidate in frontier]
    excluded_tensor_count = sum(excluded_reason_counts.values())

    return {
        "objective": "pareto_balanced",
        "selection_method": "closest_to_ideal",
        "budget_bytes": selected["total_size_bytes"],
        "budget_gb": selected["total_size_bytes"] / _GB,
        "budget_utilization": 1.0,
        "total_size_bytes": selected["total_size_bytes"],
        "total_size_gb": selected["total_size_bytes"] / _GB,
        "runtime_proxy_bytes": selected["runtime_proxy_bytes"],
        "runtime_proxy_gb": selected["runtime_proxy_bytes"] / _GB,
        "total_loss": selected["total_loss"],
        "total_params": total_params,
        "average_bits": selected["average_bits"],
        "bits_distribution": {
            str(bits): {
                "params": params,
                "percentage": params / total_params * 100 if total_params else 0.0,
            }
            for bits, params in sorted(selected["bits_distribution"].items())
        },
        "solver": "frontier_grid_search",
        "solver_runtime_ms": 0.0,
        "sqnr_floor_db": SQNR_FLOOR_DB,
        "num_tensors": selected["num_tensors"],
        "frontier_size": len(frontier),
        "frontier": frontier_points,
        "excluded_tensor_count": excluded_tensor_count,
        "excluded_reason_counts": {
            "not_live_in_mlxlm": excluded_reason_counts.get("not_live_in_mlxlm", 0),
            **{
                key: value
                for key, value in sorted(excluded_reason_counts.items())
                if key != "not_live_in_mlxlm"
            },
        },
        "allocations": selected["allocations"],
    }


def main():
    parser = argparse.ArgumentParser(description="MINT MLX frontier allocator")
    parser.add_argument("--rd-curves", required=True, help="RD curves JSON from compute_rd_curves.py")
    parser.add_argument("--model-dir", required=True, help="Path to BF16 model directory")
    parser.add_argument("--output", required=True, help="Output allocation JSON")
    args = parser.parse_args()

    t0 = time.time()
    rd_data = json.load(open(args.rd_curves))
    model_dir = Path(args.model_dir)
    config = json.load(open(model_dir / "config.json"))

    live_modules = load_live_module_paths(model_dir)
    filtered_rd, excluded_reason_counts, excluded_tensor_count = filter_rd_by_live_modules(rd_data, live_modules)
    logger.info(
        f"Live-scope filter kept {len(filtered_rd['tensors'])} tensors and excluded {excluded_tensor_count}"
    )

    tensor_specs, expert_members = build_tensor_specs(filtered_rd, filtered_rd.get("total_layers", 48))
    tensor_meta = filtered_rd["tensors"]
    moe_top_k = resolve_moe_top_k(config, tensor_specs)
    objective_tables = build_objective_tables(tensor_specs, expert_members, tensor_meta, moe_top_k)
    frontier, selected = search_frontier(tensor_specs, objective_tables, expert_members, tensor_meta)

    result = build_result(selected, frontier, excluded_reason_counts)
    result["solver_runtime_ms"] = (time.time() - t0) * 1000

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved frontier allocation to {args.output}")
    logger.info(f"  Frontier points: {result['frontier_size']}")
    logger.info(f"  Selected size: {result['total_size_gb']:.2f} GB")
    logger.info(f"  Runtime proxy: {result['runtime_proxy_gb']:.2f} GB")
    logger.info(f"  Avg bits: {result['average_bits']:.2f}")
    logger.info(f"  Excluded tensors: {result['excluded_tensor_count']}")


if __name__ == "__main__":
    main()
