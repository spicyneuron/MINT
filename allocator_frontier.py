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
_PACKED_EXPERT_PATTERN = re.compile(r".+\.experts\.(gate_up_proj|down_proj)(?:\.(?:weight|bias))?$")
_GROUPED_EXPERT_PATTERN = re.compile(r".+\.experts\.\*\..+")
_MOE_TOP_K_KEYS = (
    "num_experts_per_tok",
    "num_experts_per_token",
    "num_local_experts_per_tok",
    "moe_top_k",
    "num_selected_experts",
)
_DEFAULT_SPEED_CFG = (4, 64)
_GROUP_OVERHEAD_BYTES = 2.0
_EIGHT_BIT_PREMIUM = 0.10
_NONDEFAULT_CFG_PREMIUM = 0.02


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


def estimate_runtime_proxy(
    num_params: int,
    bits: int,
    group_size: int,
    active_factor: float,
) -> float:
    """Estimate runtime in byte-equivalent units with small kernel-shape nudges."""
    size = estimate_size(num_params, bits, group_size)
    runtime = float(size)

    if group_size > 0:
        runtime += _GROUP_OVERHEAD_BYTES * int(math.ceil(num_params / group_size))

    if bits == 8:
        runtime += _EIGHT_BIT_PREMIUM * size

    if bits < 16 and (bits, group_size) != _DEFAULT_SPEED_CFG:
        runtime += _NONDEFAULT_CFG_PREMIUM * size

    return runtime * active_factor


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
            runtime = estimate_runtime_proxy(spec["num_params"], bits, group_size, active_factor)
            configs.append({
                "cfg": (bits, group_size),
                "loss": loss,
                "size": size,
                "runtime": runtime,
            })

        configs = prune_local_configs(configs)
        losses = [cfg["loss"] for cfg in configs]
        sizes = [cfg["size"] for cfg in configs]
        runtimes = [cfg["runtime"] for cfg in configs]

        loss_min, loss_max = min(losses), max(losses)
        size_min, size_max = min(sizes), max(sizes)
        runtime_min, runtime_max = min(runtimes), max(runtimes)
        loss_range = loss_max - loss_min
        size_range = size_max - size_min
        runtime_range = runtime_max - runtime_min

        for cfg in configs:
            cfg["norm_loss"] = 0.0 if loss_max == loss_min else (cfg["loss"] - loss_min) / (loss_max - loss_min)
            cfg["norm_size"] = 0.0 if size_max == size_min else (cfg["size"] - size_min) / (size_max - size_min)
            cfg["norm_runtime"] = 0.0 if runtime_max == runtime_min else (cfg["runtime"] - runtime_min) / (runtime_max - runtime_min)
            cfg["active_factor"] = active_factor

        tables.append({
            "name": spec["name"],
            "configs": configs,
            "loss_range": loss_range,
            "size_range": size_range,
            "runtime_range": runtime_range,
        })

    max_loss_range = max((table["loss_range"] for table in tables), default=0.0)
    max_size_range = max((table["size_range"] for table in tables), default=0.0)
    max_runtime_range = max((table["runtime_range"] for table in tables), default=0.0)

    for table in tables:
        loss_scale = 0.0 if max_loss_range == 0 else table["loss_range"] / max_loss_range
        size_scale = 0.0 if max_size_range == 0 else table["size_range"] / max_size_range
        runtime_scale = 0.0 if max_runtime_range == 0 else table["runtime_range"] / max_runtime_range
        table["loss_scale"] = loss_scale
        table["size_scale"] = size_scale
        table["runtime_scale"] = runtime_scale

        for cfg in table["configs"]:
            cfg["scaled_loss"] = cfg["norm_loss"] * loss_scale
            cfg["scaled_size"] = cfg["norm_size"] * size_scale
            cfg["scaled_runtime"] = cfg["norm_runtime"] * runtime_scale

    return tables


def prune_local_configs(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop per-tensor configs that are worse in both quality and runtime.

    Size is not a primary optimization axis, but we keep the minimum-size
    config as a guardrail anchor so extremely compact options remain reachable.
    """
    if len(configs) <= 1:
        return configs

    min_size_cfg = min(
        configs,
        key=lambda cfg: (
            cfg["size"],
            cfg["runtime"],
            cfg["loss"],
            cfg["cfg"][0],
            cfg["cfg"][1],
        ),
    )

    frontier = []
    for candidate in configs:
        dominated = False
        for other in configs:
            if other is candidate:
                continue
            no_worse = (
                other["loss"] <= candidate["loss"]
                and other["runtime"] <= candidate["runtime"]
            )
            strictly_better = (
                other["loss"] < candidate["loss"]
                or other["runtime"] < candidate["runtime"]
            )
            smaller_tie = (
                other["loss"] == candidate["loss"]
                and other["runtime"] == candidate["runtime"]
                and other["size"] < candidate["size"]
            )
            if no_worse and (strictly_better or smaller_tie):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)

    if min_size_cfg not in frontier:
        frontier.append(min_size_cfg)

    frontier.sort(
        key=lambda cfg: (
            cfg["runtime"],
            cfg["loss"],
            cfg["size"],
            cfg["cfg"][0],
            cfg["cfg"][1],
        )
    )
    return frontier


def selection_signature(selection: Dict[str, Dict[str, Any]]) -> Tuple[Tuple[str, int, int], ...]:
    return tuple(
        (name, cfg["cfg"][0], cfg["cfg"][1])
        for name, cfg in sorted(selection.items())
    )


def _state_beats(
    loss: float,
    runtime: float,
    size: int,
    incumbent: Optional[Dict[str, Any]],
) -> bool:
    if incumbent is None:
        return True
    return (loss, runtime, size) < (
        incumbent["loss"],
        incumbent["runtime"],
        incumbent["size"],
    )


def _runtime_bucket_idx(
    candidate_runtime: float,
    runtime_min_total: float,
    bucket_width: float,
    runtime_buckets: int,
) -> int:
    if candidate_runtime <= runtime_min_total:
        return 0
    return min(
        runtime_buckets,
        int((candidate_runtime - runtime_min_total) / bucket_width),
    )


def solve_frontier_dp(
    tensor_specs: List[Dict[str, Any]],
    objective_tables: List[Dict[str, Any]],
    expert_members: Dict[str, List[str]],
    tensor_meta: Dict[str, Dict[str, Any]],
    runtime_buckets: int = 1024,
    runtime_bucket_bytes: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Solve the global quality/speed frontier via runtime-bucket DP."""
    if runtime_buckets <= 0:
        raise ValueError("runtime_buckets must be positive")
    if runtime_bucket_bytes is not None and runtime_bucket_bytes <= 0:
        raise ValueError("runtime_bucket_bytes must be positive")

    fixed_tables = []
    variable_tables = []
    for table in objective_tables:
        if len(table["configs"]) == 1:
            fixed_tables.append(table)
        else:
            variable_tables.append(table)

    fixed_selection = {}
    fixed_loss = 0.0
    fixed_runtime = 0.0
    fixed_size = 0
    for table in fixed_tables:
        cfg = table["configs"][0]
        fixed_selection[table["name"]] = cfg
        fixed_loss += cfg["loss"]
        fixed_runtime += cfg["runtime"]
        fixed_size += cfg["size"]

    min_runtimes = [min(cfg["runtime"] for cfg in table["configs"]) for table in variable_tables]
    max_runtimes = [max(cfg["runtime"] for cfg in table["configs"]) for table in variable_tables]
    runtime_min_total = fixed_runtime + sum(min_runtimes)
    runtime_max_total = fixed_runtime + sum(max_runtimes)
    runtime_span = runtime_max_total - runtime_min_total
    bucket_width = (
        runtime_bucket_bytes
        if runtime_bucket_bytes is not None
        else max(1.0, runtime_span / runtime_buckets)
    )

    remaining_min_runtimes = [0.0] * (len(variable_tables) + 1)
    for idx in range(len(variable_tables) - 1, -1, -1):
        remaining_min_runtimes[idx] = remaining_min_runtimes[idx + 1] + min_runtimes[idx]

    states: List[Dict[str, Any]] = [{
        "loss": fixed_loss,
        "runtime": fixed_runtime,
        "size": fixed_size,
        "prev_state_id": None,
        "tensor_idx": -1,
        "cfg_idx": -1,
    }]
    live_state_ids: List[Optional[int]] = [None] * (runtime_buckets + 1)
    live_state_ids[0] = 0
    dp_peak_live_states = 1

    for tensor_idx, table in enumerate(variable_tables):
        next_live_state_ids: List[Optional[int]] = [None] * (runtime_buckets + 1)
        remaining_min_runtime = remaining_min_runtimes[tensor_idx + 1]

        for state_id in live_state_ids:
            if state_id is None:
                continue
            state = states[state_id]

            for cfg_idx, cfg in enumerate(table["configs"]):
                loss = state["loss"] + cfg["loss"]
                runtime = state["runtime"] + cfg["runtime"]
                size = state["size"] + cfg["size"]
                candidate_runtime = runtime + remaining_min_runtime
                bucket_idx = _runtime_bucket_idx(
                    candidate_runtime,
                    runtime_min_total,
                    bucket_width,
                    runtime_buckets,
                )

                incumbent_id = next_live_state_ids[bucket_idx]
                incumbent = None if incumbent_id is None else states[incumbent_id]
                if not _state_beats(loss, runtime, size, incumbent):
                    continue

                states.append({
                    "loss": loss,
                    "runtime": runtime,
                    "size": size,
                    "prev_state_id": state_id,
                    "tensor_idx": tensor_idx,
                    "cfg_idx": cfg_idx,
                })
                next_live_state_ids[bucket_idx] = len(states) - 1

        live_state_ids = next_live_state_ids
        dp_peak_live_states = max(
            dp_peak_live_states,
            sum(state_id is not None for state_id in live_state_ids),
        )

    terminal_candidates = {}
    for state_id in live_state_ids:
        if state_id is None:
            continue

        selection = dict(fixed_selection)
        cursor_id = state_id
        while cursor_id is not None:
            state = states[cursor_id]
            if state["tensor_idx"] >= 0:
                table = variable_tables[state["tensor_idx"]]
                selection[table["name"]] = table["configs"][state["cfg_idx"]]
            cursor_id = state["prev_state_id"]

        signature = selection_signature(selection)
        if signature in terminal_candidates:
            continue

        allocations, bits_dist, total_loss, total_size, total_runtime, average_bits = expand_allocations(
            tensor_specs,
            selection,
            expert_members,
            tensor_meta,
        )
        terminal_candidates[signature] = {
            "signature": signature,
            "allocations": allocations,
            "bits_distribution": bits_dist,
            "total_loss": total_loss,
            "total_size_bytes": total_size,
            "runtime_proxy_bytes": total_runtime,
            "average_bits": average_bits,
            "num_tensors": len(allocations),
        }

    frontier = pareto_prune(list(terminal_candidates.values()))
    if not frontier:
        raise ValueError("Frontier search produced no candidate allocations")

    return frontier, {
        "runtime_bucket_count": runtime_buckets + 1,
        "runtime_bucket_bytes": bucket_width,
        "dp_peak_live_states": dp_peak_live_states,
        "num_fixed_tensors": len(fixed_tables),
        "num_variable_tensors": len(variable_tables),
    }


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
    deduped = {}
    for candidate in candidates:
        key = (
            candidate["total_loss"],
            candidate["runtime_proxy_bytes"],
        )
        existing = deduped.get(key)
        if existing is None or (
            candidate["total_size_bytes"],
            candidate["signature"],
        ) < (
            existing["total_size_bytes"],
            existing["signature"],
        ):
            deduped[key] = candidate

    candidates = list(deduped.values())
    frontier = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            no_worse = (
                other["total_loss"] <= candidate["total_loss"]
                and other["runtime_proxy_bytes"] <= candidate["runtime_proxy_bytes"]
            )
            strictly_better = (
                other["total_loss"] < candidate["total_loss"]
                or other["runtime_proxy_bytes"] < candidate["runtime_proxy_bytes"]
            )
            smaller_tie = (
                other["total_loss"] == candidate["total_loss"]
                and other["runtime_proxy_bytes"] == candidate["runtime_proxy_bytes"]
                and other["total_size_bytes"] < candidate["total_size_bytes"]
            )
            if no_worse and (strictly_better or smaller_tie):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)

    frontier.sort(
        key=lambda candidate: (
            candidate["runtime_proxy_bytes"],
            candidate["total_loss"],
            candidate["total_size_bytes"],
        )
    )
    return frontier


def choose_closest_to_ideal(frontier: List[Dict[str, Any]]) -> Dict[str, Any]:
    mins = {
        "total_loss": min(candidate["total_loss"] for candidate in frontier),
        "runtime_proxy_bytes": min(candidate["runtime_proxy_bytes"] for candidate in frontier),
    }
    maxs = {
        "total_loss": max(candidate["total_loss"] for candidate in frontier),
        "runtime_proxy_bytes": max(candidate["runtime_proxy_bytes"] for candidate in frontier),
    }

    def distance(candidate: Dict[str, Any]) -> float:
        total = 0.0
        for key in ("total_loss", "runtime_proxy_bytes"):
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


def build_loss_runtime_curve(frontier: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Project the candidate set onto the loss/runtime tradeoff curve."""
    ordered = sorted(
        frontier,
        key=lambda candidate: (
            candidate["runtime_proxy_bytes"],
            candidate["total_loss"],
            candidate["total_size_bytes"],
        ),
    )

    curve = []
    best_loss = float("inf")
    for candidate in ordered:
        if candidate["total_loss"] < best_loss:
            curve.append(candidate)
            best_loss = candidate["total_loss"]

    return curve


def choose_knee_point(curve: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Select the knee of the loss/runtime curve via max distance to the chord."""
    if len(curve) < 3:
        return None

    runtimes = [candidate["runtime_proxy_bytes"] for candidate in curve]
    losses = [candidate["total_loss"] for candidate in curve]
    runtime_min, runtime_max = min(runtimes), max(runtimes)
    loss_min, loss_max = min(losses), max(losses)

    if runtime_max == runtime_min or loss_max == loss_min:
        return None

    start = (0.0, 0.0)
    end = (1.0, 1.0)
    line_len = math.hypot(end[0] - start[0], end[1] - start[1])
    if line_len == 0:
        return None

    best = None
    for candidate in curve[1:-1]:
        x = (candidate["runtime_proxy_bytes"] - runtime_min) / (runtime_max - runtime_min)
        y = (loss_max - candidate["total_loss"]) / (loss_max - loss_min)
        distance = abs((end[1] - start[1]) * x - (end[0] - start[0]) * y) / line_len
        point = {
            "candidate": candidate,
            "distance": distance,
        }
        if best is None or (
            point["distance"],
            -candidate["total_loss"],
            -candidate["runtime_proxy_bytes"],
            -candidate["total_size_bytes"],
        ) > (
            best["distance"],
            -best["candidate"]["total_loss"],
            -best["candidate"]["runtime_proxy_bytes"],
            -best["candidate"]["total_size_bytes"],
        ):
            best = point

    if best is None or best["distance"] <= 0:
        return None

    return best


def select_frontier_candidate(
    frontier: List[Dict[str, Any]],
    size_guardrail_bytes: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Pick the recommended point from the quality/runtime frontier."""
    scoped_frontier = frontier
    if size_guardrail_bytes is not None:
        scoped_frontier = [
            candidate for candidate in frontier
            if candidate["total_size_bytes"] <= size_guardrail_bytes
        ]
        if not scoped_frontier:
            selected = min(
                frontier,
                key=lambda candidate: (
                    candidate["total_size_bytes"],
                    candidate["runtime_proxy_bytes"],
                    candidate["total_loss"],
                ),
            )
            return selected, {
                "selection_method": "size_guardrail_smallest_fallback",
                "selection_curve_size": 1,
                "size_guardrail_bytes": size_guardrail_bytes,
            }

    curve = build_loss_runtime_curve(scoped_frontier)
    knee_point = choose_knee_point(curve)

    if knee_point is None:
        selected = choose_closest_to_ideal(scoped_frontier)
        return selected, {
            "selection_method": "closest_to_ideal_fallback",
            "selection_curve_size": len(curve),
            **(
                {"size_guardrail_bytes": size_guardrail_bytes}
                if size_guardrail_bytes is not None else {}
            ),
        }

    loss_cap = knee_point["candidate"]["total_loss"]
    eligible = [
        candidate for candidate in scoped_frontier
        if candidate["total_loss"] <= loss_cap + 1e-12
    ]
    selected = min(
        eligible,
        key=lambda candidate: (
            candidate["runtime_proxy_bytes"],
            candidate["total_size_bytes"],
            candidate["total_loss"],
        ),
    )
    return selected, {
        "selection_method": "knee_loss_cap_fastest_under_cap",
        "selection_curve_size": len(curve),
        "loss_cap": loss_cap,
        "knee_point_signature": knee_point["candidate"]["signature"],
        "knee_point_distance": knee_point["distance"],
        **(
            {"size_guardrail_bytes": size_guardrail_bytes}
            if size_guardrail_bytes is not None else {}
        ),
    }


def search_frontier(
    tensor_specs: List[Dict[str, Any]],
    objective_tables: List[Dict[str, Any]],
    expert_members: Dict[str, List[str]],
    tensor_meta: Dict[str, Dict[str, Any]],
    size_guardrail_bytes: Optional[int] = None,
    runtime_buckets: int = 1024,
    runtime_bucket_bytes: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    frontier, solver_meta = solve_frontier_dp(
        tensor_specs,
        objective_tables,
        expert_members,
        tensor_meta,
        runtime_buckets=runtime_buckets,
        runtime_bucket_bytes=runtime_bucket_bytes,
    )
    selected, selection_meta = select_frontier_candidate(frontier, size_guardrail_bytes=size_guardrail_bytes)
    selection_meta.update(solver_meta)
    return frontier, selected, selection_meta


def build_result(
    selected: Dict[str, Any],
    frontier: List[Dict[str, Any]],
    excluded_reason_counts: Dict[str, int],
    selection_meta: Dict[str, Any],
) -> Dict[str, Any]:
    total_params = sum(selected["bits_distribution"].values())
    frontier_points = [frontier_summary(candidate, selected["signature"]) for candidate in frontier]
    excluded_tensor_count = sum(excluded_reason_counts.values())

    return {
        "objective": "quality_speed_frontier",
        "selection_method": selection_meta["selection_method"],
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
        "selection_axes": ["total_loss", "runtime_proxy_bytes"],
        "size_role": "guardrail_tiebreaker",
        "runtime_proxy_profile": {
            "base": "size_bytes",
            "group_overhead_bytes_per_group": _GROUP_OVERHEAD_BYTES,
            "eight_bit_premium": _EIGHT_BIT_PREMIUM,
            "nondefault_cfg_premium": _NONDEFAULT_CFG_PREMIUM,
            "default_speed_cfg": list(_DEFAULT_SPEED_CFG),
        },
        "bits_distribution": {
            str(bits): {
                "params": params,
                "percentage": params / total_params * 100 if total_params else 0.0,
            }
            for bits, params in sorted(selected["bits_distribution"].items())
        },
        "solver": "frontier_dp_2d",
        "solver_runtime_ms": 0.0,
        "runtime_bucket_count": selection_meta["runtime_bucket_count"],
        "runtime_bucket_bytes": selection_meta["runtime_bucket_bytes"],
        "dp_peak_live_states": selection_meta["dp_peak_live_states"],
        "num_fixed_tensors": selection_meta["num_fixed_tensors"],
        "num_variable_tensors": selection_meta["num_variable_tensors"],
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
        "selection_curve_size": selection_meta["selection_curve_size"],
        **(
            {"selection_loss_cap": selection_meta["loss_cap"]}
            if "loss_cap" in selection_meta else {}
        ),
        **(
            {"selection_knee_point_distance": selection_meta["knee_point_distance"]}
            if "knee_point_distance" in selection_meta else {}
        ),
        **(
            {
                "size_guardrail_bytes": selection_meta["size_guardrail_bytes"],
                "size_guardrail_gb": selection_meta["size_guardrail_bytes"] / _GB,
            }
            if "size_guardrail_bytes" in selection_meta else {}
        ),
        "allocations": selected["allocations"],
    }


def main():
    parser = argparse.ArgumentParser(description="MINT MLX frontier allocator")
    parser.add_argument("--rd-curves", required=True, help="RD curves JSON from compute_rd_curves.py")
    parser.add_argument("--model-dir", required=True, help="Path to BF16 model directory")
    parser.add_argument("--output", required=True, help="Output allocation JSON")
    parser.add_argument(
        "--max-size-gb",
        type=float,
        default=None,
        help="Optional size guardrail for final point selection",
    )
    parser.add_argument(
        "--runtime-buckets",
        type=int,
        default=1024,
        help="Number of runtime buckets for the DP frontier solver",
    )
    args = parser.parse_args()
    if args.runtime_buckets <= 0:
        parser.error("--runtime-buckets must be positive")

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
    size_guardrail_bytes = None if args.max_size_gb is None else int(args.max_size_gb * _GB)
    frontier, selected, selection_meta = search_frontier(
        tensor_specs,
        objective_tables,
        expert_members,
        tensor_meta,
        size_guardrail_bytes=size_guardrail_bytes,
        runtime_buckets=args.runtime_buckets,
    )

    result = build_result(selected, frontier, excluded_reason_counts, selection_meta)
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
