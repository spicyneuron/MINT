#!/usr/bin/env python3
"""Turn RD curves into tiered hand-tuning recommendations.

This script does not try to solve a budgeted optimization problem. Instead,
it turns RD curves into empirical sensitivity signals that can inform a
hand-tuned mixed-precision recipe with four effective tiers:

- base: 4-bit
- tier 6: somewhat sensitive
- tier 8: very sensitive
- tier 16: still sensitive even at 8-bit

The input RD curves only measure 2/3/4/8/16-bit points, so the 6-bit tier is
heuristic. It is used as an interpolation slot between the base 4-bit tier
and the measured 8-bit tier.

The script emits:
- family summaries across all quantizable 2D tensors
- exact outlier tensors for layer-specific exceptions
- paste-ready `--q-override` stubs using HF-style names with assigned tiers

Usage:
    python recommend_overrides.py --rd-curves rd_curves.json
"""

import argparse
import json
import logging
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mint.recommend")

INDEX_LABELS = {
    "layers": "layer",
    "experts": "expert",
    "blocks": "block",
}

BASELINE_4 = "4_64"
TIER_VALUES = {
    "tier16": "16",
    "tier8": "8",
    "tier6": "6",
}


def parse_config(config_key: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)_(\d+)", config_key)
    if not m:
        raise ValueError(f"Invalid config key '{config_key}', expected bits_group")
    return int(m.group(1)), int(m.group(2))


def strip_tensor_suffix(name: str) -> str:
    if name.endswith(".weight") or name.endswith(".bias"):
        return name.rsplit(".", 1)[0]
    return name


def family_key(name: str) -> str:
    parts = strip_tensor_suffix(name).split(".")
    out = []
    for idx, part in enumerate(parts):
        if part.isdigit():
            parent = parts[idx - 1] if idx > 0 else "idx"
            label = INDEX_LABELS.get(parent, "idx")
            out.append("{" + label + "}")
        else:
            out.append(part)
    return ".".join(out)


def family_regex(name: str) -> str:
    parts = family_key(name).split(".")
    out = []
    for part in parts:
        if part.startswith("{") and part.endswith("}"):
            out.append(r"(\d+)")
        else:
            out.append(re.escape(part))
    return r"\.".join(out)


def exact_regex(name: str) -> str:
    return re.escape(strip_tensor_suffix(name))


def shorten(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    left = (width - 3) // 2
    right = width - 3 - left
    return text[:left] + "..." + text[-right:]


def subsystem(name: str) -> str:
    if name == "lm_head.weight" or name.startswith("model.language_model."):
        return "language"
    if name.startswith("mtp."):
        return "mtp"
    if name.startswith("model.visual."):
        return "vision"
    return "other"


def config_size_bytes(num_params: int, bits: int, group_size: int, source_bits: int) -> float:
    if bits >= source_bits:
        return num_params * (source_bits / 8.0)

    payload = num_params * (bits / 8.0)
    if group_size <= 0:
        return payload

    metadata = math.ceil(num_params / group_size) * 2
    return payload + metadata


def best_cfg_for_bit(curve: dict, bits: int):
    matches = []
    for key, loss in curve.items():
        cfg_bits, cfg_gs = parse_config(key)
        if cfg_bits == bits:
            matches.append((float(loss), key, cfg_gs))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[2]))
    loss, key, gs = matches[0]
    return {
        "config": key,
        "loss": loss,
        "group_size": gs,
    }


def dominant_value(values: list[str]) -> str:
    counts = Counter(values)
    return counts.most_common(1)[0][0]


def tier_band_count(total: int, frac: float) -> int:
    if total <= 0:
        return 0
    return max(1, math.ceil(total * frac))


def build_tensor_rows(rd_data: dict) -> list[dict]:
    source_bits = int(rd_data.get("source_bits", 16))
    rows = []
    skipped = 0

    for name, info in rd_data["tensors"].items():
        if info.get("is_1d"):
            skipped += 1
            continue

        curve = info["rd_curve"]
        best2 = best_cfg_for_bit(curve, 2)
        best3 = best_cfg_for_bit(curve, 3)
        best4 = best_cfg_for_bit(curve, 4)
        best8 = best_cfg_for_bit(curve, 8)
        best16 = best_cfg_for_bit(curve, 16)

        if best4 is None or best8 is None or best16 is None:
            skipped += 1
            continue

        default4 = {
            "config": BASELINE_4,
            "loss": float(curve[BASELINE_4]),
            "group_size": 64,
        } if BASELINE_4 in curve else best4

        params = int(info["num_params"])

        extra8_bytes = (
            config_size_bytes(params, 8, best8["group_size"], source_bits)
            - config_size_bytes(params, 4, default4["group_size"], source_bits)
        )
        extra16_bytes = (
            config_size_bytes(params, 16, 0, source_bits)
            - config_size_bytes(params, 4, default4["group_size"], source_bits)
        )

        rows.append(
            {
                "tensor_name": name,
                "module_name": strip_tensor_suffix(name),
                "family": family_key(name),
                "family_regex": family_regex(name),
                "exact_regex": exact_regex(name),
                "subsystem": subsystem(name),
                "layer_idx": info.get("layer_idx"),
                "num_params": params,
                "params_b": params / 1e9,
                "default4_cfg": default4["config"],
                "default4_loss": default4["loss"],
                "best2_cfg": None if best2 is None else best2["config"],
                "best2_loss": None if best2 is None else best2["loss"],
                "best3_cfg": None if best3 is None else best3["config"],
                "best3_loss": None if best3 is None else best3["loss"],
                "best4_cfg": best4["config"],
                "best4_loss": best4["loss"],
                "best8_cfg": best8["config"],
                "best8_loss": best8["loss"],
                "best16_cfg": best16["config"],
                "best16_loss": best16["loss"],
                "gain_4gs": default4["loss"] - best4["loss"],
                "gain_4_to_8": default4["loss"] - best8["loss"],
                "gain_best4_to_8": best4["loss"] - best8["loss"],
                "gain_8_to_16": best8["loss"] - best16["loss"],
                "gain_4_to_16": default4["loss"] - best16["loss"],
                "extra8_bytes": extra8_bytes,
                "extra8_mib": extra8_bytes / (1024 ** 2),
                "extra16_bytes": extra16_bytes,
                "extra16_mib": extra16_bytes / (1024 ** 2),
            }
        )

    logger.info("Built %d tensor rows (%d skipped)", len(rows), skipped)
    return rows


def summarize_families(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)

    summaries = []
    for family, items in grouped.items():
        top4 = max(items, key=lambda item: item["gain_4_to_8"])
        top16 = max(items, key=lambda item: item["gain_8_to_16"])
        summaries.append(
            {
                "family": family,
                "family_regex": items[0]["family_regex"],
                "subsystem": dominant_value([item["subsystem"] for item in items]),
                "count": len(items),
                "params_b": sum(item["params_b"] for item in items),
                "median_default4": statistics.median(item["default4_loss"] for item in items),
                "median_gain_4gs": statistics.median(item["gain_4gs"] for item in items),
                "median_gain_4_to_8": statistics.median(item["gain_4_to_8"] for item in items),
                "median_gain_best4_to_8": statistics.median(item["gain_best4_to_8"] for item in items),
                "median_gain_8_to_16": statistics.median(item["gain_8_to_16"] for item in items),
                "median_gain_4_to_16": statistics.median(item["gain_4_to_16"] for item in items),
                "dominant_best4": dominant_value([item["best4_cfg"] for item in items]),
                "dominant_best8": dominant_value([item["best8_cfg"] for item in items]),
                "total_extra8_mib": sum(item["extra8_mib"] for item in items),
                "total_extra16_mib": sum(item["extra16_mib"] for item in items),
                "top_tensor_4_to_8": top4["tensor_name"],
                "top_tensor_8_to_16": top16["tensor_name"],
                "example_module": top4["module_name"],
            }
        )
    return summaries


def select_blanket_tiers(summaries: list[dict], min_family_count: int) -> dict[str, list[dict]]:
    candidates = [
        item
        for item in summaries
        if item["count"] >= min_family_count and item["median_gain_4_to_8"] > 0
    ]

    # Tier 16 should be selective. Use residual 8->16 sensitivity.
    tier16_count = tier_band_count(len(candidates), 0.15)
    tier16 = sorted(
        candidates,
        key=lambda item: (
            item["median_gain_8_to_16"],
            item["median_gain_4_to_8"],
            -item["total_extra16_mib"],
        ),
        reverse=True,
    )[:tier16_count]
    tier16_families = {item["family"] for item in tier16}

    remaining = [item for item in candidates if item["family"] not in tier16_families]

    tier8_count = tier_band_count(len(candidates), 0.25)
    tier8 = sorted(
        remaining,
        key=lambda item: (
            item["median_gain_4_to_8"],
            item["median_gain_best4_to_8"],
            -item["total_extra8_mib"],
        ),
        reverse=True,
    )[:tier8_count]
    tier8_families = {item["family"] for item in tier8}

    remaining = [item for item in remaining if item["family"] not in tier8_families]

    tier6_count = tier_band_count(len(candidates), 0.25)
    tier6 = sorted(
        remaining,
        key=lambda item: (
            item["median_gain_4_to_8"],
            item["median_gain_4gs"],
            -item["total_extra8_mib"],
        ),
        reverse=True,
    )[:tier6_count]

    return {
        "tier16": tier16,
        "tier8": tier8,
        "tier6": tier6,
    }


def pick_exact_outliers(
    rows: list[dict],
    covered_families: set[str],
    key_name: str,
    limit: int,
    max_per_family: int = 4,
) -> list[dict]:
    ranked = [
        row
        for row in rows
        if row["family"] not in covered_families and row[key_name] > 0
    ]
    ranked.sort(
        key=lambda row: (
            row[key_name],
            row["gain_4_to_8"],
            -row["extra8_mib"],
        ),
        reverse=True,
    )

    picked = []
    per_family = Counter()
    seen = set()
    for row in ranked:
        if row["tensor_name"] in seen:
            continue
        if per_family[row["family"]] >= max_per_family:
            continue
        picked.append(row)
        seen.add(row["tensor_name"])
        per_family[row["family"]] += 1
        if len(picked) >= limit:
            break
    return picked


def print_family_table(title: str, summaries: list[dict], limit: int):
    print(f"\n--- {title} ---")
    if not summaries:
        print("  (none)")
        return

    print(
        f"  {'Family':<52s} {'N':>3s} {'4->8':>8s} {'8->16':>8s} "
        f"{'4gs':>8s} {'best4':>7s} {'best8':>7s}"
    )
    for item in summaries[:limit]:
        label = shorten(item["family"], 52)
        print(
            f"  {label:<52s} {item['count']:>3d} "
            f"{item['median_gain_4_to_8']:>8.5f} {item['median_gain_8_to_16']:>8.5f} "
            f"{item['median_gain_4gs']:>8.5f} {item['dominant_best4']:>7s} "
            f"{item['dominant_best8']:>7s}"
        )


def print_tensor_table(title: str, rows: list[dict], limit: int):
    print(f"\n--- {title} ---")
    if not rows:
        print("  (none)")
        return

    print(
        f"  {'Tensor':<64s} {'4->8':>8s} {'8->16':>8s} {'4gs':>8s}"
    )
    for row in rows[:limit]:
        label = shorten(row["module_name"], 64)
        print(
            f"  {label:<64s} {row['gain_4_to_8']:>8.5f} "
            f"{row['gain_8_to_16']:>8.5f} {row['gain_4gs']:>8.5f}"
        )


def print_stub_section(title: str, families: list[dict], rows: list[dict], tier_name: str):
    value = TIER_VALUES[tier_name]
    print(f"\n--- {title} ---")
    if not families and not rows:
        print("  (none)")
        return

    for item in families:
        print(
            f'  --q-override "{item["family_regex"]}={value}"'
            f'    # p50 4->8={item["median_gain_4_to_8"]:.5f}'
            f' 8->16={item["median_gain_8_to_16"]:.5f}'
            f' 4gs={item["median_gain_4gs"]:.5f}'
            f' best4={item["dominant_best4"]} best8={item["dominant_best8"]}'
            f' extra@8={item["total_extra8_mib"]:.2f}MiB'
            f' extra@16={item["total_extra16_mib"]:.2f}MiB'
        )

    for row in rows:
        print(
            f'  --q-override "{row["exact_regex"]}={value}"'
            f'    # 4->8={row["gain_4_to_8"]:.5f}'
            f' 8->16={row["gain_8_to_16"]:.5f}'
            f' 4gs={row["gain_4gs"]:.5f}'
            f' best4={row["best4_cfg"]} best8={row["best8_cfg"]}'
            f' extra@8={row["extra8_mib"]:.3f}MiB'
            f' extra@16={row["extra16_mib"]:.3f}MiB'
            f' name={row["module_name"]}'
        )


def build_json_report(rd_path: Path, rd_data: dict, summaries: list[dict], blanket: dict, exacts: dict):
    return {
        "rd_curves": str(rd_path),
        "model": rd_data.get("model", rd_path.stem),
        "baseline": BASELINE_4,
        "tier_values": TIER_VALUES,
        "note": "6-bit tier is heuristic; RD curves only measure 2/3/4/8/16",
        "families": summaries,
        "blanket_tiers": blanket,
        "exact_tiers": exacts,
    }


def main():
    parser = argparse.ArgumentParser(description="Recommend tiered overrides from RD curves")
    parser.add_argument("--rd-curves", required=True, help="Path to rd_curves.json")
    parser.add_argument(
        "--min-family-count",
        type=int,
        default=4,
        help="Minimum tensors before a family becomes a blanket candidate",
    )
    parser.add_argument(
        "--top-family-table",
        type=int,
        default=20,
        help="Number of family rows to print in summary tables",
    )
    parser.add_argument(
        "--top-exact-per-tier",
        type=int,
        default=10,
        help="Number of exact tensor outliers to print per tier",
    )
    parser.add_argument("--json-output", help="Optional path to write machine-readable JSON")
    args = parser.parse_args()

    rd_path = Path(args.rd_curves)
    with open(rd_path) as f:
        rd_data = json.load(f)

    rows = build_tensor_rows(rd_data)
    if not rows:
        raise SystemExit("No usable 2D tensors found in RD curves")

    summaries = summarize_families(rows)

    by_4_to_8 = sorted(
        summaries,
        key=lambda item: (
            item["median_gain_4_to_8"],
            item["median_gain_best4_to_8"],
            -item["total_extra8_mib"],
        ),
        reverse=True,
    )
    by_8_to_16 = sorted(
        summaries,
        key=lambda item: (
            item["median_gain_8_to_16"],
            item["median_gain_4_to_8"],
            -item["total_extra16_mib"],
        ),
        reverse=True,
    )
    by_4gs = sorted(
        summaries,
        key=lambda item: (
            item["median_gain_4gs"],
            item["median_gain_4_to_8"],
            -item["total_extra8_mib"],
        ),
        reverse=True,
    )

    blanket = select_blanket_tiers(summaries, args.min_family_count)
    covered_families = {
        item["family"]
        for tier_items in blanket.values()
        for item in tier_items
    }

    exact_pool = [row for row in rows if row["family"] not in covered_families]
    exacts = {}

    exacts["tier16"] = pick_exact_outliers(exact_pool, set(), "gain_8_to_16", args.top_exact_per_tier)
    chosen = {row["tensor_name"] for row in exacts["tier16"]}

    tier8_pool = [row for row in exact_pool if row["tensor_name"] not in chosen]
    exacts["tier8"] = pick_exact_outliers(tier8_pool, set(), "gain_4_to_8", args.top_exact_per_tier)
    chosen.update(row["tensor_name"] for row in exacts["tier8"])

    tier6_pool = [row for row in exact_pool if row["tensor_name"] not in chosen]
    exacts["tier6"] = pick_exact_outliers(tier6_pool, set(), "gain_4_to_8", args.top_exact_per_tier)

    print(f"\n{'=' * 82}")
    print(f"RD Sensitivity Recommendations: {rd_data.get('model', rd_path.stem)}")
    print(f"{'=' * 82}")
    print(f"  RD curves:         {rd_path}")
    print(f"  Baseline tier:     4-bit ({BASELINE_4})")
    print(f"  Upgrade tiers:     6 / 8 / 16")
    print(f"  Blanket min count: {args.min_family_count}")
    print("  Note: 6-bit is heuristic because RD curves only measure 2/3/4/8/16")

    print_family_table("Top Families By Default-4 To Best-8 Gain", by_4_to_8, args.top_family_table)
    print_family_table("Top Families By Residual Best-8 To 16 Gain", by_8_to_16, args.top_family_table)
    print_family_table("Top Families By 4-bit Group-Size Relief", by_4gs, args.top_family_table)

    print_tensor_table("Tier 16 Exact Outliers", exacts["tier16"], args.top_exact_per_tier)
    print_tensor_table("Tier 8 Exact Outliers", exacts["tier8"], args.top_exact_per_tier)
    print_tensor_table("Tier 6 Exact Outliers", exacts["tier6"], args.top_exact_per_tier)

    print_stub_section("Tier 16 Blanket And Exact Stubs", blanket["tier16"], exacts["tier16"], "tier16")
    print_stub_section("Tier 8 Blanket And Exact Stubs", blanket["tier8"], exacts["tier8"], "tier8")
    print_stub_section("Tier 6 Blanket And Exact Stubs", blanket["tier6"], exacts["tier6"], "tier6")
    print(f"\n{'=' * 82}\n")

    if args.json_output:
        report = build_json_report(rd_path, rd_data, summaries, blanket, exacts)
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
