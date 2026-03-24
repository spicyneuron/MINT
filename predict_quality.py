#!/usr/bin/env python3
"""Predict quantized model quality before conversion.

Sweeps budget levels using the MCKP allocator and estimates perplexity
from the allocation loss curve. Requires only the RD curves from Step 1 —
no model conversion or evaluation needed.

Two modes:
  1. Without calibration: shows relative quality (allocation loss) vs budget.
     Useful for finding diminishing returns and choosing a budget.
  2. With calibration (--calibrate): fits a prediction curve using 1-2 known
     PPL measurements, then predicts PPL at any budget.

Usage:
    # Quick sweep — find the right budget (no PPL eval needed)
    python mint/predict_quality.py \
        --rd-curves analysis/model-rd-curves.json \
        --min-gb 15 --max-gb 50

    # Calibrated prediction — predict PPL at any budget
    python mint/predict_quality.py \
        --rd-curves analysis/model-rd-curves.json \
        --min-gb 15 --max-gb 50 \
        --calibrate 20.0:6.693 --calibrate 30.0:6.587
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from allocator import build_tensor_specs, allocate_greedy, compute_min_safe_size

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mint.predict")


def sweep_budgets(rd_data, total_layers, min_gb, max_gb, num_points=20,
                  speed_mode="balanced", moe_aggregation="weighted_mean"):
    """Run the allocator at many budget points and collect loss vs size."""
    specs, expert_members = build_tensor_specs(
        rd_data, total_layers, moe_aggregation, speed_mode=speed_mode
    )
    min_safe_bytes = compute_min_safe_size(specs)
    min_safe_gb = min_safe_bytes / (1024**3)

    # Generate budget points
    actual_min = max(min_gb, min_safe_gb)
    budgets_gb = np.linspace(actual_min, max_gb, num_points)

    results = []
    for budget_gb in budgets_gb:
        budget_bytes = int(budget_gb * 1024**3)
        if budget_bytes < min_safe_bytes:
            budget_bytes = min_safe_bytes

        result = allocate_greedy(specs, budget_bytes, expert_members)
        results.append({
            'budget_gb': budget_gb,
            'actual_gb': result['total_size_gb'],
            'avg_bits': result['average_bits'],
            'total_loss': result['total_loss'],
            'bits_distribution': result['bits_distribution'],
        })

    return results, min_safe_gb


def fit_prediction_curve(sweep_results, calibration_points):
    """Fit PPL = a + b * loss^c using calibration points.

    calibration_points: list of (budget_gb, measured_ppl)
    """
    # Find the allocation loss at each calibration budget
    cal_losses = []
    cal_ppls = []
    for cal_budget, cal_ppl in calibration_points:
        # Find closest sweep point
        closest = min(sweep_results, key=lambda r: abs(r['budget_gb'] - cal_budget))
        cal_losses.append(closest['total_loss'])
        cal_ppls.append(cal_ppl)

    cal_losses = np.array(cal_losses)
    cal_ppls = np.array(cal_ppls)

    if len(calibration_points) == 1:
        # With one point, use log model: PPL = a + b * ln(1 + loss)
        # Assume BF16 PPL ≈ measured_ppl * 0.98 (conservative)
        loss, ppl = cal_losses[0], cal_ppls[0]
        # Solve: ppl = a + b * ln(1 + loss), assume a ≈ ppl - 0.15 for typical models
        b = 0.055  # empirical default from Qwen3.5 fit
        a = ppl - b * np.log1p(loss)
        return lambda l: a + b * np.log1p(l), {'a': a, 'b': b, 'model': 'log(1-point)'}

    elif len(calibration_points) >= 2:
        # With 2+ points, fit log model properly
        from scipy.optimize import curve_fit

        def log_model(loss, a, b):
            return a + b * np.log1p(loss)

        try:
            popt, _ = curve_fit(log_model, cal_losses, cal_ppls, p0=[cal_ppls[-1], 0.05])
            return lambda l: log_model(l, *popt), {'a': popt[0], 'b': popt[1], 'model': 'log(fitted)'}
        except Exception:
            # Fallback to linear interpolation
            slope = (cal_ppls[0] - cal_ppls[-1]) / (cal_losses[0] - cal_losses[-1] + 1e-12)
            intercept = cal_ppls[-1] - slope * cal_losses[-1]
            return lambda l: intercept + slope * l, {'slope': slope, 'intercept': intercept, 'model': 'linear'}

    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Predict quantized model quality before conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick quality-vs-size sweep (no conversion needed):
    python mint/predict_quality.py --rd-curves rd_curves.json --min-gb 15 --max-gb 50

    # With one calibration point (e.g., you measured PPL at 20 GB):
    python mint/predict_quality.py --rd-curves rd_curves.json \\
        --min-gb 15 --max-gb 50 --calibrate 20.0:6.693

    # With two calibration points (more accurate):
    python mint/predict_quality.py --rd-curves rd_curves.json \\
        --min-gb 15 --max-gb 50 \\
        --calibrate 20.0:6.693 --calibrate 30.0:6.587
        """,
    )
    parser.add_argument("--rd-curves", required=True, help="RD curves JSON from compute_rd_curves.py")
    parser.add_argument("--min-gb", type=float, default=None, help="Minimum budget (default: min-safe)")
    parser.add_argument("--max-gb", type=float, required=True, help="Maximum budget to sweep")
    parser.add_argument("--num-points", type=int, default=20, help="Number of budget points (default: 20)")
    parser.add_argument("--speed-mode", choices=["full", "balanced", "fast"], default="balanced")
    parser.add_argument("--calibrate", action="append", metavar="BUDGET:PPL",
                        help="Calibration point as BUDGET_GB:MEDIAN_PPL (can specify multiple)")
    parser.add_argument("--output", help="Save results to JSON")
    args = parser.parse_args()

    rd_data = json.load(open(args.rd_curves))
    total_layers = rd_data.get("total_layers", 48)
    model_name = rd_data.get("model", "unknown")

    # Parse calibration points
    calibration_points = []
    if args.calibrate:
        for cal in args.calibrate:
            parts = cal.split(":")
            if len(parts) != 2:
                parser.error(f"Invalid calibration format: {cal} (expected BUDGET:PPL)")
            calibration_points.append((float(parts[0]), float(parts[1])))

    # Sweep
    min_gb = args.min_gb if args.min_gb else 0.0
    sweep_results, min_safe_gb = sweep_budgets(
        rd_data, total_layers, min_gb if min_gb > 0 else min_safe_gb,
        args.max_gb, args.num_points, args.speed_mode
    )

    # Fit prediction if calibration provided
    predict_fn = None
    fit_params = None
    if calibration_points:
        predict_fn, fit_params = fit_prediction_curve(sweep_results, calibration_points)
        if fit_params:
            print(f"Prediction model: {fit_params['model']}")
            for k, v in fit_params.items():
                if k != 'model':
                    print(f"  {k} = {v:.6f}")

    # Display
    print(f"\n{'='*80}")
    print(f"  MINT Quality Prediction: {model_name}")
    print(f"  Speed mode: {args.speed_mode}, Min safe: {min_safe_gb:.1f} GB")
    if calibration_points:
        print(f"  Calibration: {', '.join(f'{b:.0f}GB→{p:.3f}' for b, p in calibration_points)}")
    print(f"{'='*80}")

    header = f"  {'Budget':>8s} {'Size':>8s} {'Avg Bits':>9s} {'Loss':>10s}"
    if predict_fn:
        header += f" {'Pred PPL':>10s} {'vs BF16':>8s}"
    header += f"  {'Bit Distribution'}"
    print(header)
    print(f"  {'-'*8} {'-'*8} {'-'*9} {'-'*10}", end="")
    if predict_fn:
        print(f" {'-'*10} {'-'*8}", end="")
    print(f"  {'-'*30}")

    bf16_ppl = predict_fn(0.0) if predict_fn else None

    for r in sweep_results:
        # Compact bit distribution
        bits_str = ", ".join(f"{b}b:{info['percentage']:.0f}%"
                           for b, info in sorted(r['bits_distribution'].items(), key=lambda x: int(x[0]))
                           if info['percentage'] >= 1.0)

        line = f"  {r['budget_gb']:>8.1f} {r['actual_gb']:>8.1f} {r['avg_bits']:>9.2f} {r['total_loss']:>10.4f}"
        if predict_fn:
            pred_ppl = predict_fn(r['total_loss'])
            delta = (pred_ppl - bf16_ppl) / bf16_ppl * 100 if bf16_ppl else 0
            line += f" {pred_ppl:>10.4f} {delta:>+7.2f}%"
        line += f"  {bits_str}"
        print(line)

    # Add BF16 reference
    if predict_fn:
        print(f"  {'BF16':>8s} {'---':>8s} {'16.00':>9s} {'0.0000':>10s} {bf16_ppl:>10.4f} {'+0.00%':>8s}  16b:100%")

    print(f"{'='*80}")

    # Recommendations
    print(f"\nRecommendations:")
    if len(sweep_results) >= 3:
        # Find diminishing returns point (where loss reduction per GB drops below 10%)
        for i in range(1, len(sweep_results)):
            prev = sweep_results[i-1]
            curr = sweep_results[i]
            loss_reduction = prev['total_loss'] - curr['total_loss']
            gb_increase = curr['actual_gb'] - prev['actual_gb']
            if gb_increase > 0 and prev['total_loss'] > 0:
                efficiency = loss_reduction / prev['total_loss']  # fractional improvement
                if efficiency < 0.02 and curr['budget_gb'] > sweep_results[0]['budget_gb']:
                    print(f"  Diminishing returns above ~{curr['budget_gb']:.0f} GB "
                          f"(loss drops <2% per step)")
                    break

        # Find sweet spot (best efficiency)
        best_eff = 0
        best_point = sweep_results[0]
        for i in range(1, len(sweep_results)):
            prev = sweep_results[i-1]
            curr = sweep_results[i]
            loss_reduction = prev['total_loss'] - curr['total_loss']
            gb_increase = curr['actual_gb'] - prev['actual_gb']
            if gb_increase > 0:
                eff = loss_reduction / gb_increase
                if eff > best_eff:
                    best_eff = eff
                    best_point = curr
        print(f"  Best value: ~{best_point['budget_gb']:.0f} GB "
              f"({best_point['avg_bits']:.1f} avg bits, loss {best_point['total_loss']:.3f})")

    if predict_fn:
        # Find budget that matches within 1% of BF16
        for r in sweep_results:
            pred = predict_fn(r['total_loss'])
            if (pred - bf16_ppl) / bf16_ppl < 0.01:
                print(f"  Matches BF16 (within 1%): ~{r['budget_gb']:.0f} GB")
                break

    # Save
    if args.output:
        output = {
            'model': model_name,
            'speed_mode': args.speed_mode,
            'min_safe_gb': min_safe_gb,
            'calibration': calibration_points,
            'fit_params': fit_params,
            'sweep': sweep_results,
        }
        if predict_fn:
            output['predictions'] = [
                {'budget_gb': r['budget_gb'], 'predicted_ppl': predict_fn(r['total_loss'])}
                for r in sweep_results
            ]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
