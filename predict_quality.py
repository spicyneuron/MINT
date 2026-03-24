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


def plot_quality_curve(sweep_results, model_name, predict_fn=None, bf16_ppl=None,
                       calibration_points=None, output_path="quality_curve.png"):
    """Plot quality vs size curve."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    sizes = [r['actual_gb'] for r in sweep_results]
    losses = [r['total_loss'] for r in sweep_results]
    avg_bits = [r['avg_bits'] for r in sweep_results]

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))

    if predict_fn and bf16_ppl:
        # Plot predicted PPL
        ppls = [predict_fn(r['total_loss']) for r in sweep_results]
        ax1.plot(sizes, ppls, 'o-', color='#991b1b', linewidth=2.5, markersize=6,
                 label='MINT predicted PPL', zorder=3)

        # BF16 reference line
        ax1.axhline(y=bf16_ppl, color='#2563eb', linestyle='--', linewidth=1.5,
                     alpha=0.7, label=f'BF16 reference ({bf16_ppl:.3f})')

        # +1% and +2% reference lines
        ax1.axhline(y=bf16_ppl * 1.01, color='#16a34a', linestyle=':', linewidth=1,
                     alpha=0.5, label='+1% vs BF16')
        ax1.axhline(y=bf16_ppl * 1.02, color='#ca8a04', linestyle=':', linewidth=1,
                     alpha=0.5, label='+2% vs BF16')

        # Calibration points
        if calibration_points:
            cal_sizes = []
            cal_ppls_actual = []
            for cal_budget, cal_ppl in calibration_points:
                closest = min(sweep_results, key=lambda r: abs(r['budget_gb'] - cal_budget))
                cal_sizes.append(closest['actual_gb'])
                cal_ppls_actual.append(cal_ppl)
            ax1.scatter(cal_sizes, cal_ppls_actual, s=120, color='#991b1b', marker='*',
                        zorder=5, label='Measured PPL (calibration)')

        ax1.set_ylabel('Predicted Median PPL', fontsize=12)
        ylabel_text = 'Predicted Median Perplexity'
    else:
        # Plot allocation loss (no calibration)
        ax1.plot(sizes, losses, 'o-', color='#991b1b', linewidth=2.5, markersize=6,
                 label='Allocation loss (lower = better)', zorder=3)
        ax1.set_ylabel('Allocation Loss (NRMSE sum)', fontsize=12)
        ylabel_text = 'Allocation Loss'

    ax1.set_xlabel('Model Size (GB)', fontsize=12)
    ax1.set_title(f'MINT Quality vs Size: {model_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)

    # Add avg bits as secondary x-axis labels on top
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    # Pick a subset of tick positions
    tick_indices = list(range(0, len(sizes), max(1, len(sizes) // 8)))
    ax2.set_xticks([sizes[i] for i in tick_indices])
    ax2.set_xticklabels([f'{avg_bits[i]:.1f}b' for i in tick_indices], fontsize=9)
    ax2.set_xlabel('Average Bits', fontsize=10, labelpad=8)

    # Annotate the knee/sweet spot
    if len(sweep_results) >= 5:
        # Find where 3-bit drops out
        for i, r in enumerate(sweep_results):
            has_3bit = any(int(b) <= 3 and info['percentage'] > 1
                         for b, info in r['bits_distribution'].items())
            if not has_3bit and i > 0:
                knee_size = r['actual_gb']
                if predict_fn and bf16_ppl:
                    knee_ppl = predict_fn(r['total_loss'])
                    ax1.annotate(f'No 3-bit\n({knee_size:.0f} GB)',
                                xy=(knee_size, knee_ppl),
                                xytext=(knee_size + (sizes[-1] - sizes[0]) * 0.08,
                                        knee_ppl + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.08),
                                arrowprops=dict(arrowstyle='->', color='#666'),
                                fontsize=9, color='#666')
                break

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGraph saved to {output_path}")


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
    parser.add_argument("--graph", nargs="?", const="auto", default=None,
                        help="Save and open quality-vs-size graph. Optionally specify output path "
                             "(default: auto-generated temp file, opened in browser)")
    parser.add_argument("--no-graph", action="store_true", help="Suppress automatic graph display")
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

    # Graph — show by default unless --no-graph
    if not args.no_graph:
        import tempfile, platform, subprocess
        if args.graph and args.graph != "auto":
            graph_path = args.graph
        else:
            graph_path = tempfile.mktemp(suffix=".png", prefix="mint_quality_")
        plot_quality_curve(sweep_results, model_name, predict_fn, bf16_ppl,
                           calibration_points, graph_path)
        # Auto-open
        try:
            if platform.system() == "Darwin":
                subprocess.Popen(["open", graph_path])
            elif platform.system() == "Linux":
                subprocess.Popen(["xdg-open", graph_path])
            elif platform.system() == "Windows":
                subprocess.Popen(["start", graph_path], shell=True)
        except Exception:
            pass  # silently skip if no display

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
