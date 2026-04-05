<p align="center">
  <a href="https://baa.ai">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="./baa-logo-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="./baa-logo.svg">
      <img alt="Black Sheep AI" src="./baa-logo.svg" width="390">
    </picture>
    <br>https://baa.ai
  </a>
</p>

# MINT: Mixed-precision Integer Quantization via Rate-Distortion Optimization

**MINT** (**M**emory-**I**nformed **N**-bit **T**uning) is a data-free, per-tensor mixed-precision quantization framework for large language models. Given a user-specified memory budget, MINT jointly selects the optimal (bit-width, group-size) configuration for each weight tensor by solving a Multiple-Choice Knapsack Problem (MCKP) over per-tensor rate-distortion curves.

**No calibration data. No gradient computation. Under 50 minutes on commodity hardware.**

**You choose the exact model size you want, MINT finds the optimal quantization for that size.**

> **Paper**: [MINT: Budget-Aware Data-Free Mixed-Precision Quantization for LLMs via Rate-Distortion Optimization](https://huggingface.co/spaces/baa-ai/MINT) (preprint)

## Key Results

| Model | Size | MINT PPL | Baseline PPL | Delta |
|-------|------|----------|-------------|-------|
| **Qwen3.5-35B-A3B** | **28.3 GB (42% of BF16)** | **6.587** | **6.586 (BF16)** | **+0.0%** |
| Qwen3-30B-A3B | 19 GB (33% of BF16) | 8.798 | 8.789 (BF16) | +0.1% |
| Mixtral-8x7B | 24.5 GB | 4.264 | 4.387 (uniform 4-bit) | **-2.8%** |
| Llama-4-Scout (109B) | 58 GB | 7.703 | 7.899 (uniform 4-bit) | **-2.5%** |

### vs GPTQ (calibration-based)

MINT consistently outperforms GPTQ despite being entirely data-free:

| Model | MINT PPL | GPTQ PPL | Delta |
|-------|----------|----------|-------|
| Qwen3-30B-A3B | 9.020 | 9.160 | **-1.5%** |
| Qwen2-57B-A14B | 6.356 | 6.396 | **-0.6%** |
| Mixtral-8x7B | 4.266 | 4.640 | **-4.6%** |

## How It Works

```
BF16 Model (HuggingFace safetensors)
    │
    ▼
[Step 1] compute_rd_curves.py     ── NRMSE + SQNR at 13 (bits, group_size) configs per tensor
    │
    ├─── predict_quality.py       ── Sweep budgets, graph quality curve, pick your target
    │         (optional)              (opens graph automatically — no conversion needed)
    ▼
[Step 2] allocator.py             ── MCKP solver: pick (bits, gs) per tensor under your budget
    │
    ▼
[Step 3] build_manifest.py        ── Bundle allocation + tensor metadata → manifest JSON
    │
    ├──────────────────────┐
    ▼                      ▼
[Step 4a] convert.py       [Step 4b] convert_gguf.py
    MLX model                 GGUF model
    (Apple Silicon)            (Ollama / llama.cpp / LM Studio)
    │
    ▼
[Step 5] eval_perplexity.py ── WikiText-2 PPL (mean, median, trimmed)
```

## Quick Start

### Install

```bash
pip install torch safetensors mlx mlx-lm numpy scipy datasets
```

> `torch` is only needed for Step 1 (rate-distortion analysis on CPU). It is not needed at inference time. `scipy` is optional (LP/ILP solvers; the default greedy solver has no dependency).

### Full Pipeline (MLX — Apple Silicon)

```bash
# Step 1: Analyze model (~30 min for 30B, ~2 hours for 100B+)
python compute_rd_curves.py \
    --model-dir /path/to/Model-BF16 \
    --output rd_curves.json

# Step 2: Allocate to your memory budget (< 1 second)
python allocator.py \
    --rd-curves rd_curves.json \
    --budget-gb 28.0 \
    --output allocation.json

# Step 3: Build manifest
python build_manifest.py \
    --allocation allocation.json \
    --model-dir /path/to/Model-BF16 \
    --output manifest.json

# Step 4: Convert to MLX quantized model
python convert.py \
    --hf-path /path/to/Model-BF16 \
    --mlx-path /path/to/Model-MINT \
    --manifest manifest.json

# Step 5: Evaluate perplexity
python eval_perplexity.py \
    --model /path/to/Model-MINT \
    --output ppl_results.json
```

### Full Pipeline (GGUF — Any Platform)

Steps 1–3 are the same. Then use `convert_gguf.py`:

```bash
# Steps 1-3: same as above ...

# Step 4b: Convert to GGUF (requires llama.cpp)
python convert_gguf.py \
    --model-dir /path/to/Model-BF16 \
    --allocation allocation.json \
    --output Model-MINT.gguf \
    --llama-cpp /path/to/llama.cpp

# Use with Ollama
ollama create mymodel -f <(echo "FROM ./Model-MINT.gguf")
ollama run mymodel
```

### Multiple Budgets from One Analysis

Step 1 only runs once. Re-run Step 2 with different budgets (< 1 sec each):

```bash
python compute_rd_curves.py --model-dir /path/to/Model --output rd_curves.json

python allocator.py --rd-curves rd_curves.json --budget-gb 16 --output alloc-16gb.json
python allocator.py --rd-curves rd_curves.json --budget-gb 24 --output alloc-24gb.json
python allocator.py --rd-curves rd_curves.json --budget-gb 48 --output alloc-48gb.json
python allocator.py --rd-curves rd_curves.json --min-safe   --output alloc-min-safe.json
```

### Predict Quality Before Converting

`predict_quality.py` sweeps budget levels and estimates perplexity **before you spend time on conversion and evaluation**. It auto-opens an interactive quality-vs-size graph so you can pick the right budget visually.

```bash
# Sweep budgets and show the quality curve (graph opens automatically)
python predict_quality.py --rd-curves rd_curves.json --max-gb 50

# With calibration — measure PPL at one budget, predict all others
python predict_quality.py --rd-curves rd_curves.json --max-gb 50 \
    --calibrate 20.0:6.693

# Two calibration points for higher accuracy (r=0.97, RMSE ~0.01 PPL)
python predict_quality.py --rd-curves rd_curves.json --max-gb 50 \
    --calibrate 20.0:6.693 --calibrate 30.0:6.587

# Save graph to a specific file
python predict_quality.py --rd-curves rd_curves.json --max-gb 50 \
    --graph quality_curve.png

# Table only, no graph
python predict_quality.py --rd-curves rd_curves.json --max-gb 50 --no-graph
```

The graph shows:
- **Predicted PPL curve** vs model size (with calibration) or allocation loss (without)
- **BF16 reference line** with +1% and +2% quality guides
- **Calibration points** (measured PPL) plotted as stars
- **Average bits** on the top axis
- **Knee annotation** where 3-bit allocations drop out

Without calibration, the tool shows relative quality (allocation loss) — useful for comparing budgets and finding diminishing returns. With one or two calibration points (a single PPL evaluation), it fits a prediction curve and estimates absolute PPL at every budget.

## Allocator Options

### `--budget-gb <size>` / `--min-safe`

Set your target model size in GB. `--min-safe` produces the smallest model that respects the SQNR safety floor (typically all 3-bit).

### `--speed-mode <fast|balanced|full>`

Controls which bit widths the allocator may use. Default: **`balanced`**.

| Mode | Bits Allowed | Use When |
|------|-------------|----------|
| `fast` | 2, 4, 8, 16 | You want fastest possible MLX dequantization |
| **`balanced`** | 2, 3, 4, 6, 8, 16 | **Default.** Best quality/speed tradeoff |
| `full` | 2, 3, 4, 5, 6, 8, 16 | You want maximum PPL optimization |

**Why `balanced` is the default:** 6-bit fills the 2× cost gap between 4-bit and 8-bit. On Qwen3.5-35B at 30 GB, balanced mode matches BF16 perplexity at 42% of the size and achieves +1.5% higher generation throughput than `fast` mode (a mostly-6-bit model has more uniform dequantization). The 5-bit option (dropped in balanced) has the worst MLX Metal dequant overhead due to irregular bit packing.

```bash
# Balanced (default — recommended)
python allocator.py --rd-curves rd_curves.json --budget-gb 30.0 --output alloc.json

# Fastest inference
python allocator.py --rd-curves rd_curves.json --budget-gb 30.0 --speed-mode fast --output alloc.json

# Maximum quality
python allocator.py --rd-curves rd_curves.json --budget-gb 30.0 --speed-mode full --output alloc.json
```

### `--moe-aggregation <weighted_mean|max>`

How to aggregate NRMSE across MoE experts. Default: `weighted_mean`.

## Pipeline Details

### Step 1: Rate-Distortion Curves

Simulates group-wise RTN quantization at **13 configurations**:

```
2-bit:  (2,32)  (2,64)
3-bit:  (3,32)  (3,64)
4-bit:  (4,32)  (4,64)  (4,128)
5-bit:  (5,32)  (5,64)
6-bit:  (6,32)  (6,64)
8-bit:  (8,64)  (8,128)
16-bit: (16,0)  ← zero-distortion anchor
```

Measures NRMSE and SQNR per tensor. 3D MoE tensors use worst-case across experts. Higher-dimensional tensors (vision patch embeddings) are reshaped to 2D.

### Step 2: MCKP Allocation

```
minimize   Σ  prior_i × NRMSE_i(bits_i, gs_i)
subject to Σ  size_i(bits_i, gs_i)  ≤  Budget
```

- **SQNR safety veto** (9 dB) — blocks catastrophic 2-bit
- **Soft priors** — ∞ for norms/embeddings, 3× first layer, 2× last layer
- **MoE grouping** — all experts in a SwitchLinear share quantization
- **Speed modes** — control the bit-width search space

### Step 4a: MLX Conversion

`bridge.py` maps manifest tensor names to MLX module paths, handling:
- VLM prefix rewriting (`model.language_model.X` ↔ `language_model.model.X`)
- SwitchLinear expert aggregation (mode of expert bit-widths)
- Fused `gate_up_proj` → separate `gate_proj`/`up_proj` splitting
- 2/3/4/5/6/8-bit returns with per-tensor group size

### Step 4b: GGUF Conversion

Maps MINT allocations to GGUF quant types:

| MINT bits | GGUF type |
|-----------|-----------|
| 2 | Q2_K |
| 3 | Q3_K |
| 4 | Q4_K |
| 5 | Q5_K |
| 6 | Q6_K |
| 8 | Q8_0 |
| 16 | F16 |

Requires llama.cpp: `brew install llama.cpp` or build from [source](https://github.com/ggml-org/llama.cpp).

## Pre-quantized Models

Available on [HuggingFace baa-ai](https://huggingface.co/baa-ai):

### MLX (Apple Silicon)

| Model | Size | PPL | Link |
|-------|------|-----|------|
| **Qwen3.5-35B-A3B (balanced)** | **28.3 GB** | **6.587** | [baa-ai/Qwen3.5-35B-A3B-MINT-MLX-28GB](https://huggingface.co/baa-ai/Qwen3.5-35B-A3B-MINT-MLX-28GB) |
| Qwen3.5-35B-A3B (21 GB) | 21 GB | 6.713 | [baa-ai/Qwen3.5-35B-A3B-MINT-MLX-21GB](https://huggingface.co/baa-ai/Qwen3.5-35B-A3B-MINT-MLX-21GB) |
| Qwen3.5-122B-A10B (3-bit) | 52 GB | 6.07 | [baa-ai/Qwen3.5-122B-A10B-MINT-3bit-MLX](https://huggingface.co/baa-ai/Qwen3.5-122B-A10B-MINT-3bit-MLX) |
| Qwen3-30B-A3B | 17 GB | 8.97 | [baa-ai/Qwen3-30B-A3B-MINT-4bit-MLX](https://huggingface.co/baa-ai/Qwen3-30B-A3B-MINT-4bit-MLX) |

### GGUF (Ollama / llama.cpp / LM Studio)

| Model | Sizes | Link |
|-------|-------|------|
| Qwen3.5-35B-A3B | 15–48 GB | [baa-ai/Qwen3.5-35B-A3B-MINT-*-GGUF](https://huggingface.co/baa-ai) |
| Qwen3-30B-A3B | 17 GB | [baa-ai/Qwen3-30B-A3B-MINT-GGUF](https://huggingface.co/baa-ai/Qwen3-30B-A3B-MINT-GGUF) |

## File Reference

| File | Step | Description |
|------|------|-------------|
| `compute_rd_curves.py` | 1 | Rate-distortion analysis (13 configs per tensor) |
| `allocator.py` | 2 | MCKP solver with `--speed-mode` and `--budget-gb` |
| `build_manifest.py` | 3 | Allocation → manifest JSON |
| `bridge.py` | 4a | Manifest → MLX `quant_predicate` |
| `convert.py` | 4a | MLX model conversion |
| `convert_gguf.py` | 4b | GGUF conversion via llama.cpp |
| `eval_perplexity.py` | 5 | WikiText-2 perplexity evaluation |
| `run_experiment.py` | 4a+5 | Orchestrator: convert + eval in one command |
| `predict_quality.py` | 1→2 | Predict PPL before conversion (sweep + graph) |
| `analyze_allocation.py` | — | Inspect/compare allocations |

## Requirements

- Python 3.10+
- macOS with Apple Silicon for MLX inference (Steps 4a, 5)
- Any platform for analysis (Steps 1-3) and GGUF conversion (Step 4b)
- ~2× model size in RAM for Step 1

```
# Core (Steps 1-3)
torch          # CPU tensor loading for RD analysis
safetensors    # Read model shards
numpy          # Numerical operations

# MLX path (Steps 4a, 5)
mlx            # Apple Silicon ML framework
mlx-lm         # Model loading and conversion

# GGUF path (Step 4b)
# brew install llama.cpp

# Evaluation
datasets       # WikiText-2 download

# Optional
scipy          # LP/ILP solvers (greedy default has no dependency)
matplotlib     # Quality curve graph (predict_quality.py)
```

## Citation

```bibtex
@article{mint2026,
  title={MINT: Budget-Aware Data-Free Mixed-Precision Quantization
         for Large Language Models via Rate-Distortion Optimization},
  author={baa.ai},
  year={2026},
  url={https://github.com/baa-ai/MINT}
}
```

## License

PolyForm Noncommercial 1.0.0 — see [LICENSE](LICENSE).
