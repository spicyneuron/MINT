# MINT: Mixed-precision Integer Quantization via Rate-Distortion Optimization

**MINT** (**M**emory-**I**nformed **N**-bit **T**uning) is a data-free, per-tensor mixed-precision quantization framework for large language models. Given a user-specified memory budget, MINT jointly selects the optimal (bit-width, group-size) configuration for each weight tensor by solving a Multiple-Choice Knapsack Problem (MCKP) over per-tensor rate-distortion curves.

**No calibration data. No gradient computation. Under 50 minutes on commodity hardware.**

> **Paper**: [MINT: Compute-Optimal Data-Free Mixed-Precision Quantization for Large Language Models via Rate-Distortion Optimization](https://huggingface.co/spaces/baa-ai/MINT) (preprint)

## Key Results

| Model | Size | MINT PPL | Uniform 4-bit PPL | Delta |
|-------|------|----------|-------------------|-------|
| Qwen3-30B-A3B | 15.5 GB (26% of BF16) | 8.798 | 9.031 | **-2.6%** |
| Qwen2-57B-A14B | 32 GB | 8.236 | 8.346 | **-1.3%** |
| Mixtral-8x7B | 24.5 GB | 5.370 | 5.631 | **-4.6%** |
| Llama-4-Scout (109B) | 58 GB | 7.703 | 7.899 | **-2.5%** |
| MiniMax-M2.5 (229B) | 118 GB | 8.787 | 8.957 | **-1.9%** |

All comparisons at matched or smaller model size. PPL values are median perplexity on WikiText-2.

### vs GPTQ (calibration-based)

MINT consistently outperforms GPTQ despite being entirely data-free:

| Model | MINT PPL | GPTQ PPL | Delta |
|-------|----------|----------|-------|
| Qwen3-30B-A3B | 8.798 | 8.949 | **-1.7%** |
| Qwen2-57B-A14B | 8.236 | 8.315 | **-0.95%** |
| Mixtral-8x7B | 5.370 | 5.631 | **-4.6%** |

## How It Works

```
BF16 Model
    |
    v
[Step 1] compute_rd_curves.py     -- NRMSE + SQNR at 8 (bits, group_size) configs per tensor
    |
    v
[Step 2] allocator.py             -- MCKP solver: choose (bits, gs) per tensor under budget
    |
    v
[Step 3] build_manifest.py        -- Bundle allocation + tensor metadata into manifest JSON
    |
    v
[Step 4] convert.py               -- MLX quantization using manifest's per-tensor decisions
    |
    v
[Step 5] eval_perplexity.py       -- WikiText-2 perplexity (mean, median, trimmed)
```

MINT formulates quantization as a constrained optimization:

```
minimize   sum_i  prior_i * NRMSE_i(bits_i, gs_i)
subject to sum_i  size_i(bits_i, gs_i) <= Budget
```

Where each tensor has a rate-distortion curve measuring reconstruction error at 8 different (bits, group_size) configurations. The MCKP solver finds the provably optimal allocation in under 1 second.

**Key insight**: Group size is a first-class allocation variable. MINT frequently chooses g32 over g128 at the same bit-width -- the 4x more quantization groups provide better accuracy at minimal overhead, often yielding larger quality gains than bit-width upgrades.

## Quick Start

```bash
# Install dependencies
pip install torch safetensors mlx mlx-lm numpy scipy datasets

# Full pipeline for a model targeting 19 GB:
python compute_rd_curves.py --model-dir /path/to/Model-BF16 --output rd_curves.json
python allocator.py --rd-curves rd_curves.json --budget-gb 19.0 --output allocation.json
python build_manifest.py --allocation allocation.json --model-dir /path/to/Model-BF16 --output manifest.json
python convert.py --hf-path /path/to/Model-BF16 --mlx-path /path/to/Model-MINT --manifest manifest.json
python eval_perplexity.py --model /path/to/Model-MINT --output ppl_results.json
```

> **Note:** `torch` is only needed for Step 1 (rate-distortion analysis). It is not needed at inference time. `scipy` is optional (only for LP/ILP solvers; the default greedy solver has no dependency).

## Multiple Budget Points from One Analysis

The key advantage: Steps 1 and 3 only run once per model. Re-run Step 2 with different budgets (< 1 sec each):

```bash
# Analyze once (~30 min for 30B model)
python compute_rd_curves.py --model-dir /path/to/Model --output rd_curves.json

# Generate allocations at different budgets (< 1 sec each)
python allocator.py --rd-curves rd_curves.json --budget-gb 16 --output alloc-16gb.json
python allocator.py --rd-curves rd_curves.json --budget-gb 24 --output alloc-24gb.json
python allocator.py --rd-curves rd_curves.json --budget-gb 48 --output alloc-48gb.json
python allocator.py --rd-curves rd_curves.json --min-safe   --output alloc-min-safe.json
```

## Pipeline Steps

### Step 1: Compute Rate-Distortion Curves

```bash
python compute_rd_curves.py --model-dir /path/to/Model-BF16 --output rd_curves.json
```

Loads each safetensor shard and simulates quantization at 8 configurations: `(2,32), (2,64), (3,64), (4,32), (4,64), (4,128), (8,64), (8,128)`. Measures NRMSE and SQNR for every 2D tensor. 3D MoE expert tensors use worst-case across experts.

**Runtime:** ~30 min for 30B, ~2 hours for 100B+.

### Step 2: Run the Knapsack Allocator

```bash
python allocator.py --rd-curves rd_curves.json --budget-gb 19.0 --output allocation.json
```

Solves the MCKP with:
- **SQNR safety veto** (9 dB floor) -- blocks configurations causing severe degradation
- **Protection priors**: embeddings/lm_head/routers/norms hard-protected at 16-bit; first/last layers get 3x/2x loss multiplier
- **MoE expert grouping**: worst-case NRMSE across experts per (layer, projection)

**Runtime:** < 1 second for any model size.

### Step 3: Build Manifest

```bash
python build_manifest.py --allocation allocation.json --model-dir /path/to/Model-BF16 --output manifest.json
```

Combines allocation decisions with tensor metadata from model files.

### Step 4: Convert Model

```bash
python convert.py --hf-path /path/to/Model-BF16 --mlx-path /path/to/Model-MINT --manifest manifest.json
```

Creates an MLX `quant_predicate` that returns `{"bits": N, "group_size": G}` per tensor, then calls `mlx_lm.convert()`. Output is ready for `mlx_lm.load()`.

**Runtime:** 10-60 min depending on model size.

### Step 5: Evaluate Perplexity

```bash
python eval_perplexity.py --model /path/to/Model-MINT --num-samples 256 --output ppl.json
```

Reports standard, median, and trimmed mean perplexity on WikiText-2.

## MCKP Formulation

Each tensor *i* has a set of valid configurations *C_i* after SQNR veto:

- **Objective**: `minimize sum_i prior_i * NRMSE_i(b_i, g_i)`
- **Constraint**: `sum_i size_i(b_i, g_i) <= Budget`
- **NRMSE_i(b, g)**: reconstruction error from rate-distortion curve
- **prior_i**: protection multiplier (inf for embeddings/norms, 3x first layer, 2x last layer, 1x default)
- **size_i(b, g)**: `num_params * b/8 + (num_params / g) * 2` bytes

The default greedy solver starts all tensors at their lowest valid bit-width, sorts upgrade options by loss-reduction-per-byte, and greedily upgrades until the budget is exhausted. This achieves near-optimal solutions (typically within 0.1% of the LP relaxation bound).

## SQNR Safety Veto

Configurations with SQNR < 9 dB are vetoed. This exploits a natural gap in the SQNR distribution:
- 2-bit quantization: max observed 8.7 dB (catastrophic -- PPL triples)
- 3-bit quantization: min observed 10.4 dB (usable)

The 9 dB threshold sits cleanly in this gap, providing an absolute quality floor.

## MoE Expert Grouping

For Mixture-of-Experts models, MLX's `SwitchLinear` module requires all experts in a layer to share quantization parameters. MINT groups expert tensors by (layer, projection) using:
- Worst-case NRMSE across experts (conservative quality)
- Minimum SQNR across experts (safety)
- Sum of parameters (correct size accounting)

## Pre-quantized Models

Models quantized with MINT are available on HuggingFace under [baa-ai](https://huggingface.co/baa-ai):

### GGUF (cross-platform: llama.cpp, ollama, LM Studio)

| Model | Size | HuggingFace |
|-------|------|-------------|
| Mixtral-8x7B-Instruct | 26 GB | [baa-ai/Mixtral-8x7B-Instruct-SWAN-4bit-GGUF](https://huggingface.co/baa-ai/Mixtral-8x7B-Instruct-SWAN-4bit-GGUF) |
| Qwen3-30B-A3B | 16 GB | [baa-ai/Qwen3-30B-A3B-SWAN-4bit-GGUF](https://huggingface.co/baa-ai/Qwen3-30B-A3B-SWAN-4bit-GGUF) |

### MLX (Apple Silicon)

| Model | Size | HuggingFace |
|-------|------|-------------|
| Llama-4-Scout | 58 GB | [baa-ai/Llama-4-Scout-17B-16E-Instruct-SWAN-4bit-MLX](https://huggingface.co/baa-ai/Llama-4-Scout-17B-16E-Instruct-SWAN-4bit-MLX) |
| Llama-4-Maverick | 172 GB | [baa-ai/Llama-4-Maverick-17B-128E-Instruct-SWAN-4bit-MLX](https://huggingface.co/baa-ai/Llama-4-Maverick-17B-128E-Instruct-SWAN-4bit-MLX) |
| MiniMax-M2.5 | 118 GB | [baa-ai/MiniMax-M2.5-SWAN-4bit-MLX](https://huggingface.co/baa-ai/MiniMax-M2.5-SWAN-4bit-MLX) |
| GLM-4.7-Flash | 16 GB | [baa-ai/GLM-4.7-Flash-SWAN-4bit-MLX](https://huggingface.co/baa-ai/GLM-4.7-Flash-SWAN-4bit-MLX) |
| Llama-3.1-70B | 47 GB | [baa-ai/Llama-3.1-70B-Instruct-SWAN-5bit-MLX](https://huggingface.co/baa-ai/Llama-3.1-70B-Instruct-SWAN-5bit-MLX) |
| Llama-3.3-70B | 47 GB | [baa-ai/Llama-3.3-70B-Instruct-SWAN-5bit-MLX](https://huggingface.co/baa-ai/Llama-3.3-70B-Instruct-SWAN-5bit-MLX) |

## File Inventory

| File | Purpose |
|------|---------|
| `compute_rd_curves.py` | Step 1: Rate-distortion analysis |
| `allocator.py` | Step 2: MCKP budget-constrained solver |
| `build_manifest.py` | Step 3: Allocation + model metadata -> manifest |
| `bridge.py` | Manifest -> MLX quant_predicate function |
| `convert.py` | Step 4: MLX model conversion |
| `eval_perplexity.py` | Step 5: WikiText-2 perplexity evaluation |
| `analyze_allocation.py` | Optional: analyze/compare allocations |
| `run_experiment.py` | Optional: orchestrate convert + eval |
| `MINT.tex` | Paper source |

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4) for Steps 4-5
- Python 3.10+
- ~2x model size in RAM for Step 1 (loading BF16 weights)

### Dependencies

```
torch          # Step 1 only (not needed at inference)
safetensors    # Steps 1, 3
mlx            # Steps 4, 5
mlx-lm         # Steps 4, 5
numpy          # Steps 1, 2
scipy          # Optional (LP/ILP solvers)
datasets       # Step 5 (WikiText-2)
```

## Citation

If you use MINT in your research, please cite:

```bibtex
@article{mint2026,
  title={MINT: Compute-Optimal Data-Free Mixed-Precision Quantization for Large Language Models via Rate-Distortion Optimization},
  author={Kennedy, Trevor},
  year={2026},
  url={https://github.com/baa-ai/MINT}
}
```

## License

PolyForm Noncommercial 1.0.0 --- see [LICENSE](LICENSE) for details.
