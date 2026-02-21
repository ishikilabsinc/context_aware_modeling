# Context-Aware Turn-Taking Modeling

This repository trains and evaluates language models for **context-aware turn-taking**: deciding whether a target speaker should **SPEAK** or **STAY SILENT** after a pause in multi-party conversation, using full dialogue context.

## Overview

- **Task:** Given conversation history and the most recent utterance, predict if the target speaker should take the turn (SPEAK) or remain silent (SILENT).
- **Models:** Qwen (Qwen2.5-7B, Qwen3-4B-Instruct, Qwen3-8B), Llama 3.1-8B-Instruct, GPT-OSS-20B, Mistral-7B-Instruct. Baseline evaluation and LoRA fine-tuning.
- **Datasets:** AMI, Friends, SPGI (each with train/val/test splits).
- **Pipeline:** Prepare data → Validate (optional) → Run benchmark (all baselines) or single-run eval → Fine-tune (optional) → Evaluate fine-tuned.

---

## Repository Structure

| Directory       | Purpose |
|----------------|---------|
| `data/`        | Raw and processed data per dataset (train/val/test). Gitignored. |
| `benchmarking/`| Prepare data, validate, **run_benchmark** (all models x datasets x system prompt 1&2), or single-run evaluate_baseline. |
| `fine_tuning/` | LoRA fine-tuning (config, train, checkpoints). |
| `evaluation/`  | Load fine-tuned model, evaluate, compare to baseline, reports. |
| `utils/`       | Shared constants and data utilities. |
| `ami/`         | AMI pipeline (stages for data prep, labeling, etc.). |

---

## Setup

**Python:** 3.9 or 3.10 recommended (not tested on 3.11+).

From the repo root, install dependencies once:

```bash
pip install -r requirements.txt
```

This covers benchmarking, fine-tuning, evaluation, and the optional AMI pipeline. Use a virtual environment and a Python with CUDA support for GPU runs.

---

## Quick Start

### 1. Prepare data

From repo root:

```bash
# One dataset (default: ami)
python benchmarking/prepare_data.py --dataset ami

# Others
python benchmarking/prepare_data.py --dataset friends
python benchmarking/prepare_data.py --dataset spgi
```

This loads sample files from `data/<dataset>/` (metadata such as `filtering_summary.json` is excluded), splits into train/val/test, and writes:

- `data/<dataset>/train/train_samples.jsonl`
- `data/<dataset>/val/val_samples.jsonl`
- `data/<dataset>/test/test_samples.jsonl`

### 2. Validate data (optional)

```bash
python benchmarking/validate_data.py --dataset ami
# Or: --dataset friends, --dataset spgi. Omit --dataset to use default ami (one dataset per run).
```

### 3. Run benchmark (all models, datasets, system prompt repeats)

From repo root, run the full benchmark:

```bash
python benchmarking/run_benchmark.py
```

This runs baseline evaluation for **every** combination of:

- **Models:** `qwen2.5-7b` | `qwen3-4b-instruct` | `qwen3-8b` | `llama3.1-8b-instruct` | `gpt-oss-20b` | `mistral-7b-instruct`
- **Datasets:** those with a test split (e.g. ami, friends, spgi after prepare_data)
- **System prompt repeat:** 1 and 2

Then it writes:

- **Prediction JSONs:** `benchmarking/results/baseline_predictions_<dataset>_<model>_sp<N>.json` — per-sample predictions and run metadata only; no metrics.
- **Comparison report:** `benchmarking/results/reports/benchmark_comparison.txt` and `benchmarking/results/benchmark_comparison.json` — one row per (dataset, model, SP) with Acc, MacroAcc, MacroF1, FPR, FNR, latency mean, N, nulls; the JSON includes the full set (precision/recall/F1 for Speak and Silent, etc.).
- **Per-run analysis:** `benchmarking/results/reports/baseline_analysis_<dataset>_<model>_sp<N>.txt` — overall metrics, precision/recall/F1, error rates, latency stats, per-category accuracy, confusion matrix, error analysis.
- **Category tables:** `benchmarking/results/reports/category_table_<dataset>.txt` — tabular overall metrics and per-category accuracy grid for that dataset.

Metrics are computed from the prediction files when reports are generated. To change metric definitions, edit `benchmarking/metrics.py` and re-run reporting only (`--skip-run`).

Options:

```bash
# Regenerate reports from existing prediction files (no model runs; recomputes metrics)
python benchmarking/run_benchmark.py --skip-run

# Limit scope
python benchmarking/run_benchmark.py --datasets ami spgi --models qwen2.5-7b --system-prompt-repeats 1 2

# With 4 GPUs, up to 4 evaluations run in parallel (one GPU per run). Override with --max-parallel N or --max-parallel 1 for sequential.
```

For a single (dataset, model, repeat) run, use:

```bash
python benchmarking/evaluate_baseline.py --dataset ami --model qwen2.5-7b --system-prompt-repeat 1
```

Add `--debug-prompts` to print the full system prompt, instruction, and model input for the first few samples (useful for inspecting prompt formatting):

```bash
python benchmarking/evaluate_baseline.py --dataset ami --model qwen2.5-7b --system-prompt-repeat 1 --debug-prompts
```

### 4. Fine-tune (optional)

Training uses the **train** split; the **val** split is used for validation during training. Set `MODEL` and optionally `DATASET`, then train:

```bash
export MODEL=qwen2.5-7b
# Single dataset (default: ami)
python fine_tuning/train_lora.py --dataset ami

# All three datasets combined (single training run)
python fine_tuning/train_lora.py --dataset all

# Try higher LoRA rank for more capacity (saved to checkpoints/<model>_r32/ or _r64/)
python fine_tuning/train_lora.py --dataset all --lora-rank 32
python fine_tuning/train_lora.py --dataset all --lora-rank 64

# FSDP: reduce OOM by sharding model across GPUs (use with accelerate launch)
# Config uses 4 GPUs by default; override with --num_processes N if needed
accelerate launch --config_file fine_tuning/accelerate_fsdp_config.yaml fine_tuning/train_lora.py --fsdp --dataset all
```

With `--dataset all`, train (and val) samples from ami, friends, and spgi are loaded and concatenated. With **`--fsdp`**, the model is loaded on CPU and sharded across GPUs (Fully Sharded Data Parallel) to cut memory per GPU and avoid OOM; you must run with **`accelerate launch`** and 2+ processes (e.g. `fine_tuning/accelerate_fsdp_config.yaml` uses 4 GPUs by default; override with `--num_processes N`). Balanced SPEAK/SILENT batching (~50% each per batch) is used for both single-GPU and FSDP runs (a distributed balanced batch sampler partitions data across ranks when using `--fsdp`). With `--lora-rank N`, LoRA rank is set to N (default from config is 8); output is written to `checkpoints/<model>_r<N>/` so you can compare runs. **`--max-length N`** sets the training sequence length (default 256 for fast runs; use 512 or 1024 for benchmark-aligned training; 1024 may require smaller batch size in config). Evaluate with `python evaluation/evaluate_finetuned.py --dataset ami --model qwen2.5-7b --lora-rank 32`. Checkpoints (per model) are under `fine_tuning/checkpoints/<model_key>/`. For faster training with no impact on model quality: **Flash Attention 2** is used automatically when `flash-attn` is installed; otherwise **SDPA** (PyTorch built-in) is used for faster attention with no extra install. Eval/save run every 4000 steps; fused optimizer is used on CUDA (set `optim` to `adamw_torch` in `fine_tuning/config.py` if your environment does not support it). After each run, a **training curve** (train loss, eval loss, learning rate) is saved as `training_curve.png` in that directory to inspect stability; you can also run `python fine_tuning/plot_training_curve.py --model <key>` to regenerate from `trainer_state.json`.

### 5. Evaluate fine-tuned model

Evaluation runs on the **test** split. Uses dataset and model from config/env; loads LoRA from `fine_tuning/checkpoints/<model>/`:

```bash
python evaluation/evaluate_finetuned.py --dataset ami --model qwen2.5-7b
```

To see if the needle moved after the first checkpoint (e.g. one epoch), evaluate a specific checkpoint. Results are written to separate files so you can compare baseline vs first checkpoint vs final:

```bash
python evaluation/evaluate_finetuned.py --dataset ami --model qwen2.5-7b --checkpoint checkpoint-2000
# Default is --checkpoint final (the best model saved at end of training).
```

---

## Configuration

- **Dataset:** `DATASET` env or `--dataset` (ami | friends | spgi | **all**). Used by prepare_data, validate_data, evaluate_baseline, fine_tuning, evaluation. For fine-tuning, **all** combines train/val from ami, friends, and spgi into one run.
- **Model:** `MODEL` env or `--model` in benchmarking (qwen2.5-7b | qwen3-4b-instruct | qwen3-8b | llama3.1-8b-instruct | gpt-oss-20b | mistral-7b-instruct). Defined in `fine_tuning/config.py`; benchmarking and evaluation read the same options. If you set `MODEL`, it must be one of these keys or the process will raise at import (e.g. when starting run_benchmark or train_lora). Omit `MODEL` to use the default (qwen2.5-7b).
- **System prompt:** Single `SYSTEM_PROMPT` in `benchmarking/evaluate_baseline.py`; repeated 1 or 2 times via `--system-prompt-repeat` (run_benchmark runs both).

---

## Metrics

- **Classification:** Accuracy, macro accuracy (class-balanced), per-category accuracy (data-driven: e.g. SPEAK_explicit, SPEAK_implicit, SILENT_no_ref, SILENT_ref), confusion matrix.
- **Precision / recall / F1:** Speak and Silent precision, recall, and F1; macro F1 (SPEAK = positive class).
- **Rates:** False positive (SILENT→SPEAK), false negative (SPEAK→SILENT).
- **Latency:** Mean, median, p50, p95, p99, min, max (reported in baseline and evaluation).

---

## Data format

Samples are JSONL with at least:

- `context_turns`: list of `{speaker, text}` before the decision point.
- `current_turn`: `{speaker, text}` of the most recent utterance (after which we decide).
- `target_speaker`, `decision` (SPEAK | SILENT), optional `category`, `confidence`.

Training/eval prompt format: system + instruction + context + current utterance → model outputs `<decision>SPEAK|SILENT</decision>` (and optionally reasoning/confidence). See `benchmarking/evaluate_baseline.py` and `fine_tuning/data_loader.py` for the exact prompt layout.
