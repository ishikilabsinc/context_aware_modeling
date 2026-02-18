# Context-Aware Turn-Taking Modeling

This repository trains and evaluates language models for **context-aware turn-taking**: deciding whether a target speaker should **SPEAK** or **STAY SILENT** after a pause in multi-party conversation, using full dialogue context.

## Overview

- **Task:** Given conversation history and the most recent utterance, predict if the target speaker should take the turn (SPEAK) or remain silent (SILENT).
- **Models:** Qwen family (Qwen2.5-7B, Qwen3-4B-Instruct, Qwen3-8B). Baseline evaluation and LoRA fine-tuning.
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

This loads from `data/<dataset>/`, splits into train/val/test, and writes:

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

- **Models:** `qwen2.5-7b` | `qwen3-4b-instruct` | `qwen3-8b`
- **Datasets:** those with a test split (e.g. ami, friends, spgi after prepare_data)
- **System prompt repeat:** 1 and 2

Then it writes:

- **Prediction JSONs:** `benchmarking/results/baseline_predictions_<dataset>_<model>_sp<N>.json` (per-sample predictions and run metadata)
- **Comparison report:** `benchmarking/results/benchmark_comparison.txt` (and `.json`) with accuracy, FPR, FNR, latency per run
- **Per-run analysis:** `benchmarking/results/reports/baseline_analysis_<dataset>_<model>_sp<N>.txt` for each run

Metrics are computed from the prediction files when reports are generated. To change metrics definitions, edit `benchmarking/metrics.py` and re-run reporting only (no need to re-run the model).

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

Set `MODEL` and `DATASET` (and optionally `HF_TOKEN`), then train:

```bash
export DATASET=ami
export MODEL=qwen2.5-7b
python fine_tuning/train_lora.py
```

Checkpoints (per model) are under `fine_tuning/checkpoints/<model_key>/`.

### 5. Evaluate fine-tuned model

```bash
# Uses DATASET and MODEL from config / env; loads LoRA from fine_tuning/checkpoints/<model>/
python evaluation/evaluate_finetuned.py
```

---

## Configuration

- **Dataset:** `DATASET` env or `--dataset` (ami | friends | spgi). Used by prepare_data, validate_data, evaluate_baseline, fine_tuning, evaluation.
- **Model:** `MODEL` env or `--model` in benchmarking (qwen2.5-7b | qwen3-4b-instruct | qwen3-8b). Defined in `fine_tuning/config.py`; benchmarking and evaluation read the same options. If you set `MODEL`, it must be one of these keys or the process will raise at import (e.g. when starting run_benchmark or train_lora). Omit `MODEL` to use the default (qwen2.5-7b).
- **System prompt:** Single `SYSTEM_PROMPT` in `benchmarking/evaluate_baseline.py`; repeated 1 or 2 times via `--system-prompt-repeat` (run_benchmark runs both).

---

## Metrics

- **Classification:** Accuracy (SPEAK vs SILENT), per-category (e.g. I1–I3, S1–S5), confusion matrix.
- **Rates:** False positive (SILENT→SPEAK), false negative (SPEAK→SILENT).
- **Latency:** Mean, p50, p95, p99 (reported in baseline and evaluation).

---

## Data format

Samples are JSONL with at least:

- `context_turns`: list of `{speaker, text}` before the decision point.
- `current_turn`: `{speaker, text}` of the most recent utterance (after which we decide).
- `target_speaker`, `decision` (SPEAK | SILENT), optional `category`, `confidence`.

Training/eval prompt format: system + instruction + context + current utterance → model outputs `<decision>SPEAK|SILENT</decision>` (and optionally reasoning/confidence). See `benchmarking/evaluate_baseline.py` and `fine_tuning/data_loader.py` for the exact prompt layout.
