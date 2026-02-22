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
| `fine_tuning/` | LoRA fine-tuning: **train_lora.py** (config, checkpoints, FSDP). |
| `evaluation/`  | **evaluate_finetuned.py**: load fine-tuned model (vLLM), evaluate, compare to baseline. |
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

### 4. Fine-tune

**Single script:** `fine_tuning/train_lora.py`. Training uses the **train** split; **val** is used for validation. Checkpoints and the final PEFT adapter are written to `fine_tuning/checkpoints/<model>_r<N>/` (or `<model>/` if you omit `--lora-rank`).

**Multi-GPU LoRA with FSDP (recommended):**

```bash
# Same dataset/env as your data; use --lora-rank 32 (or 64) for more capacity
accelerate launch --config_file fine_tuning/accelerate_fsdp_config.yaml \
  fine_tuning/train_lora.py --dataset friends --lora-rank 32 --fsdp
```

- **Output:** `fine_tuning/checkpoints/qwen3-4b-instruct_r32/checkpoint-*` during training; at the end the PEFT adapter is saved to `.../final_model/` (adapter_config.json + weights + tokenizer). The config uses 4 GPUs by default; override with `--num_processes N` if needed.
- **If training stops before saving:** Run the same command and add `--resume-from-checkpoint fine_tuning/checkpoints/qwen3-4b-instruct_r32/checkpoint-1008` (use your latest checkpoint). The script loads that checkpoint, does zero new steps, and writes the adapter to `final_model/`.

**Single-GPU (no FSDP):**

```bash
python fine_tuning/train_lora.py --dataset friends --lora-rank 32
```

**Options:** `--dataset` (ami | friends | spgi | all). `--lora-rank N` writes to `checkpoints/<model>_r<N>/`. `--max-length 0` (default) means no truncation; use `--max-length 512` or `1024` only for memory-limited runs (see `fine_tuning/CONSTRAINTS.md`). With `--dataset all`, train/val from ami, friends, and spgi are combined. Balanced SPEAK/SILENT batching is used. After training, a **training curve** is saved as `training_curve.png` in the run directory.

### 5. Evaluate fine-tuned model

**Single script:** `evaluation/evaluate_finetuned.py`. Uses **vLLM** for fast inference. Evaluation runs on the **test** split and loads the PEFT adapter from `fine_tuning/checkpoints/<model>_r<N>/final_model/` (or `final_model` under `<model>/` if no `--lora-rank`).

```bash
# Match --dataset, --model, and --lora-rank to your training run
python evaluation/evaluate_finetuned.py --dataset friends --model qwen3-4b-instruct --lora-rank 32
```

Results are written to `evaluation/results/` (e.g. `finetuned_results_friends_qwen3-4b-instruct_r32.json` and `baseline_vs_finetuned_...`). To evaluate a specific checkpoint instead of `final_model`, use `--checkpoint checkpoint-1008` (default is `--checkpoint final`).

---

## Configuration

- **Dataset:** `DATASET` env or `--dataset` (ami | friends | spgi | **all**). Used by prepare_data, validate_data, evaluate_baseline, fine_tuning, evaluation. For fine-tuning, **all** combines train/val from ami, friends, and spgi into one run.
- **Model:** `MODEL` env or `--model` (qwen2.5-7b | qwen3-4b-instruct | qwen3-8b | llama3.1-8b-instruct | gpt-oss-20b | mistral-7b-instruct). Defined in `fine_tuning/config.py`; benchmarking and evaluation use the same options. Default is qwen3-4b-instruct.
- **System prompt:** Single `SYSTEM_PROMPT` in `benchmarking/evaluate_baseline.py`. Training and fine-tuned evaluation use it once; baseline benchmarking uses `--system-prompt-repeat` 1 or 2 (run_benchmark runs both).

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

---

## Fine-tuning: memory and compute

Inputs and outputs are **not truncated** by default (`--max-length 0`). For resource-limited runs, use `--max-length 512` or `1024`; see **`fine_tuning/CONSTRAINTS.md`** for:

- DDP vs FSDP vs single-GPU caps (e.g. 22GB GPU, max_length cap 1536 for DDP)
- Batch size and gradient accumulation when using a max length
- System prompt usage (training uses prompt once; evaluation uses it twice)
- Recommended commands for full-context vs memory-limited training
