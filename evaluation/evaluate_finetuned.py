#!/usr/bin/env python3
"""Evaluate fine-tuned LoRA model on test set and compare with baseline."""
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from fine_tuning.config import MODEL_OPTIONS, MODEL as DEFAULT_MODEL
import benchmarking.evaluate_baseline as baseline_module
from benchmarking.evaluate_baseline import extract_decision_from_output
from fine_tuning.data_loader import create_training_prompt, TRAINING_MODE_DECISION_ONLY, TRAINING_MODE_COT, TRAINING_MODES
from evaluation.load_finetuned import load_finetuned_model
from utils.data_utils import load_samples, filter_samples_with_context

USE_VLLM = True
BATCH_SIZE = 32 if USE_VLLM else 4
MAX_NEW_TOKENS = 32
MAX_NEW_TOKENS_COT = 256
TEMPERATURE = 0.0
STOP_AFTER_DECISION = True
STOP_SEQUENCES = ["</decision>"]
STOP_SEQUENCES_COT = ["</confidence>", "</decision>"]
DECISION_ONLY_PROMPT_MAX_LENGTH = 2048


def extract_reasoning_from_output(text: str) -> str:
    """Extract content between <reasoning> and </reasoning> for CoT evaluation."""
    if not text:
        return ""
    m = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def infer_with_vllm(
    model,
    tokenizer,
    prompts: List[str],
    max_tokens: int = MAX_NEW_TOKENS,
    stop_sequences: List[str] = None,
) -> List[Tuple[str, float]]:
    from vllm import SamplingParams

    stop_sequences = stop_sequences or STOP_SEQUENCES
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
        stop=stop_sequences,
        include_stop_str_in_output=True,
    )
    
    lora_adapter = getattr(model, '_lora_adapter_name', None)
    lora_path = getattr(model, '_lora_adapter_path', None)

    start_time = time.time()
    if lora_adapter and lora_path:
        try:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest(lora_adapter, 1, lora_path)
            outputs = model.generate(prompts, sampling_params, lora_request=lora_request)
        except Exception as e:
            print(f"Warning: Could not use LoRA adapter in generation: {e}")
            outputs = model.generate(prompts, sampling_params)
    else:
        outputs = model.generate(prompts, sampling_params)
    total_time = time.time() - start_time

    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        latency = total_time / len(prompts)

        if STOP_AFTER_DECISION and "</confidence>" not in (stop_sequences or []):
            decision_match = re.search(r"<decision>.*?</decision>", generated_text, re.DOTALL | re.IGNORECASE)
            if decision_match:
                generated_text = generated_text[: decision_match.end()]

        results.append((generated_text, latency))

    return results


def infer_with_model_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_tokens: int = MAX_NEW_TOKENS,
) -> List[Tuple[str, float]]:
    import torch

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    tokenizer.padding_side = original_padding_side

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=TEMPERATURE if TEMPERATURE > 0 else None,
            do_sample=TEMPERATURE > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    total_time = time.time() - start_time
    avg_latency = total_time / len(prompts)
    
    results = []
    input_lengths = inputs['attention_mask'].sum(dim=1).cpu().tolist()
    
    for i, output in enumerate(outputs):
        generated_ids = output[input_lengths[i]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if STOP_AFTER_DECISION:
            decision_match = re.search(r"<decision>.*?</decision>", generated_text, re.DOTALL | re.IGNORECASE)
            if decision_match:
                generated_text = generated_text[: decision_match.end()]

        results.append((generated_text, avg_latency))

    return results



def evaluate_samples(samples: List[Dict], model, tokenizer, mode: str = TRAINING_MODE_DECISION_ONLY) -> Dict:
    if mode not in TRAINING_MODES:
        raise ValueError(f"mode must be one of {TRAINING_MODES}, got {mode!r}")
    print(f"\nEvaluating {len(samples)} samples...")
    baseline_module.SYSTEM_PROMPT_REPEAT = 1

    is_vllm_model = hasattr(model, "_llm_engine") or (hasattr(model, "llm_engine") and model.llm_engine is not None)
    effective_batch_size = BATCH_SIZE if is_vllm_model else min(BATCH_SIZE, 4)
    print(f"Batch size: {effective_batch_size} ({'vLLM' if is_vllm_model else 'transformers'})")
    print(f"Eval mode: {mode}")

    use_cot = mode == TRAINING_MODE_COT
    max_tokens = MAX_NEW_TOKENS_COT if use_cot else MAX_NEW_TOKENS
    stop_sequences = STOP_SEQUENCES_COT if use_cot else STOP_SEQUENCES

    prompts = [
        create_training_prompt(sample, tokenizer, DECISION_ONLY_PROMPT_MAX_LENGTH, training_mode=mode).rstrip() + "\n"
        for sample in samples
    ]
    all_predictions = []
    all_latencies = []

    for i in range(0, len(prompts), effective_batch_size):
        batch_prompts = prompts[i : i + effective_batch_size]
        batch_samples = samples[i : i + effective_batch_size]
        batch_num = i // effective_batch_size + 1
        total_batches = (len(prompts) + effective_batch_size - 1) // effective_batch_size
        if batch_num % 10 == 0 or batch_num == 1:
            print(f"  Processing batch {batch_num}/{total_batches} "
                  f"({len(batch_prompts)} samples, {i+1}/{len(samples)} total)...", end=" ", flush=True)

        if USE_VLLM and is_vllm_model:
            batch_results = infer_with_vllm(
                model, tokenizer, batch_prompts, max_tokens=max_tokens, stop_sequences=stop_sequences
            )
        else:
            batch_results = infer_with_model_batch(model, tokenizer, batch_prompts, max_tokens=max_tokens)

        if batch_num % 50 == 0:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        
        for (output_text, latency), sample in zip(batch_results, batch_samples):
            prediction = extract_decision_from_output(output_text)
            ground_truth = sample.get("decision", "UNKNOWN")
            pred_dict = {
                "sample_id": sample.get("decision_point_id", f"sample_{i}"),
                "ground_truth": ground_truth,
                "prediction": prediction,
                "category": sample.get("category", "UNKNOWN"),
                "output_text": output_text,
                "latency": latency,
            }
            if use_cot:
                pred_dict["reasoning"] = extract_reasoning_from_output(output_text)
            all_predictions.append(pred_dict)
            all_latencies.append(latency)
        
        if batch_num % 10 == 0 or batch_num == 1:
            recent_latencies = all_latencies[-effective_batch_size*10:] if len(all_latencies) >= effective_batch_size*10 else all_latencies
            elapsed_samples = min(i + len(batch_prompts), len(samples))
            remaining_samples = len(samples) - elapsed_samples
            if len(recent_latencies) > 0:
                avg_lat = np.mean(recent_latencies)
                est_remaining = (avg_lat * remaining_samples) / BATCH_SIZE
                print(f"(avg: {avg_lat:.3f}s/sample, ~{est_remaining/60:.1f}min remaining)")
    
    correct = sum(1 for p in all_predictions if p['prediction'] == p['ground_truth'])
    accuracy = correct / len(all_predictions) if all_predictions else 0
    
    category_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    for pred in all_predictions:
        cat = pred['category']
        category_metrics[cat]['total'] += 1
        if pred['prediction'] == pred['ground_truth']:
            category_metrics[cat]['correct'] += 1
    
    confusion = defaultdict(int)
    for pred in all_predictions:
        key = f"{pred['ground_truth']}_->_{pred['prediction']}"
        confusion[key] += 1
    
    latencies = np.array(all_latencies)
    latency_stats = {
        'mean': float(np.mean(latencies)),
        'median': float(np.median(latencies)),
        'p50': float(np.percentile(latencies, 50)),
        'p95': float(np.percentile(latencies, 95)),
        'p99': float(np.percentile(latencies, 99)),
        'min': float(np.min(latencies)),
        'max': float(np.max(latencies)),
    }
    
    false_positives = sum(1 for p in all_predictions
                          if p['ground_truth'] == 'SILENT' and p['prediction'] == 'SPEAK')
    false_negatives = sum(1 for p in all_predictions
                          if p['ground_truth'] == 'SPEAK' and p['prediction'] == 'SILENT')
    true_positives = sum(1 for p in all_predictions
                         if p['ground_truth'] == 'SPEAK' and p['prediction'] == 'SPEAK')
    true_negatives = sum(1 for p in all_predictions
                         if p['ground_truth'] == 'SILENT' and p['prediction'] == 'SILENT')

    total_silent = sum(1 for p in all_predictions if p['ground_truth'] == 'SILENT')
    total_speak = sum(1 for p in all_predictions if p['ground_truth'] == 'SPEAK')

    fpr = false_positives / total_silent if total_silent > 0 else 0
    fnr = false_negatives / total_speak if total_speak > 0 else 0

    prec_speak = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    rec_speak = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_speak = 2 * prec_speak * rec_speak / (prec_speak + rec_speak) if (prec_speak + rec_speak) > 0 else 0.0
    prec_silent = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0.0
    rec_silent = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
    f1_silent = 2 * prec_silent * rec_silent / (prec_silent + rec_silent) if (prec_silent + rec_silent) > 0 else 0.0
    macro_f1 = (f1_speak + f1_silent) / 2.0
    balanced_accuracy = (rec_speak + rec_silent) / 2.0

    results = {
        'total_samples': len(all_predictions),
        'accuracy': accuracy,
        'correct': correct,
        'incorrect': len(all_predictions) - correct,
        'macro_f1': macro_f1,
        'balanced_accuracy': balanced_accuracy,
        'category_accuracy': {
            cat: {
                'accuracy': m['correct'] / m['total'] if m['total'] > 0 else 0,
                'correct': m['correct'],
                'total': m['total']
            }
            for cat, m in category_metrics.items()
        },
        'confusion_matrix': dict(confusion),
        'latency_stats': latency_stats,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        "predictions": all_predictions,
    }
    return results



def find_baseline_results(baseline_dir: Path, dataset: str, model_key: str) -> Optional[Path]:
    for name in [
        f"baseline_predictions_{dataset}_{model_key}_sp1.json",
        f"baseline_predictions_{dataset}_{model_key}_sp2.json",
    ]:
        p = baseline_dir / name
        if p.exists():
            return p
    return None


def load_baseline_results(baseline_file: Path) -> Optional[Dict]:
    if not baseline_file or not baseline_file.exists():
        return None
    with open(baseline_file, "r") as f:
        data = json.load(f)
    if "accuracy" in data:
        return data
    predictions = data.get("predictions", [])
    from benchmarking.metrics import compute_metrics
    metrics = compute_metrics(predictions)
    return {**metrics, "predictions": predictions}


def compare_results(baseline: Dict, finetuned: Dict) -> Dict:
    baseline_macro_f1 = baseline.get('macro_f1', 0)
    baseline_bal_acc = baseline.get('balanced_accuracy', baseline.get('macro_accuracy', 0))
    finetuned_macro_f1 = finetuned.get('macro_f1', 0)
    finetuned_bal_acc = finetuned.get('balanced_accuracy', 0)
    comparison = {
        'baseline': {
            'accuracy': baseline.get('accuracy', 0),
            'macro_f1': baseline_macro_f1,
            'balanced_accuracy': baseline_bal_acc,
            'false_positive_rate': baseline.get('false_positive_rate', 0),
            'false_negative_rate': baseline.get('false_negative_rate', 0),
            'latency_mean': baseline.get('latency_stats', {}).get('mean', 0),
        },
        'finetuned': {
            'accuracy': finetuned.get('accuracy', 0),
            'macro_f1': finetuned_macro_f1,
            'balanced_accuracy': finetuned_bal_acc,
            'false_positive_rate': finetuned.get('false_positive_rate', 0),
            'false_negative_rate': finetuned.get('false_negative_rate', 0),
            'latency_mean': finetuned.get('latency_stats', {}).get('mean', 0),
        },
        'improvements': {
            'accuracy_delta': finetuned.get('accuracy', 0) - baseline.get('accuracy', 0),
            'accuracy_improvement_pct': ((finetuned.get('accuracy', 0) - baseline.get('accuracy', 0)) / max(baseline.get('accuracy', 0), 0.01)) * 100 if baseline.get('accuracy', 0) > 0 else 0,
            'macro_f1_delta': finetuned_macro_f1 - baseline_macro_f1,
            'balanced_accuracy_delta': finetuned_bal_acc - baseline_bal_acc,
            'fpr_delta': finetuned.get('false_positive_rate', 0) - baseline.get('false_positive_rate', 0),
            'fnr_delta': finetuned.get('false_negative_rate', 0) - baseline.get('false_negative_rate', 0),
            'latency_delta': finetuned.get('latency_stats', {}).get('mean', 0) - baseline.get('latency_stats', {}).get('mean', 0),
        },
        'category_comparison': {}
    }
    
    # Compare per-category accuracy
    baseline_cats = baseline.get('category_accuracy', {})
    finetuned_cats = finetuned.get('category_accuracy', {})
    
    all_categories = set(baseline_cats.keys()) | set(finetuned_cats.keys())
    for cat in all_categories:
        baseline_acc = baseline_cats.get(cat, {}).get('accuracy', 0)
        finetuned_acc = finetuned_cats.get(cat, {}).get('accuracy', 0)
        
        comparison['category_comparison'][cat] = {
            'baseline': baseline_acc,
            'finetuned': finetuned_acc,
            'delta': finetuned_acc - baseline_acc,
            'improvement_pct': ((finetuned_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
        }
    
    return comparison



def main(
    dataset: str = "ami",
    model: Optional[str] = None,
    checkpoint: Optional[str] = None,
    lora_rank: Optional[int] = None,
    mode: str = TRAINING_MODE_DECISION_ONLY,
    filter_no_context: bool = True,
):
    if mode not in TRAINING_MODES:
        raise ValueError(f"mode must be one of {TRAINING_MODES}, got {mode!r}")

    model_key = model if model is not None else DEFAULT_MODEL
    if model_key not in MODEL_OPTIONS:
        model_key = next((k for k in MODEL_OPTIONS.keys() if k == model_key), DEFAULT_MODEL)

    DATA_DIR = BASE_DIR / "data" / dataset
    TEST_FILE = DATA_DIR / "test" / "test_samples.jsonl"
    EVAL_FILE = TEST_FILE

    checkpoint_name = (checkpoint or "final").strip().lower()
    base_run_name = f"{model_key}_r{lora_rank}" if lora_rank is not None else model_key
    run_dir_name = f"{base_run_name}_cot" if mode == TRAINING_MODE_COT else base_run_name
    checkpoints_dir = BASE_DIR / "fine_tuning" / "checkpoints" / run_dir_name
    if checkpoint_name == "final":
        adapter_path = checkpoints_dir / "final_model"
    else:
        adapter_path = checkpoints_dir / checkpoint_name

    RESULTS_DIR = Path(__file__).parent / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = "" if checkpoint_name == "final" else f"_{checkpoint_name}"
    if lora_rank is not None:
        suffix = f"_r{lora_rank}{suffix}"
    if mode == TRAINING_MODE_COT:
        suffix = f"{suffix}_cot"
    RESULTS_FILE = RESULTS_DIR / f"finetuned_results_{dataset}_{model_key}{suffix}.json"
    COMPARISON_FILE = RESULTS_DIR / f"baseline_vs_finetuned_{dataset}_{model_key}{suffix}.json"

    print("=" * 70)
    print(f"EVALUATING FINE-TUNED MODEL: {model_key}")
    print(f"Dataset: {dataset.upper()}")
    print(f"Mode: {mode} (checkpoint dir: {run_dir_name})")
    print(f"Checkpoint: {checkpoint_name}")
    if lora_rank is not None:
        print(f"LoRA rank run: r{lora_rank}")
    print("=" * 70)

    print("\nLoading fine-tuned model...")
    model_obj, tokenizer = load_finetuned_model(
        use_vllm=USE_VLLM,
        adapter_path=adapter_path,
        lora_rank=lora_rank,
    )

    print(f"\nLoading samples from {EVAL_FILE}...")
    samples = load_samples(EVAL_FILE)
    print(f"Loaded {len(samples)} samples")
    if filter_no_context:
        n_before = len(samples)
        samples = filter_samples_with_context(samples)
        n_removed = n_before - len(samples)
        print(f"Filtered to samples with context_turns: {len(samples)} (removed {n_removed} with no context)")
    if not samples:
        print("No samples left after filtering. Exiting.")
        return None

    results = evaluate_samples(samples, model_obj, tokenizer, mode=mode)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nTotal samples: {results['total_samples']:,}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Macro F1: {results.get('macro_f1', 0):.2%}")
    print(f"Correct: {results['correct']:,}")
    print(f"Incorrect: {results['incorrect']:,}")
    print(f"\nFalse Positive Rate (SILENT -> SPEAK): {results['false_positive_rate']:.2%}")
    print(f"False Negative Rate (SPEAK -> SILENT): {results['false_negative_rate']:.2%}")
    print(f"\nLatency Statistics:")
    for stat, value in results["latency_stats"].items():
        print(f"  {stat}: {value:.4f}s")
    print(f"\nPer-Category Accuracy (SPEAK_explicit, SPEAK_implicit, SILENT_ref, SILENT_no_ref):")
    for cat in ("SPEAK_explicit", "SPEAK_implicit", "SILENT_ref", "SILENT_no_ref"):
        if cat in results["category_accuracy"]:
            metrics = results["category_accuracy"][cat]
            print(f"  {cat}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    for cat in sorted(results["category_accuracy"].keys()):
        if cat not in ("SPEAK_explicit", "SPEAK_implicit", "SILENT_ref", "SILENT_no_ref"):
            metrics = results["category_accuracy"][cat]
            print(f"  {cat}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")

    print(f"\nSaving results to {RESULTS_FILE}...")
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved")

    # Human-readable target vs predicted for every sample (same stem as JSON, .txt)
    target_vs_pred_path = RESULTS_FILE.with_suffix(".target_vs_predicted.txt")
    with open(target_vs_pred_path, "w", encoding="utf-8") as f:
        f.write(f"Target vs predicted — {dataset} test set, model {model_key}{f' r{lora_rank}' if lora_rank is not None else ''}\n")
        f.write(f"Total: {results['total_samples']}  Correct: {results['correct']}  Accuracy: {results['accuracy']:.2%}\n\n")
        for i, p in enumerate(results["predictions"]):
            target = p.get("ground_truth", "?")
            pred = p.get("prediction", "?")
            ok = "ok" if target == pred else "wrong"
            cat = p.get("category", "")
            line = f"  sample {i}: target={target}  predicted={pred}  {ok}"
            if cat:
                line += f"  [{cat}]"
            f.write(line + "\n")
    print(f"Target vs predicted (all samples) written to {target_vs_pred_path}")

    if mode == TRAINING_MODE_COT:
        reasoning_path = RESULTS_FILE.with_name(RESULTS_FILE.stem + "_reasoning.txt")
        with open(reasoning_path, "w", encoding="utf-8") as f:
            f.write(f"Generated reasoning — {dataset} test set, model {model_key}{f' r{lora_rank}' if lora_rank else ''} (CoT)\n\n")
            for i, p in enumerate(results["predictions"]):
                sid = p.get("sample_id", f"sample_{i}")
                target = p.get("ground_truth", "?")
                pred = p.get("prediction", "?")
                reasoning = p.get("reasoning", "")
                f.write(f"--- {sid} (target={target} pred={pred}) ---\n{reasoning}\n\n")
        print(f"Generated reasoning for manual review written to {reasoning_path}")

    print("\n" + "=" * 70)
    print("COMPARING WITH BASELINE")
    print("=" * 70)
    baseline_dir = BASE_DIR / "benchmarking" / "results"
    baseline_file = find_baseline_results(baseline_dir, dataset, model_key)
    baseline = load_baseline_results(baseline_file) if baseline_file else None
    if baseline:
        comparison = compare_results(baseline, results)
        
        print(f"\nBaseline Accuracy: {comparison['baseline']['accuracy']:.2%}")
        print(f"Fine-Tuned Accuracy: {comparison['finetuned']['accuracy']:.2%}")
        print(f"Improvement: {comparison['improvements']['accuracy_delta']:+.2%} "
              f"({comparison['improvements']['accuracy_improvement_pct']:+.1f}%)")
        
        print(f"\nFalse Positive Rate:")
        print(f"  Baseline: {comparison['baseline']['false_positive_rate']:.2%}")
        print(f"  Fine-Tuned: {comparison['finetuned']['false_positive_rate']:.2%}")
        print(f"  Delta: {comparison['improvements']['fpr_delta']:+.2%}")
        
        print(f"\nFalse Negative Rate:")
        print(f"  Baseline: {comparison['baseline']['false_negative_rate']:.2%}")
        print(f"  Fine-Tuned: {comparison['finetuned']['false_negative_rate']:.2%}")
        print(f"  Delta: {comparison['improvements']['fnr_delta']:+.2%}")
        
        print(f"\nCategory Improvements:")
        for cat in sorted(comparison['category_comparison'].keys()):
            comp = comparison['category_comparison'][cat]
            print(f"  {cat}: {comp['baseline']:.2%} -> {comp['finetuned']:.2%} "
                  f"({comp['delta']:+.2%}, {comp['improvement_pct']:+.1f}%)")
        
        # Save comparison
        print(f"\nSaving comparison to {COMPARISON_FILE}...")
        with open(COMPARISON_FILE, "w") as f:
            json.dump(comparison, f, indent=2)
        print("Comparison saved")
    else:
        print(f"\nBaseline results not found in {baseline_dir} for {dataset}/{model_key}.")
        print("Run benchmarking (e.g. run_benchmark.py or evaluate_baseline.py) first.")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["ami", "friends", "spgi"],
        help="Dataset (default: from DATASET env or ami)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(MODEL_OPTIONS.keys()),
        help="Model key (default: from MODEL env or config)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint name: 'final' (default) or e.g. 'checkpoint-2000'.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate a run trained with --lora-rank N (loads from checkpoints/<model>_r<N>/ or <model>_r<N>_cot/).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="decision_only",
        choices=["decision_only", "cot"],
        help="Match training mode: decision_only (default) or cot. Uses mode-specific checkpoint dir (e.g. ..._cot).",
    )
    parser.add_argument(
        "--filter-no-context",
        action="store_true",
        default=True,
        help="Exclude samples with no context_turns from evaluation (default: True).",
    )
    parser.add_argument(
        "--no-filter-no-context",
        action="store_false",
        dest="filter_no_context",
        help="Do not filter; include samples with no context_turns.",
    )
    args = parser.parse_args()
    dataset = args.dataset or os.environ.get("DATASET", "ami")
    main(
        dataset=dataset,
        model=args.model,
        checkpoint=args.checkpoint,
        lora_rank=args.lora_rank,
        mode=args.mode,
        filter_no_context=args.filter_no_context,
    )
