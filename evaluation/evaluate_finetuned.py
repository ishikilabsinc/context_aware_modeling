#!/usr/bin/env python3
"""
Evaluate fine-tuned model on test set and compare with baseline. Uses config for model/dataset.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "benchmarking"))
sys.path.insert(0, str(BASE_DIR))
from fine_tuning.config import MODEL_OPTIONS, MODEL as DEFAULT_MODEL
from load_finetuned import load_finetuned_model
from evaluate_baseline import (
    format_sample_for_inference,
    extract_decision_from_output,
)
from utils.data_utils import load_samples

USE_VLLM = True
BATCH_SIZE = 32 if USE_VLLM else 4
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.0
STOP_AFTER_DECISION = True



def infer_with_vllm(model, tokenizer, prompts: List[str]) -> List[Tuple[str, float]]:
    """Run inference using vLLM with batching."""
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
        stop=None,
    )
    
    lora_adapter = getattr(model, '_lora_adapter_name', None)
    
    start_time = time.time()
    if lora_adapter:
        try:
            from vllm.lora.request import LoRARequest
            lora_requests = [LoRARequest(lora_adapter, 1, 0)] * len(prompts)
            outputs = model.generate(prompts, sampling_params, lora_request=lora_requests)
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
        
        if STOP_AFTER_DECISION:
            decision_match = re.search(r'<decision>.*?</decision>', generated_text, re.DOTALL | re.IGNORECASE)
            if decision_match:
                generated_text = generated_text[:decision_match.end()]
        
        results.append((generated_text, latency))
    
    return results


def infer_with_model_batch(model, tokenizer, prompts: List[str]) -> List[Tuple[str, float]]:
    """Process a batch of prompts using transformers."""
    import torch
    
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    tokenizer.padding_side = original_padding_side
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
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
            decision_match = re.search(r'<decision>.*?</decision>', generated_text, re.DOTALL | re.IGNORECASE)
            if decision_match:
                generated_text = generated_text[:decision_match.end()]
        
        results.append((generated_text, avg_latency))
    
    return results



def evaluate_samples(
    samples: List[Dict],
    model,
    tokenizer
) -> Dict:
    print(f"\nEvaluating {len(samples)} samples...")
    
    is_vllm_model = hasattr(model, '_llm_engine') or (hasattr(model, 'llm_engine') and model.llm_engine is not None)
    effective_batch_size = BATCH_SIZE if is_vllm_model else min(BATCH_SIZE, 4)
    print(f"Batch size: {effective_batch_size} ({'vLLM' if is_vllm_model else 'transformers'})")
    
    prompts = [format_sample_for_inference(sample) for sample in samples]
    all_predictions = []
    all_latencies = []

    for i in range(0, len(prompts), effective_batch_size):
        batch_prompts = prompts[i : i + effective_batch_size]
        batch_samples = samples[i : i + effective_batch_size]
        batch_num = i // effective_batch_size + 1
        total_batches = (len(prompts) + effective_batch_size - 1) // effective_batch_size
        if batch_num % 10 == 0 or batch_num == 1:
            print(f"  Processing batch {batch_num}/{total_batches} "
                  f"({len(batch_prompts)} samples, {i+1}/{len(samples)} total)...", end=' ', flush=True)
        
        if USE_VLLM and is_vllm_model:
            batch_results = infer_with_vllm(model, tokenizer, batch_prompts)
        else:
            batch_results = infer_with_model_batch(model, tokenizer, batch_prompts)
        
        if batch_num % 50 == 0:
            import torch
            torch.cuda.empty_cache()
        
        for (output_text, latency), sample in zip(batch_results, batch_samples):
            prediction = extract_decision_from_output(output_text)
            ground_truth = sample.get('decision', 'UNKNOWN')
            
            all_predictions.append({
                'sample_id': sample.get('decision_point_id', f'sample_{i}'),
                'ground_truth': ground_truth,
                'prediction': prediction,
                'category': sample.get('category', 'UNKNOWN'),
                'output_text': output_text[:200],
                'latency': latency
            })
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
    
    total_silent = sum(1 for p in all_predictions if p['ground_truth'] == 'SILENT')
    total_speak = sum(1 for p in all_predictions if p['ground_truth'] == 'SPEAK')
    
    fpr = false_positives / total_silent if total_silent > 0 else 0
    fnr = false_negatives / total_speak if total_speak > 0 else 0
    
    results = {
        'total_samples': len(all_predictions),
        'accuracy': accuracy,
        'correct': correct,
        'incorrect': len(all_predictions) - correct,
        'category_accuracy': {
            cat: {
                'accuracy': metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0,
                'correct': metrics['correct'],
                'total': metrics['total']
            }
            for cat, metrics in category_metrics.items()
        },
        'confusion_matrix': dict(confusion),
        'latency_stats': latency_stats,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        "predictions": all_predictions[:100],
    }
    return results



def find_baseline_results(baseline_dir: Path, dataset: str, model_key: str) -> Optional[Path]:
    for name in [
        f"baseline_results_{dataset}_{model_key}.json",
        f"baseline_results_{dataset}_{model_key}_sp1.json",
        f"baseline_results_{dataset}_{model_key}_sp2.json",
    ]:
        p = baseline_dir / name
        if p.exists():
            return p
    return None


def load_baseline_results(baseline_file: Path) -> Optional[Dict]:
    if not baseline_file.exists():
        return None
    with open(baseline_file, "r") as f:
        return json.load(f)


def compare_results(baseline: Dict, finetuned: Dict) -> Dict:
    comparison = {
        'baseline': {
            'accuracy': baseline.get('accuracy', 0),
            'false_positive_rate': baseline.get('false_positive_rate', 0),
            'false_negative_rate': baseline.get('false_negative_rate', 0),
            'latency_mean': baseline.get('latency_stats', {}).get('mean', 0),
        },
        'finetuned': {
            'accuracy': finetuned.get('accuracy', 0),
            'false_positive_rate': finetuned.get('false_positive_rate', 0),
            'false_negative_rate': finetuned.get('false_negative_rate', 0),
            'latency_mean': finetuned.get('latency_stats', {}).get('mean', 0),
        },
        'improvements': {
            'accuracy_delta': finetuned.get('accuracy', 0) - baseline.get('accuracy', 0),
            'accuracy_improvement_pct': ((finetuned.get('accuracy', 0) - baseline.get('accuracy', 0)) / max(baseline.get('accuracy', 0), 0.01)) * 100 if baseline.get('accuracy', 0) > 0 else 0,
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



def main(dataset: str = "ami", model: Optional[str] = None):
    model_key = model if model is not None else DEFAULT_MODEL
    if model_key not in MODEL_OPTIONS:
        model_key = next((k for k in MODEL_OPTIONS.keys() if k == model_key), DEFAULT_MODEL)

    DATA_DIR = BASE_DIR / "data" / dataset
    TEST_FILE = DATA_DIR / "test" / "test_samples.jsonl"
    EVAL_FILE = TEST_FILE

    RESULTS_DIR = Path(__file__).parent / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    RESULTS_FILE = RESULTS_DIR / f"finetuned_results_{dataset}_{model_key}.json"
    COMPARISON_FILE = RESULTS_DIR / f"baseline_vs_finetuned_{dataset}_{model_key}.json"

    print("=" * 70)
    print(f"EVALUATING FINE-TUNED MODEL: {model_key}")
    print(f"Dataset: {dataset.upper()}")
    print("=" * 70)

    print("\nLoading fine-tuned model...")
    model_obj, tokenizer = load_finetuned_model(use_vllm=USE_VLLM)

    print(f"\nLoading samples from {EVAL_FILE}...")
    samples = load_samples(EVAL_FILE)
    print(f"Loaded {len(samples)} samples")

    results = evaluate_samples(samples, model_obj, tokenizer)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nTotal samples: {results['total_samples']:,}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Correct: {results['correct']:,}")
    print(f"Incorrect: {results['incorrect']:,}")
    print(f"\nFalse Positive Rate (SILENT -> SPEAK): {results['false_positive_rate']:.2%}")
    print(f"False Negative Rate (SPEAK -> SILENT): {results['false_negative_rate']:.2%}")
    print(f"\nLatency Statistics:")
    for stat, value in results["latency_stats"].items():
        print(f"  {stat}: {value:.4f}s")
    print(f"\nPer-Category Accuracy:")
    for cat in sorted(results["category_accuracy"].keys()):
        metrics = results["category_accuracy"][cat]
        print(f"  {cat}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")

    print(f"\nSaving results to {RESULTS_FILE}...")
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved")

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
    args = parser.parse_args()
    dataset = args.dataset or os.environ.get("DATASET", "ami")
    main(dataset=dataset, model=args.model)
