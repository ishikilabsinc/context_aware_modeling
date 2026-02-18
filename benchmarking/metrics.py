"""
Compute evaluation metrics from prediction records and generate the baseline
analysis text report. Used by evaluate_baseline (single run) and run_benchmark (batch).
"""

from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from utils.constants import CATEGORY_NAMES, SILENT_CATEGORIES, SPEAK_CATEGORIES



def compute_metrics(predictions: List[Dict]) -> Dict:
    """
    Compute aggregate metrics from per-sample predictions.
    Each prediction dict must have: ground_truth, prediction, category, latency.
    SPEAK = positive class for precision/recall/F1.

    Returns a dict with: total_samples, accuracy, macro_accuracy, correct, incorrect,
    category_accuracy, confusion_matrix, latency_stats, false_positive_rate,
    false_negative_rate, precision_speak, recall_speak, precision_silent, recall_silent,
    f1_speak, f1_silent, macro_f1. Does not include predictions.
    """
    empty_latency = {
        "mean": 0.0, "median": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0,
    }
    if not predictions:
        return {
            "total_samples": 0,
            "accuracy": 0.0,
            "macro_accuracy": 0.0,
            "correct": 0,
            "incorrect": 0,
            "category_accuracy": {},
            "confusion_matrix": {},
            "latency_stats": empty_latency,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "precision_speak": 0.0,
            "recall_speak": 0.0,
            "precision_silent": 0.0,
            "recall_silent": 0.0,
            "f1_speak": 0.0,
            "f1_silent": 0.0,
            "macro_f1": 0.0,
        }

    correct = sum(1 for p in predictions if p.get("prediction") == p.get("ground_truth"))
    total = len(predictions)
    accuracy = correct / total if total else 0.0

    category_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
    for p in predictions:
        cat = p.get("category", "UNKNOWN")
        category_metrics[cat]["total"] += 1
        if p.get("prediction") == p.get("ground_truth"):
            category_metrics[cat]["correct"] += 1

    confusion = defaultdict(int)
    for p in predictions:
        gt = p.get("ground_truth", "UNKNOWN")
        pred = p.get("prediction", "UNKNOWN")
        key = f"{gt}_->_{pred}"
        confusion[key] += 1

    latencies = [p.get("latency", 0.0) for p in predictions if p.get("latency") is not None]
    if latencies:
        arr = np.array(latencies)
        latency_stats = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    else:
        latency_stats = {
            "mean": 0.0,
            "median": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    total_silent = sum(1 for p in predictions if p.get("ground_truth") == "SILENT")
    total_speak = sum(1 for p in predictions if p.get("ground_truth") == "SPEAK")
    true_positives = sum(
        1 for p in predictions
        if p.get("ground_truth") == "SPEAK" and p.get("prediction") == "SPEAK"
    )
    false_positives = sum(
        1 for p in predictions
        if p.get("ground_truth") == "SILENT" and p.get("prediction") == "SPEAK"
    )
    false_negatives = sum(
        1 for p in predictions
        if p.get("ground_truth") == "SPEAK" and p.get("prediction") == "SILENT"
    )
    true_negatives = sum(
        1 for p in predictions
        if p.get("ground_truth") == "SILENT" and p.get("prediction") == "SILENT"
    )
    fpr = false_positives / total_silent if total_silent > 0 else 0.0
    fnr = false_negatives / total_speak if total_speak > 0 else 0.0

    precision_speak = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall_speak = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    precision_silent = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0.0
    recall_silent = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0

    def _f1(p: float, r: float) -> float:
        if p + r <= 0:
            return 0.0
        return 2.0 * p * r / (p + r)

    f1_speak = _f1(precision_speak, recall_speak)
    f1_silent = _f1(precision_silent, recall_silent)
    macro_f1 = (f1_speak + f1_silent) / 2.0

    acc_silent = 1.0 - fpr if total_silent > 0 else 0.0
    acc_speak = 1.0 - fnr if total_speak > 0 else 0.0
    macro_accuracy = (acc_silent + acc_speak) / 2.0

    category_accuracy = {
        cat: {
            "accuracy": m["correct"] / m["total"] if m["total"] > 0 else 0.0,
            "correct": m["correct"],
            "total": m["total"],
        }
        for cat, m in category_metrics.items()
    }

    return {
        "total_samples": total,
        "accuracy": accuracy,
        "macro_accuracy": macro_accuracy,
        "correct": correct,
        "incorrect": total - correct,
        "category_accuracy": category_accuracy,
        "confusion_matrix": dict(confusion),
        "latency_stats": latency_stats,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "precision_speak": precision_speak,
        "recall_speak": recall_speak,
        "precision_silent": precision_silent,
        "recall_silent": recall_silent,
        "f1_speak": f1_speak,
        "f1_silent": f1_silent,
        "macro_f1": macro_f1,
    }


def _analyze_errors(results: Dict) -> Dict:
    """Error rates by category and decision, and confusion patterns."""
    predictions = results.get("predictions", [])
    errors = {
        "by_category": defaultdict(lambda: {"total": 0, "errors": 0}),
        "by_decision": defaultdict(lambda: {"total": 0, "errors": 0}),
        "confusion_patterns": Counter(),
    }
    for pred in predictions:
        cat = pred.get("category", "UNKNOWN")
        gt = pred.get("ground_truth", "UNKNOWN")
        pred_val = pred.get("prediction", "UNKNOWN")
        errors["by_category"][cat]["total"] += 1
        if gt != pred_val:
            errors["by_category"][cat]["errors"] += 1
        errors["by_decision"][gt]["total"] += 1
        if gt != pred_val:
            errors["by_decision"][gt]["errors"] += 1
            errors["confusion_patterns"][f"{gt} -> {pred_val}"] += 1
    return errors


def generate_detail_report(results: Dict) -> str:
    """
    Baseline evaluation analysis as text. results must contain metrics keys
    (total_samples, accuracy, ...) and "predictions" for null count and error analysis.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("BASELINE EVALUATION ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("OVERALL METRICS")
    lines.append("-" * 70)
    lines.append(f"Total samples: {results['total_samples']:,}")
    lines.append(f"Accuracy: {results['accuracy']:.2%}")
    lines.append(f"Macro accuracy: {results.get('macro_accuracy', 0):.2%}")
    lines.append(f"Correct: {results['correct']:,}")
    lines.append(f"Incorrect: {results['incorrect']:,}")
    preds = results.get("predictions", [])
    nulls = sum(1 for p in preds if p.get("prediction") is None)
    if preds:
        lines.append(f"Null predictions: {nulls:,} ({nulls / len(preds):.1%})")
    lines.append("")
    lines.append("PRECISION / RECALL / F1 (SPEAK = positive)")
    lines.append("-" * 70)
    lines.append(f"Speak  Precision: {results.get('precision_speak', 0):.2%}  Recall: {results.get('recall_speak', 0):.2%}  F1: {results.get('f1_speak', 0):.2%}")
    lines.append(f"Silent Precision: {results.get('precision_silent', 0):.2%}  Recall: {results.get('recall_silent', 0):.2%}  F1: {results.get('f1_silent', 0):.2%}")
    lines.append(f"Macro F1: {results.get('macro_f1', 0):.2%}")
    lines.append("")
    lines.append("ERROR RATES")
    lines.append("-" * 70)
    lines.append(f"False Positive Rate (SILENT -> SPEAK): {results['false_positive_rate']:.2%}")
    lines.append(f"False Negative Rate (SPEAK -> SILENT): {results['false_negative_rate']:.2%}")
    lines.append("")
    lines.append("LATENCY STATISTICS")
    lines.append("-" * 70)
    latency = results["latency_stats"]
    lines.append(f"Mean: {latency['mean']:.4f}s")
    lines.append(f"Median: {latency['median']:.4f}s")
    lines.append(f"P50: {latency['p50']:.4f}s")
    lines.append(f"P95: {latency['p95']:.4f}s")
    lines.append(f"P99: {latency['p99']:.4f}s")
    lines.append(f"Min: {latency['min']:.4f}s")
    lines.append(f"Max: {latency['max']:.4f}s")
    lines.append("")
    lines.append("PER-CATEGORY ACCURACY")
    lines.append("-" * 70)
    lines.append("\nSPEAK Categories:")
    for cat in SPEAK_CATEGORIES:
        if cat in results.get("category_accuracy", {}):
            m = results["category_accuracy"][cat]
            lines.append(f"  {cat} ({CATEGORY_NAMES.get(cat, cat)}):")
            lines.append(f"    Accuracy: {m['accuracy']:.2%}")
            lines.append(f"    Correct: {m['correct']}/{m['total']}")
    lines.append("\nSILENT Categories:")
    for cat in SILENT_CATEGORIES:
        if cat in results.get("category_accuracy", {}):
            m = results["category_accuracy"][cat]
            lines.append(f"  {cat} ({CATEGORY_NAMES.get(cat, cat)}):")
            lines.append(f"    Accuracy: {m['accuracy']:.2%}")
            lines.append(f"    Correct: {m['correct']}/{m['total']}")
    lines.append("")
    lines.append("CONFUSION MATRIX")
    lines.append("-" * 70)
    for pattern, count in sorted(results.get("confusion_matrix", {}).items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {pattern}: {count}")
    lines.append("")
    errors = _analyze_errors(results)
    lines.append("ERROR ANALYSIS")
    lines.append("-" * 70)
    lines.append("\nError Rate by Category:")
    for cat in sorted(errors["by_category"].keys()):
        m = errors["by_category"][cat]
        rate = m["errors"] / m["total"] if m["total"] > 0 else 0
        lines.append(f"  {cat} ({CATEGORY_NAMES.get(cat, cat)}): {rate:.2%} ({m['errors']}/{m['total']})")
    lines.append("\nError Rate by Ground Truth Decision:")
    for decision in sorted(errors["by_decision"].keys()):
        m = errors["by_decision"][decision]
        rate = m["errors"] / m["total"] if m["total"] > 0 else 0
        lines.append(f"  {decision}: {rate:.2%} ({m['errors']}/{m['total']})")
    lines.append("\nMost Common Confusion Patterns:")
    for pattern, count in errors["confusion_patterns"].most_common(10):
        lines.append(f"  {pattern}: {count}")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)
