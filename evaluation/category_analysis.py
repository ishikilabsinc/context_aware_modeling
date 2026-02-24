#!/usr/bin/env python3
"""Per-category performance analysis of fine-tuned evaluation results."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict
from collections import Counter, defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from utils.constants import CATEGORY_NAMES

RESULTS_DIR = Path(__file__).parent / "results"


def load_results(file_path: Path) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)


def analyze_category_performance(results: Dict) -> Dict:
    predictions = results.get('predictions', [])
    category_analysis = {}
    categories = sorted(set(p.get('category') for p in predictions if p.get('category')))

    for cat in categories:
        cat_predictions = [p for p in predictions if p.get('category') == cat]
        if not cat_predictions:
            continue

        total = len(cat_predictions)
        correct = sum(1 for p in cat_predictions if p['prediction'] == p['ground_truth'])
        accuracy = correct / total if total > 0 else 0

        confusion = Counter()
        for p in cat_predictions:
            key = f"{p['ground_truth']}->{p['prediction']}"
            confusion[key] += 1

        category_analysis[cat] = {
            'name': CATEGORY_NAMES.get(cat, cat),
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'confusion': dict(confusion),
            'errors': total - correct
        }

    return category_analysis


def analyze_error_patterns(results: Dict) -> Dict:
    predictions = results.get('predictions', [])
    errors = [p for p in predictions if p['prediction'] != p['ground_truth']]
    
    error_analysis = {
        'total_errors': len(errors),
        'error_rate': len(errors) / len(predictions) if predictions else 0,
        'by_category': defaultdict(int),
        'by_decision_type': defaultdict(lambda: {'total': 0, 'errors': 0}),
        'confusion_patterns': Counter(),
    }
    
    for error in errors:
        cat = error.get('category', 'UNKNOWN')
        error_analysis['by_category'][cat] += 1
        
        gt = error['ground_truth']
        pred = error['prediction']
        error_analysis['by_decision_type'][gt]['total'] += 1
        error_analysis['by_decision_type'][gt]['errors'] += 1
        
        pattern = f"{gt}->{pred}"
        error_analysis['confusion_patterns'][pattern] += 1
    
    return error_analysis


def generate_report(results: Dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("CATEGORY-SPECIFIC ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("OVERALL PERFORMANCE")
    lines.append("-" * 70)
    lines.append(f"Total samples: {results['total_samples']:,}")
    lines.append(f"Accuracy: {results['accuracy']:.2%}")
    lines.append(f"Correct: {results['correct']:,}")
    lines.append(f"Incorrect: {results['incorrect']:,}")
    lines.append("")

    cat_analysis = analyze_category_performance(results)
    lines.append("PER-CATEGORY PERFORMANCE")
    lines.append("-" * 70)
    for cat in sorted(cat_analysis.keys()):
        a = cat_analysis[cat]
        lines.append(f"\n  {cat} - {a['name']}:")
        lines.append(f"    Accuracy: {a['accuracy']:.2%} Correct: {a['correct']}/{a['total']} Errors: {a['errors']}")
        if a["confusion"]:
            for pattern, count in a["confusion"].items():
                lines.append(f"      {pattern}: {count}")
    lines.append("")
    error_analysis = analyze_error_patterns(results)
    lines.append("ERROR ANALYSIS")
    lines.append("-" * 70)
    lines.append(f"\nTotal errors: {error_analysis['total_errors']:,} Rate: {error_analysis['error_rate']:.2%}")
    lines.append("\nErrors by category:")
    for cat, count in sorted(error_analysis["by_category"].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {cat} ({CATEGORY_NAMES.get(cat, cat)}): {count}")
    lines.append("\nError rate by decision type:")
    for decision, metrics in error_analysis["by_decision_type"].items():
        rate = metrics["errors"] / metrics["total"] if metrics["total"] > 0 else 0
        lines.append(f"  {decision}: {rate:.2%} ({metrics['errors']}/{metrics['total']})")
    lines.append("\nMost common confusion patterns:")
    for pattern, count in error_analysis["confusion_patterns"].most_common(10):
        lines.append(f"  {pattern}: {count}")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)



def main(dataset: str = "ami", model: str = None, suffix: str = ""):
    from fine_tuning.config import MODEL_OPTIONS, MODEL as DEFAULT_MODEL
    model_key = model or DEFAULT_MODEL
    if model_key not in MODEL_OPTIONS:
        model_key = DEFAULT_MODEL
    results_file = RESULTS_DIR / f"finetuned_results_{dataset}_{model_key}{suffix}.json"
    analysis_file = RESULTS_DIR / f"category_analysis_{dataset}_{model_key}{suffix}.txt"

    if not results_file.exists():
        print(f"Results file not found: {results_file}. Run evaluate_finetuned.py first.")
        return
    results = load_results(results_file)
    report = generate_report(results)
    print("\n" + report)
    with open(analysis_file, "w") as f:
        f.write(report)
    print(f"Analysis saved to {analysis_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Category analysis of fine-tuned results")
    parser.add_argument("--dataset", type=str, default="ami", choices=["ami", "friends", "spgi"])
    parser.add_argument("--model", type=str, default=None, help="Model key (default: from MODEL env)")
    parser.add_argument("--lora-rank", type=int, default=None, metavar="N", help="Use results from run with --lora-rank N (suffix _rN)")
    parser.add_argument("--suffix", type=str, default="", help="Filename suffix (e.g. _checkpoint-2000). Overridden by --lora-rank.")
    args = parser.parse_args()
    suffix = args.suffix
    if args.lora_rank is not None:
        suffix = f"_r{args.lora_rank}"
    main(dataset=args.dataset, model=args.model, suffix=suffix)
