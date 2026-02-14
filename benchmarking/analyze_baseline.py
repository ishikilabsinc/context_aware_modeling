#!/usr/bin/env python3
"""
Analyze Baseline Results

Generates detailed analysis reports from baseline evaluation results.
"""

import json
from pathlib import Path
from typing import Dict
from collections import Counter, defaultdict


RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_FILE = RESULTS_DIR / 'baseline_results.json'
ANALYSIS_FILE = RESULTS_DIR / 'baseline_analysis.txt'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.constants import SPEAK_CATEGORIES, SILENT_CATEGORIES, CATEGORY_NAMES


def load_results(file_path: Path) -> Dict:
    """Load evaluation results."""
    with open(file_path, 'r') as f:
        return json.load(f)


def analyze_errors(results: Dict) -> Dict:
    """Analyze error patterns."""
    predictions = results.get('predictions', [])
    
    errors = {
        'by_category': defaultdict(lambda: {'total': 0, 'errors': 0}),
        'by_decision': defaultdict(lambda: {'total': 0, 'errors': 0}),
        'confusion_patterns': Counter(),
    }
    
    for pred in predictions:
        cat = pred.get('category', 'UNKNOWN')
        gt = pred.get('ground_truth', 'UNKNOWN')
        pred_val = pred.get('prediction', 'UNKNOWN')
        
        # Category errors
        errors['by_category'][cat]['total'] += 1
        if gt != pred_val:
            errors['by_category'][cat]['errors'] += 1
        
        # Decision errors
        errors['by_decision'][gt]['total'] += 1
        if gt != pred_val:
            errors['by_decision'][gt]['errors'] += 1
            errors['confusion_patterns'][f"{gt} -> {pred_val}"] += 1
    
    return errors


def generate_report(results: Dict) -> str:
    """Generate detailed analysis report."""
    lines = []
    lines.append("="*70)
    lines.append("BASELINE EVALUATION ANALYSIS")
    lines.append("="*70)
    lines.append("")
    
    # Overall metrics
    lines.append("OVERALL METRICS")
    lines.append("-"*70)
    lines.append(f"Total samples: {results['total_samples']:,}")
    lines.append(f"Accuracy: {results['accuracy']:.2%}")
    lines.append(f"Correct: {results['correct']:,}")
    lines.append(f"Incorrect: {results['incorrect']:,}")
    lines.append("")
    
    # Error rates
    lines.append("ERROR RATES")
    lines.append("-"*70)
    lines.append(f"False Positive Rate (SILENT -> SPEAK): {results['false_positive_rate']:.2%}")
    lines.append(f"False Negative Rate (SPEAK -> SILENT): {results['false_negative_rate']:.2%}")
    lines.append("")
    
    # Latency
    lines.append("LATENCY STATISTICS")
    lines.append("-"*70)
    latency = results['latency_stats']
    lines.append(f"Mean: {latency['mean']:.4f}s")
    lines.append(f"Median: {latency['median']:.4f}s")
    lines.append(f"P50: {latency['p50']:.4f}s")
    lines.append(f"P95: {latency['p95']:.4f}s")
    lines.append(f"P99: {latency['p99']:.4f}s")
    lines.append(f"Min: {latency['min']:.4f}s")
    lines.append(f"Max: {latency['max']:.4f}s")
    lines.append("")
    
    # Per-category accuracy
    lines.append("PER-CATEGORY ACCURACY")
    lines.append("-"*70)
    
    # SPEAK categories
    lines.append("\nSPEAK Categories:")
    for cat in SPEAK_CATEGORIES:
        if cat in results['category_accuracy']:
            metrics = results['category_accuracy'][cat]
            cat_name = CATEGORY_NAMES.get(cat, cat)
            lines.append(f"  {cat} ({cat_name}):")
            lines.append(f"    Accuracy: {metrics['accuracy']:.2%}")
            lines.append(f"    Correct: {metrics['correct']}/{metrics['total']}")
    
    # SILENT categories
    lines.append("\nSILENT Categories:")
    for cat in SILENT_CATEGORIES:
        if cat in results['category_accuracy']:
            metrics = results['category_accuracy'][cat]
            cat_name = CATEGORY_NAMES.get(cat, cat)
            lines.append(f"  {cat} ({cat_name}):")
            lines.append(f"    Accuracy: {metrics['accuracy']:.2%}")
            lines.append(f"    Correct: {metrics['correct']}/{metrics['total']}")
    
    lines.append("")
    
    # Confusion matrix
    lines.append("CONFUSION MATRIX")
    lines.append("-"*70)
    confusion = results.get('confusion_matrix', {})
    for pattern, count in sorted(confusion.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {pattern}: {count}")
    lines.append("")
    
    # Error analysis
    errors = analyze_errors(results)
    lines.append("ERROR ANALYSIS")
    lines.append("-"*70)
    
    lines.append("\nError Rate by Category:")
    for cat in sorted(errors['by_category'].keys()):
        metrics = errors['by_category'][cat]
        error_rate = metrics['errors'] / metrics['total'] if metrics['total'] > 0 else 0
        cat_name = CATEGORY_NAMES.get(cat, cat)
        lines.append(f"  {cat} ({cat_name}): {error_rate:.2%} ({metrics['errors']}/{metrics['total']})")
    
    lines.append("\nError Rate by Ground Truth Decision:")
    for decision in sorted(errors['by_decision'].keys()):
        metrics = errors['by_decision'][decision]
        error_rate = metrics['errors'] / metrics['total'] if metrics['total'] > 0 else 0
        lines.append(f"  {decision}: {error_rate:.2%} ({metrics['errors']}/{metrics['total']})")
    
    lines.append("\nMost Common Confusion Patterns:")
    for pattern, count in errors['confusion_patterns'].most_common(10):
        lines.append(f"  {pattern}: {count}")
    
    lines.append("")
    lines.append("="*70)
    
    return "\n".join(lines)



def main():
    """Main analysis function."""
    print("="*70)
    print("ANALYZING BASELINE RESULTS")
    print("="*70)
    
    if not RESULTS_FILE.exists():
        print(f"\nError: Results file not found: {RESULTS_FILE}")
        print("Please run evaluate_baseline.py first.")
        return
    
    print(f"\nLoading results from {RESULTS_FILE}...")
    results = load_results(RESULTS_FILE)
    
    print("Generating analysis report...")
    report = generate_report(results)
    
    # Print to console
    print("\n" + report)
    
    # Save to file
    print(f"\nSaving analysis to {ANALYSIS_FILE}...")
    with open(ANALYSIS_FILE, 'w') as f:
        f.write(report)
    print("Analysis saved")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
