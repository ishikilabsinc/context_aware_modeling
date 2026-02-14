#!/usr/bin/env python3
"""
Category-Specific Analysis

Detailed analysis of model performance by category (I1-I3, S1-S5).
"""

import json
from pathlib import Path
from typing import Dict
from collections import defaultdict, Counter


RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_FILE = RESULTS_DIR / 'finetuned_results.json'
ANALYSIS_FILE = RESULTS_DIR / 'category_analysis.txt'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.constants import CATEGORY_NAMES


def load_results(file_path: Path) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)


def analyze_category_performance(results: Dict) -> Dict:
    predictions = results.get('predictions', [])
    
    category_analysis = {}
    
    for cat in CATEGORY_NAMES.keys():
        cat_predictions = [p for p in predictions if p.get('category') == cat]
        
        if not cat_predictions:
            continue
        
        total = len(cat_predictions)
        correct = sum(1 for p in cat_predictions if p['prediction'] == p['ground_truth'])
        accuracy = correct / total if total > 0 else 0
        
        # Confusion breakdown
        confusion = Counter()
        for p in cat_predictions:
            key = f"{p['ground_truth']}->{p['prediction']}"
            confusion[key] += 1
        
        category_analysis[cat] = {
            'name': CATEGORY_NAMES[cat],
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
    lines.append("="*70)
    lines.append("CATEGORY-SPECIFIC ANALYSIS")
    lines.append("="*70)
    lines.append("")
    
    # Overall metrics
    lines.append("OVERALL PERFORMANCE")
    lines.append("-"*70)
    lines.append(f"Total samples: {results['total_samples']:,}")
    lines.append(f"Accuracy: {results['accuracy']:.2%}")
    lines.append(f"Correct: {results['correct']:,}")
    lines.append(f"Incorrect: {results['incorrect']:,}")
    lines.append("")
    
    # Category performance
    cat_analysis = analyze_category_performance(results)
    
    lines.append("PER-CATEGORY PERFORMANCE")
    lines.append("-"*70)
    
    # SPEAK categories
    lines.append("\nSPEAK Categories:")
    for cat in ['I1', 'I2', 'I3']:
        if cat in cat_analysis:
            analysis = cat_analysis[cat]
            lines.append(f"\n  {cat} - {analysis['name']}:")
            lines.append(f"    Accuracy: {analysis['accuracy']:.2%}")
            lines.append(f"    Correct: {analysis['correct']}/{analysis['total']}")
            lines.append(f"    Errors: {analysis['errors']}")
            if analysis['confusion']:
                lines.append(f"    Confusion:")
                for pattern, count in analysis['confusion'].items():
                    lines.append(f"      {pattern}: {count}")
    
    # SILENT categories
    lines.append("\nSILENT Categories:")
    for cat in ['S1', 'S2', 'S3', 'S4', 'S5']:
        if cat in cat_analysis:
            analysis = cat_analysis[cat]
            lines.append(f"\n  {cat} - {analysis['name']}:")
            lines.append(f"    Accuracy: {analysis['accuracy']:.2%}")
            lines.append(f"    Correct: {analysis['correct']}/{analysis['total']}")
            lines.append(f"    Errors: {analysis['errors']}")
            if analysis['confusion']:
                lines.append(f"    Confusion:")
                for pattern, count in analysis['confusion'].items():
                    lines.append(f"      {pattern}: {count}")
    
    lines.append("")
    
    # Error analysis
    error_analysis = analyze_error_patterns(results)
    lines.append("ERROR ANALYSIS")
    lines.append("-"*70)
    lines.append(f"\nTotal errors: {error_analysis['total_errors']:,}")
    lines.append(f"Error rate: {error_analysis['error_rate']:.2%}")
    
    lines.append("\nErrors by category:")
    for cat, count in sorted(error_analysis['by_category'].items(), key=lambda x: x[1], reverse=True):
        cat_name = CATEGORY_NAMES.get(cat, cat)
        lines.append(f"  {cat} ({cat_name}): {count}")
    
    lines.append("\nError rate by decision type:")
    for decision, metrics in error_analysis['by_decision_type'].items():
        error_rate = metrics['errors'] / metrics['total'] if metrics['total'] > 0 else 0
        lines.append(f"  {decision}: {error_rate:.2%} ({metrics['errors']}/{metrics['total']})")
    
    lines.append("\nMost common confusion patterns:")
    for pattern, count in error_analysis['confusion_patterns'].most_common(10):
        lines.append(f"  {pattern}: {count}")
    
    lines.append("")
    lines.append("="*70)
    
    return "\n".join(lines)



def main():
    print("="*70)
    print("CATEGORY-SPECIFIC ANALYSIS")
    print("="*70)
    
    if not RESULTS_FILE.exists():
        print(f"\nError: Results file not found: {RESULTS_FILE}")
        print("Please run evaluate_finetuned.py first.")
        return
    
    print(f"\nLoading results from {RESULTS_FILE}...")
    results = load_results(RESULTS_FILE)
    
    print("Generating category analysis...")
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
