#!/usr/bin/env python3
"""
Generate Final Evaluation Report

Compiles all evaluation results into a comprehensive report.
"""

import json
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).parent / 'results'

BASELINE_FILE = BASE_DIR / 'benchmarking' / 'results' / 'baseline_results.json'
FINETUNED_FILE = RESULTS_DIR / 'finetuned_results.json'
COMPARISON_FILE = RESULTS_DIR / 'baseline_vs_finetuned.json'
CATEGORY_ANALYSIS_FILE = RESULTS_DIR / 'category_analysis.txt'

REPORT_FILE = RESULTS_DIR / 'final_evaluation_report.md'


def load_json_file(file_path: Path) -> dict:
    if not file_path.exists():
        return None
    with open(file_path, 'r') as f:
        return json.load(f)


def load_text_file(file_path: Path) -> str:
    if not file_path.exists():
        return None
    with open(file_path, 'r') as f:
        return f.read()


def generate_markdown_report() -> str:
    lines = []
    
    # Header
    lines.append("# Context-Aware Turn-Taking Model: Final Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Load data
    baseline = load_json_file(BASELINE_FILE)
    finetuned = load_json_file(FINETUNED_FILE)
    comparison = load_json_file(COMPARISON_FILE)
    category_analysis = load_text_file(CATEGORY_ANALYSIS_FILE)
    
    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    if baseline and finetuned:
        baseline_acc = baseline.get('accuracy', 0)
        finetuned_acc = finetuned.get('accuracy', 0)
        improvement = finetuned_acc - baseline_acc
        improvement_pct = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0
        
        lines.append(f"- **Baseline Accuracy**: {baseline_acc:.2%}")
        lines.append(f"- **Fine-Tuned Accuracy**: {finetuned_acc:.2%}")
        lines.append(f"- **Improvement**: {improvement:+.2%} ({improvement_pct:+.1f}%)")
        lines.append("")
    else:
        lines.append("Baseline or fine-tuned results not available.")
        lines.append("")
    
    # Baseline Results
    if baseline:
        lines.append("## Baseline Results")
        lines.append("")
        lines.append(f"- **Total Samples**: {baseline.get('total_samples', 0):,}")
        lines.append(f"- **Accuracy**: {baseline.get('accuracy', 0):.2%}")
        lines.append(f"- **False Positive Rate**: {baseline.get('false_positive_rate', 0):.2%}")
        lines.append(f"- **False Negative Rate**: {baseline.get('false_negative_rate', 0):.2%}")
        lines.append("")
        if 'latency_stats' in baseline:
            latency = baseline['latency_stats']
            lines.append("### Latency")
            lines.append(f"- Mean: {latency.get('mean', 0):.4f}s")
            lines.append(f"- P95: {latency.get('p95', 0):.4f}s")
            lines.append(f"- P99: {latency.get('p99', 0):.4f}s")
            lines.append("")
    
    # Fine-Tuned Results
    if finetuned:
        lines.append("## Fine-Tuned Results")
        lines.append("")
        lines.append(f"- **Total Samples**: {finetuned.get('total_samples', 0):,}")
        lines.append(f"- **Accuracy**: {finetuned.get('accuracy', 0):.2%}")
        lines.append(f"- **False Positive Rate**: {finetuned.get('false_positive_rate', 0):.2%}")
        lines.append(f"- **False Negative Rate**: {finetuned.get('false_negative_rate', 0):.2%}")
        lines.append("")
        if 'latency_stats' in finetuned:
            latency = finetuned['latency_stats']
            lines.append("### Latency")
            lines.append(f"- Mean: {latency.get('mean', 0):.4f}s")
            lines.append(f"- P95: {latency.get('p95', 0):.4f}s")
            lines.append(f"- P99: {latency.get('p99', 0):.4f}s")
            lines.append("")
    
    # Comparison
    if comparison:
        lines.append("## Comparison: Baseline vs Fine-Tuned")
        lines.append("")
        lines.append("### Overall Metrics")
        lines.append("")
        lines.append("| Metric | Baseline | Fine-Tuned | Delta |")
        lines.append("|--------|----------|------------|-------|")
        
        comp = comparison
        lines.append(f"| Accuracy | {comp['baseline']['accuracy']:.2%} | {comp['finetuned']['accuracy']:.2%} | {comp['improvements']['accuracy_delta']:+.2%} |")
        lines.append(f"| FPR | {comp['baseline']['false_positive_rate']:.2%} | {comp['finetuned']['false_positive_rate']:.2%} | {comp['improvements']['fpr_delta']:+.2%} |")
        lines.append(f"| FNR | {comp['baseline']['false_negative_rate']:.2%} | {comp['finetuned']['false_negative_rate']:.2%} | {comp['improvements']['fnr_delta']:+.2%} |")
        lines.append("")
        
        # Category comparison
        if 'category_comparison' in comp:
            lines.append("### Per-Category Comparison")
            lines.append("")
            lines.append("| Category | Baseline | Fine-Tuned | Improvement |")
            lines.append("|----------|----------|------------|-------------|")
            
            for cat in sorted(comp['category_comparison'].keys()):
                cat_comp = comp['category_comparison'][cat]
                lines.append(f"| {cat} | {cat_comp['baseline']:.2%} | {cat_comp['finetuned']:.2%} | {cat_comp['delta']:+.2%} |")
            lines.append("")
    
    # Category Analysis
    if category_analysis:
        lines.append("## Detailed Category Analysis")
        lines.append("")
        lines.append("```")
        lines.append(category_analysis)
        lines.append("```")
        lines.append("")
    
    # Files
    lines.append("## Generated Files")
    lines.append("")
    lines.append("- `baseline_results.json` - Baseline evaluation results")
    lines.append("- `finetuned_results.json` - Fine-tuned evaluation results")
    lines.append("- `baseline_vs_finetuned.json` - Comparison data")
    lines.append("- `category_analysis.txt` - Detailed category breakdown")
    lines.append("- `final_evaluation_report.md` - This report")
    lines.append("")
    
    return "\n".join(lines)


def main():
    print("="*70)
    print("GENERATING FINAL EVALUATION REPORT")
    print("="*70)
    
    print("\nCompiling results...")
    report = generate_markdown_report()
    
    # Save report
    print(f"\nSaving report to {REPORT_FILE}...")
    with open(REPORT_FILE, 'w') as f:
        f.write(report)
    print("Report saved")
    
    # Print summary
    print("\n" + "="*70)
    print("REPORT SUMMARY")
    print("="*70)
    print("\n" + report[:1000] + "\n...")
    
    print("\n" + "="*70)
    print("REPORT GENERATION COMPLETE")
    print("="*70)
    print(f"\nFull report saved to: {REPORT_FILE}")


if __name__ == '__main__':
    main()
