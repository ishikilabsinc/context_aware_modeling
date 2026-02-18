#!/usr/bin/env python3
"""
Compile evaluation results (baseline, fine-tuned, comparison, category analysis) into one markdown report.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
BENCHMARK_RESULTS = BASE_DIR / "benchmarking" / "results"
RESULTS_DIR = Path(__file__).parent / "results"


def load_json_file(file_path: Path):
    if not file_path or not file_path.exists():
        return None
    with open(file_path, "r") as f:
        data = json.load(f)
    if file_path.name.startswith("baseline_") and "accuracy" not in data:
        from benchmarking.metrics import compute_metrics
        predictions = data.get("predictions", [])
        metrics = compute_metrics(predictions)
        return {**metrics, "predictions": predictions}
    return data


def load_text_file(file_path: Path):
    if not file_path or not file_path.exists():
        return None
    with open(file_path, "r") as f:
        return f.read()


def find_baseline_file(dataset: str, model_key: str) -> Path:
    for name in [
        f"baseline_predictions_{dataset}_{model_key}_sp1.json",
        f"baseline_predictions_{dataset}_{model_key}_sp2.json",
    ]:
        p = BENCHMARK_RESULTS / name
        if p.exists():
            return p
    return None


def generate_markdown_report(dataset: str, model_key: str) -> str:
    baseline_file = find_baseline_file(dataset, model_key)
    finetuned_file = RESULTS_DIR / f"finetuned_results_{dataset}_{model_key}.json"
    comparison_file = RESULTS_DIR / f"baseline_vs_finetuned_{dataset}_{model_key}.json"
    category_file = RESULTS_DIR / f"category_analysis_{dataset}_{model_key}.txt"

    lines = []
    lines.append("# Context-Aware Turn-Taking Model: Final Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"Dataset: {dataset} | Model: {model_key}")
    lines.append("")
    lines.append("---")
    lines.append("")

    baseline = load_json_file(baseline_file)
    finetuned = load_json_file(finetuned_file)
    comparison = load_json_file(comparison_file)
    category_analysis = load_text_file(category_file)

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
        if "category_comparison" in comp:
            lines.append("### Per-Category Comparison")
            lines.append("")
            lines.append("| Category | Baseline | Fine-Tuned | Improvement |")
            lines.append("|----------|----------|------------|-------------|")
            
            for cat in sorted(comp['category_comparison'].keys()):
                cat_comp = comp['category_comparison'][cat]
                lines.append(f"| {cat} | {cat_comp['baseline']:.2%} | {cat_comp['finetuned']:.2%} | {cat_comp['delta']:+.2%} |")
            lines.append("")
    if category_analysis:
        lines.append("## Detailed Category Analysis")
        lines.append("")
        lines.append("```")
        lines.append(category_analysis)
        lines.append("```")
        lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    lines.append(f"- `finetuned_results_{dataset}_{model_key}.json`")
    lines.append(f"- `baseline_vs_finetuned_{dataset}_{model_key}.json`")
    lines.append(f"- `category_analysis_{dataset}_{model_key}.txt`")
    lines.append(f"- `final_evaluation_report_{dataset}_{model_key}.md`")
    lines.append("")
    return "\n".join(lines)


def main(dataset: str = "ami", model: str = None):
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from fine_tuning.config import MODEL_OPTIONS, MODEL as DEFAULT_MODEL
    model_key = model or DEFAULT_MODEL
    if model_key not in MODEL_OPTIONS:
        model_key = DEFAULT_MODEL

    report_file = RESULTS_DIR / f"final_evaluation_report_{dataset}_{model_key}.md"
    print("=" * 70)
    print("GENERATING FINAL EVALUATION REPORT")
    print("=" * 70)
    report = generate_markdown_report(dataset, model_key)
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Report saved to {report_file}")
    print("\n" + report[:800] + "\n...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate final evaluation report")
    parser.add_argument("--dataset", type=str, default="ami", choices=["ami", "friends", "spgi"])
    parser.add_argument("--model", type=str, default=None, help="Model key (default: from MODEL env)")
    args = parser.parse_args()
    main(dataset=args.dataset, model=args.model)
