#!/usr/bin/env python3
"""
Run baseline evaluation for all models, datasets, and system-prompt repeats (1 and 2),
then generate per-run analysis reports and a comparison report.
When multiple GPUs are available, runs up to 4 evaluations in parallel (one GPU per run).
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"
REPORTS_DIR = RESULTS_DIR / "reports"  # .txt reports; .json results stay in RESULTS_DIR
EVALUATE_SCRIPT = BENCHMARK_DIR / "evaluate_baseline.py"

sys.path.insert(0, str(REPO_ROOT))
from fine_tuning.config import MODEL_OPTIONS
from utils.constants import (
    ALL_CATEGORIES,
    CATEGORY_NAMES,
    SILENT_CATEGORIES,
    SPEAK_CATEGORIES,
)

DATASETS = ["ami", "friends", "spgi"]


def get_datasets_with_test() -> List[str]:
    out = []
    for d in DATASETS:
        test_file = REPO_ROOT / "data" / d / "test" / "test_samples.jsonl"
        if test_file.exists():
            out.append(d)
    return out


def get_num_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout:
            return len([l for l in out.stdout.strip().split("\n") if l.startswith("GPU ")])
    except Exception:
        pass
    return 0


def run_one(
    dataset: str,
    model_key: str,
    repeat: int,
    gpu_id: Optional[int] = None,
) -> Tuple[Tuple[str, str, int], bool]:
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--dataset", dataset,
        "--model", model_key,
        "--system-prompt-repeat", str(repeat),
    ]
    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
        return ((dataset, model_key, repeat), True)
    except subprocess.CalledProcessError:
        return ((dataset, model_key, repeat), False)


def load_results(file_path: Path) -> Dict:
    with open(file_path, "r") as f:
        return json.load(f)


def analyze_errors(results: Dict) -> Dict:
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
    lines = []
    lines.append("=" * 70)
    lines.append("BASELINE EVALUATION ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("OVERALL METRICS")
    lines.append("-" * 70)
    lines.append(f"Total samples: {results['total_samples']:,}")
    lines.append(f"Accuracy: {results['accuracy']:.2%}")
    lines.append(f"Correct: {results['correct']:,}")
    lines.append(f"Incorrect: {results['incorrect']:,}")
    preds = results.get("predictions", [])
    nulls = sum(1 for p in preds if p.get("prediction") is None)
    if preds:
        lines.append(f"Null predictions: {nulls:,} ({nulls / len(preds):.1%})")
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
            metrics = results["category_accuracy"][cat]
            cat_name = CATEGORY_NAMES.get(cat, cat)
            lines.append(f"  {cat} ({cat_name}):")
            lines.append(f"    Accuracy: {metrics['accuracy']:.2%}")
            lines.append(f"    Correct: {metrics['correct']}/{metrics['total']}")
    lines.append("\nSILENT Categories:")
    for cat in SILENT_CATEGORIES:
        if cat in results.get("category_accuracy", {}):
            metrics = results["category_accuracy"][cat]
            cat_name = CATEGORY_NAMES.get(cat, cat)
            lines.append(f"  {cat} ({cat_name}):")
            lines.append(f"    Accuracy: {metrics['accuracy']:.2%}")
            lines.append(f"    Correct: {metrics['correct']}/{metrics['total']}")
    lines.append("")
    lines.append("CONFUSION MATRIX")
    lines.append("-" * 70)
    confusion = results.get("confusion_matrix", {})
    for pattern, count in sorted(confusion.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {pattern}: {count}")
    lines.append("")
    errors = analyze_errors(results)
    lines.append("ERROR ANALYSIS")
    lines.append("-" * 70)
    lines.append("\nError Rate by Category:")
    for cat in sorted(errors["by_category"].keys()):
        metrics = errors["by_category"][cat]
        rate = metrics["errors"] / metrics["total"] if metrics["total"] > 0 else 0
        cat_name = CATEGORY_NAMES.get(cat, cat)
        lines.append(f"  {cat} ({cat_name}): {rate:.2%} ({metrics['errors']}/{metrics['total']})")
    lines.append("\nError Rate by Ground Truth Decision:")
    for decision in sorted(errors["by_decision"].keys()):
        metrics = errors["by_decision"][decision]
        rate = metrics["errors"] / metrics["total"] if metrics["total"] > 0 else 0
        lines.append(f"  {decision}: {rate:.2%} ({metrics['errors']}/{metrics['total']})")
    lines.append("\nMost Common Confusion Patterns:")
    for pattern, count in errors["confusion_patterns"].most_common(10):
        lines.append(f"  {pattern}: {count}")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def discover_result_files(
    datasets: List[str],
    models: List[str],
    repeats: List[int],
) -> List[Tuple[Path, str, str, int]]:
    out = []
    for path in RESULTS_DIR.glob("baseline_results_*_sp*.json"):
        stem = path.stem
        if not stem.endswith("_sp1") and not stem.endswith("_sp2"):
            continue
        parts = stem.replace("baseline_results_", "").rsplit("_sp", 1)
        if len(parts) != 2:
            continue
        rest, sp_str = parts
        try:
            sp = int(sp_str)
        except ValueError:
            continue
        if sp not in repeats:
            continue
        sub = rest.split("_", 1)
        if len(sub) != 2:
            continue
        dataset, model_key = sub[0], sub[1]
        if dataset not in datasets or model_key not in models:
            continue
        out.append((path, dataset, model_key, sp))
    return sorted(out, key=lambda x: (x[1], x[2], x[3]))


def build_comparison_rows(
    result_entries: List[Tuple[Path, str, str, int]],
) -> List[Dict]:
    rows = []
    for path, dataset, model_key, sp in result_entries:
        try:
            r = load_results(path)
        except Exception:
            continue
        sp_val = r.get("system_prompt_repeat", sp)
        predictions = r.get("predictions", [])
        null_count = sum(1 for p in predictions if p.get("prediction") is None)
        rows.append({
            "dataset": dataset,
            "model": model_key,
            "system_prompt_repeat": sp_val,
            "accuracy": r["accuracy"],
            "false_positive_rate": r["false_positive_rate"],
            "false_negative_rate": r["false_negative_rate"],
            "latency_mean": r["latency_stats"]["mean"],
            "total_samples": r["total_samples"],
            "null_predictions": null_count,
        })
    return rows


def format_comparison_report(rows: List[Dict]) -> str:
    if not rows:
        return "No results to compare."
    lines = []
    lines.append("=" * 90)
    lines.append("BENCHMARK COMPARISON (all models x datasets x system prompt repeat 1 & 2)")
    lines.append("=" * 90)
    lines.append("")
    header = (
        f"{'Dataset':<10} {'Model':<22} {'SP':<4} {'Accuracy':>10} {'FPR':>10} {'FNR':>10} "
        f"{'Latency(s)':>10} {'N':>8} {'Nulls':>8}"
    )
    lines.append(header)
    lines.append("-" * 100)
    for r in rows:
        line = (
            f"{r['dataset']:<10} {r['model']:<22} {r['system_prompt_repeat']:<4} "
            f"{r['accuracy']:>10.2%} {r['false_positive_rate']:>10.2%} {r['false_negative_rate']:>10.2%} "
            f"{r['latency_mean']:>10.4f} {r['total_samples']:>8,} {r['null_predictions']:>8,}"
        )
        lines.append(line)
    lines.append("-" * 100)
    lines.append("")
    lines.append("SP = system prompt repeat. FPR = false positive rate, FNR = false negative rate. Nulls = null predictions (failed extraction).")
    lines.append("=" * 90)
    return "\n".join(lines)


def build_per_dataset_category_data(
    result_entries: List[Tuple[Path, str, str, int]],
) -> Dict[str, List[Tuple[str, int, Dict[str, Dict]]]]:
    """Group result entries by dataset and load category_accuracy for each. Returns dataset -> [(model_key, sp, category_accuracy), ...]."""
    by_dataset: Dict[str, List[Tuple[str, int, Dict[str, Dict]]]] = defaultdict(list)
    for path, dataset, model_key, sp in result_entries:
        try:
            r = load_results(path)
            cat_acc = r.get("category_accuracy") or {}
            by_dataset[dataset].append((model_key, sp, cat_acc))
        except Exception:
            continue
    for dataset in by_dataset:
        by_dataset[dataset].sort(key=lambda x: (x[0], x[1]))
    return dict(by_dataset)


def format_category_table(dataset: str, entries: List[Tuple[str, int, Dict[str, Dict]]]) -> str:
    """Format one dataset's category accuracy as a table. Columns = I1, I2, I3, S1, S2, S3, S4, S5."""
    if not entries:
        return f"Dataset: {dataset}\nNo results.\n"
    cols = [c for c in ALL_CATEGORIES if any(c in (e[2] or {}) for e in entries)]
    if not cols:
        return f"Dataset: {dataset}\nNo category accuracy data.\n"
    col_width = 8
    model_width = 22
    lines = []
    lines.append("=" * (model_width + 4 + len(cols) * (col_width + 1)))
    lines.append(f"CATEGORY ACCURACY (%) — {dataset}")
    lines.append("=" * (model_width + 4 + len(cols) * (col_width + 1)))
    lines.append("")
    header = f"{'Model':<{model_width}} {'SP':<4} " + " ".join(f"{c:>{col_width}}" for c in cols)
    lines.append(header)
    lines.append("-" * len(header))
    for model_key, sp, cat_acc in entries:
        cells = []
        for c in cols:
            if c in cat_acc and cat_acc[c].get("total", 0) > 0:
                pct = cat_acc[c]["accuracy"] * 100
                cells.append(f"{pct:>{col_width}.2f}")
            else:
                cells.append(f"{'-':>{col_width}}")
        line = f"{model_key:<{model_width}} {sp:<4} " + " ".join(cells)
        lines.append(line)
    lines.append("")
    lines.append("SP = system prompt repeat. I = SPEAK categories, S = SILENT categories.")
    lines.append("=" * (model_width + 4 + len(cols) * (col_width + 1)))
    return "\n".join(lines)


def main(
    skip_run: bool = False,
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    repeats: Optional[List[int]] = None,
    max_parallel: Optional[int] = None,
) -> None:
    datasets = datasets or get_datasets_with_test()
    models = models or list(MODEL_OPTIONS.keys())
    repeats = repeats or [1, 2]

    if not datasets:
        print("No datasets with test split found. Run prepare_data.py for at least one dataset.")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    if not skip_run:
        jobs = [
            (dataset, model_key, repeat)
            for dataset in datasets
            for model_key in models
            for repeat in repeats
        ]
        total = len(jobs)
        num_gpus = get_num_gpus()
        if max_parallel is None:
            max_parallel = min(4, num_gpus) if num_gpus > 0 else 1
        max_parallel = max(1, max_parallel)
        if num_gpus > 0 and max_parallel > num_gpus:
            max_parallel = num_gpus

        print("=" * 70)
        print("RUNNING BASELINE EVALUATIONS")
        print("=" * 70)
        print(f"Datasets: {datasets}")
        print(f"Models: {models}")
        print(f"System prompt repeats: {repeats}")
        print(f"Total jobs: {total}")
        if max_parallel > 1 and num_gpus > 0:
            print(f"Parallel: {max_parallel} workers (1 GPU each, {num_gpus} GPUs available)")
        else:
            print("Parallel: 1 (sequential)")

        failed = []
        if max_parallel <= 1:
            for i, (dataset, model_key, repeat) in enumerate(jobs, 1):
                print(f"\n[{i}/{total}] {dataset} / {model_key} / SP repeat {repeat}")
                _, ok = run_one(dataset, model_key, repeat, gpu_id=None)
                if not ok:
                    failed.append((dataset, model_key, repeat))
        else:
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                futures = {
                    executor.submit(
                        run_one,
                        dataset,
                        model_key,
                        repeat,
                        i % max_parallel,
                    ): (dataset, model_key, repeat)
                    for i, (dataset, model_key, repeat) in enumerate(jobs)
                }
                done = 0
                for future in as_completed(futures):
                    (dataset, model_key, repeat), ok = future.result()
                    done += 1
                    label = f"{dataset} / {model_key} / sp{repeat}"
                    status = "ok" if ok else "FAILED"
                    print(f"  [{done}/{total}] {label} ... {status}")
                    if not ok:
                        failed.append((dataset, model_key, repeat))

        if failed:
            print(f"\nFailed runs: {len(failed)}")
            for t in failed:
                print(f"  {t[0]} {t[1]} sp{t[2]}")
        print("\nEvaluations done.")
    else:
        print("Skipping runs (--skip-run). Using existing result files.")

    result_entries = discover_result_files(datasets, models, repeats)
    if not result_entries:
        print("No result files found. Run without --skip-run first.")
        return

    rows = build_comparison_rows(result_entries)
    comparison_report = format_comparison_report(rows)
    comparison_path = REPORTS_DIR / "benchmark_comparison.txt"
    with open(comparison_path, "w") as f:
        f.write(comparison_report)
    print("\n" + comparison_report)
    print(f"\nComparison saved to {comparison_path}")

    for path, dataset, model_key, sp in result_entries:
        try:
            r = load_results(path)
            report = generate_detail_report(r)
            detail_path = REPORTS_DIR / f"baseline_analysis_{dataset}_{model_key}_sp{sp}.txt"
            with open(detail_path, "w") as f:
                f.write(report)
        except Exception as e:
            print(f"  Warning: could not generate detail report for {path}: {e}")

    # Per-dataset category tables (I and S categories)
    per_dataset = build_per_dataset_category_data(result_entries)
    for dataset in sorted(per_dataset.keys()):
        table = format_category_table(dataset, per_dataset[dataset])
        table_path = REPORTS_DIR / f"category_table_{dataset}.txt"
        with open(table_path, "w") as f:
            f.write(table)
        print(f"\nCategory table saved to {table_path}")
        print(table)

    comparison_json = RESULTS_DIR / "benchmark_comparison.json"
    with open(comparison_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Comparison JSON saved to {comparison_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run baseline for all models x datasets x system prompt repeat 1&2, then report."
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only generate reports from existing result files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=DATASETS,
        help=f"Datasets to run (default: all with test split). Choices: {DATASETS}",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(MODEL_OPTIONS.keys()),
        help="Model keys to run (default: all).",
    )
    parser.add_argument(
        "--system-prompt-repeats",
        nargs="+",
        type=int,
        default=None,
        choices=[1, 2],
        dest="repeats",
        help="System prompt repeat values to run (default: 1 2).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        metavar="N",
        help="Max evaluations in parallel, 1 GPU each (default: min(4, num_gpus)). Set 1 to force sequential.",
    )
    args = parser.parse_args()
    main(
        skip_run=args.skip_run,
        datasets=args.datasets,
        models=args.models,
        repeats=args.repeats,
        max_parallel=args.max_parallel,
    )
