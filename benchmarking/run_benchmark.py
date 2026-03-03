#!/usr/bin/env python3
"""
Run baseline evaluation for all models, datasets, and system-prompt repeats (1 and 2).
Each run writes a prediction-only JSON; this script then computes metrics and generates
per-run analysis reports and a comparison report. With --skip-run, only recomputes
metrics and regenerates reports from existing prediction files.
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
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"
REPORTS_DIR = RESULTS_DIR / "reports"  # .txt reports; .json results stay in RESULTS_DIR
EVALUATE_SCRIPT = BENCHMARK_DIR / "evaluate_baseline.py"

sys.path.insert(0, str(REPO_ROOT))
from fine_tuning.config import MODEL_OPTIONS
from benchmarking.evaluate_baseline import BENCHMARK_MODEL_OPTIONS, API_MODEL_OPTIONS
from benchmarking.metrics import compute_metrics, generate_detail_report

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
    filter_no_context: bool = True,
    api_concurrency: int = 32,
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
        "--api-concurrency", str(api_concurrency),
    ]
    if filter_no_context:
        cmd.append("--filter-no-context")
    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
        return ((dataset, model_key, repeat), True)
    except subprocess.CalledProcessError:
        return ((dataset, model_key, repeat), False)


def load_predictions(file_path: Path) -> Dict:
    """Load a prediction-only JSON (dataset, model_key, model_id, system_prompt_repeat, predictions)."""
    with open(file_path, "r") as f:
        return json.load(f)


def _parse_baseline_stem(stem: str, prefix: str) -> Optional[Tuple[str, str, int]]:
    """Parse stem like baseline_predictions_ami_qwen2.5-7b_sp1 -> (dataset, model_key, sp)."""
    if not stem.endswith("_sp1") and not stem.endswith("_sp2"):
        return None
    parts = stem.replace(prefix, "").rsplit("_sp", 1)
    if len(parts) != 2:
        return None
    rest, sp_str = parts
    try:
        sp = int(sp_str)
    except ValueError:
        return None
    sub = rest.split("_", 1)
    if len(sub) != 2:
        return None
    return (sub[0], sub[1], sp)


def discover_result_files(
    datasets: List[str],
    models: List[str],
    repeats: List[int],
) -> List[Tuple[Path, str, str, int]]:
    out = []
    for path in RESULTS_DIR.glob("baseline_predictions_*_sp*.json"):
        stem = path.stem
        parsed = _parse_baseline_stem(stem, "baseline_predictions_")
        if not parsed:
            continue
        dataset, model_key, sp = parsed
        if sp not in repeats or dataset not in datasets or model_key not in models:
            continue
        out.append((path, dataset, model_key, sp))
    return sorted(out, key=lambda x: (x[1], x[2], x[3]))


def build_comparison_rows(
    cached: List[Tuple[Path, str, str, int, Dict, Dict]],
) -> List[Dict]:
    """Build comparison rows from cached (path, dataset, model_key, sp, data, metrics)."""
    rows = []
    for _path, dataset, model_key, sp, data, metrics in cached:
        predictions = data.get("predictions", [])
        null_count = sum(1 for p in predictions if p.get("prediction") is None)
        rows.append({
            "dataset": dataset,
            "model": model_key,
            "system_prompt_repeat": data.get("system_prompt_repeat", sp),
            "accuracy": metrics["accuracy"],
            "macro_accuracy": metrics.get("macro_accuracy", 0),
            "false_positive_rate": metrics["false_positive_rate"],
            "false_negative_rate": metrics["false_negative_rate"],
            "precision_speak": metrics.get("precision_speak", 0),
            "recall_speak": metrics.get("recall_speak", 0),
            "precision_silent": metrics.get("precision_silent", 0),
            "recall_silent": metrics.get("recall_silent", 0),
            "f1_speak": metrics.get("f1_speak", 0),
            "f1_silent": metrics.get("f1_silent", 0),
            "macro_f1": metrics.get("macro_f1", 0),
            "latency_mean": metrics["latency_stats"]["mean"],
            "total_samples": metrics["total_samples"],
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
        f"{'Dataset':<10} {'Model':<22} {'SP':<4} {'Acc':>8} {'MacroAcc':>8} {'F1':>8} {'FPR':>8} {'FNR':>8} "
        f"{'Lat(s)':>8} {'N':>6} {'Nulls':>6}"
    )
    lines.append(header)
    lines.append("-" * 110)
    for r in rows:
        line = (
            f"{r['dataset']:<10} {r['model']:<22} {r['system_prompt_repeat']:<4} "
            f"{r['accuracy']:>8.2%} {r.get('macro_accuracy', 0):>8.2%} {r.get('macro_f1', 0):>8.2%} "
            f"{r['false_positive_rate']:>8.2%} {r['false_negative_rate']:>8.2%} "
            f"{r['latency_mean']:>8.4f} {r['total_samples']:>6,} {r['null_predictions']:>6,}"
        )
        lines.append(line)
    lines.append("-" * 110)
    lines.append("")
    lines.append("SP = system prompt repeat. Acc = accuracy, MacroAcc = macro accuracy, F1 = macro F1. FPR/FNR = false positive/negative rate. Nulls = null predictions.")
    lines.append("=" * 90)
    return "\n".join(lines)


def build_per_dataset_category_data(
    cached: List[Tuple[Path, str, str, int, Dict, Dict]],
) -> Dict[str, List[Tuple[str, int, Dict]]]:
    """Group by dataset; each entry is (model_key, sp, full metrics dict)."""
    by_dataset: Dict[str, List[Tuple[str, int, Dict]]] = defaultdict(list)
    for _path, dataset, model_key, sp, _data, metrics in cached:
        by_dataset[dataset].append((model_key, sp, metrics))
    for dataset in by_dataset:
        by_dataset[dataset].sort(key=lambda x: (x[0], x[1]))
    return dict(by_dataset)


def format_category_table(dataset: str, entries: List[Tuple[str, int, Dict]]) -> str:
    """
    Tabular version of all metrics for one dataset: first an overall metrics table,
    then the existing category accuracy grid. entries = [(model_key, sp, metrics), ...].
    """
    if not entries:
        return f"Dataset: {dataset}\nNo results.\n"
    lines = []
    model_width = 22
    w = 8

    # Overall metrics table (same metrics as in baseline_analysis_*.txt)
    lines.append("=" * 120)
    lines.append(f"OVERALL METRICS - {dataset}")
    lines.append("=" * 120)
    lines.append("")
    header = (
        f"{'Model':<{model_width}} {'SP':<4} {'N':>6} {'Acc':>{w}} {'MacroAcc':>{w}} {'MacroF1':>{w}} "
        f"{'SpP':>{w}} {'SpR':>{w}} {'SpF1':>{w}} {'SiP':>{w}} {'SiR':>{w}} {'SiF1':>{w}} "
        f"{'FPR':>{w}} {'FNR':>{w}} {'Lat(s)':>{w}}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for model_key, sp, m in entries:
        row = (
            f"{model_key:<{model_width}} {sp:<4} {m.get('total_samples', 0):>6,} "
            f"{m.get('accuracy', 0):>{w}.2%} {m.get('macro_accuracy', 0):>{w}.2%} {m.get('macro_f1', 0):>{w}.2%} "
            f"{m.get('precision_speak', 0):>{w}.2%} {m.get('recall_speak', 0):>{w}.2%} {m.get('f1_speak', 0):>{w}.2%} "
            f"{m.get('precision_silent', 0):>{w}.2%} {m.get('recall_silent', 0):>{w}.2%} {m.get('f1_silent', 0):>{w}.2%} "
            f"{m.get('false_positive_rate', 0):>{w}.2%} {m.get('false_negative_rate', 0):>{w}.2%} "
            f"{m.get('latency_stats', {}).get('mean', 0):>{w}.4f}"
        )
        lines.append(row)
    lines.append("")
    lines.append("SP = system prompt repeat. Acc = accuracy. SpP/SpR/SpF1 = Speak precision/recall/F1. SiP/SiR/SiF1 = Silent. FPR/FNR = false positive/negative rate. Lat(s) = latency mean.")
    lines.append("")

    # Category accuracy table (data-driven: use all categories present in results)
    cat_acc_entries = [(mk, sp, m.get("category_accuracy") or {}) for mk, sp, m in entries]
    all_cats = set()
    for _mk, _sp, acc in cat_acc_entries:
        all_cats.update((acc or {}).keys())
    cols = sorted(all_cats)
    if not cols:
        lines.append(f"CATEGORY ACCURACY (%) - {dataset}")
        lines.append("No category accuracy data.")
        return "\n".join(lines)
    col_width = 8
    lines.append("=" * (model_width + 4 + len(cols) * (col_width + 1)))
    lines.append(f"CATEGORY ACCURACY (%) - {dataset}")
    lines.append("=" * (model_width + 4 + len(cols) * (col_width + 1)))
    lines.append("")
    header2 = f"{'Model':<{model_width}} {'SP':<4} " + " ".join(f"{c:>{col_width}}" for c in cols)
    lines.append(header2)
    lines.append("-" * len(header2))
    for model_key, sp, cat_acc in cat_acc_entries:
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
    lines.append("SP = system prompt repeat. Columns = per-category accuracy (%).")
    lines.append("=" * (model_width + 4 + len(cols) * (col_width + 1)))
    return "\n".join(lines)


def main(
    skip_run: bool = False,
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    repeats: Optional[List[int]] = None,
    max_parallel: Optional[int] = None,
    filter_no_context: bool = True,
    api_concurrency: int = 32,
) -> None:
    datasets = datasets or get_datasets_with_test()
    models = models or list(BENCHMARK_MODEL_OPTIONS.keys())
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
        api_only = all(m in API_MODEL_OPTIONS for m in models)
        if max_parallel is None:
            if num_gpus > 0:
                max_parallel = min(4, num_gpus)
            elif api_only:
                max_parallel = 4  # API runs use RAM only; no GPU needed
            else:
                max_parallel = 1
        max_parallel = max(1, max_parallel)
        if num_gpus > 0 and max_parallel > num_gpus and not api_only:
            max_parallel = num_gpus

        print("=" * 70)
        print("RUNNING BASELINE EVALUATIONS")
        print("=" * 70)
        print(f"Datasets: {datasets}")
        print(f"Models: {models}")
        print(f"System prompt repeats: {repeats}")
        print(f"Filter no-context samples: {filter_no_context}")
        if api_only:
            print(f"API concurrency per job: {api_concurrency}")
        print(f"Total jobs: {total}")
        if max_parallel > 1:
            if api_only:
                print(f"Parallel: {max_parallel} workers (API-only; no GPU, moderate RAM per worker)")
            else:
                print(f"Parallel: {max_parallel} workers (1 GPU each, {num_gpus} GPUs available)")
        else:
            print("Parallel: 1 (sequential)")

        failed = []
        if max_parallel <= 1:
            for i, (dataset, model_key, repeat) in enumerate(jobs, 1):
                print(f"\n[{i}/{total}] {dataset} / {model_key} / SP repeat {repeat}")
                _, ok = run_one(
                    dataset, model_key, repeat, gpu_id=None,
                    filter_no_context=filter_no_context, api_concurrency=api_concurrency,
                )
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
                        filter_no_context,
                        api_concurrency,
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
        print("Skipping runs (--skip-run). Using existing prediction files.")

    result_entries = discover_result_files(datasets, models, repeats)
    if not result_entries:
        print("No prediction files found. Run without --skip-run first.")
        return

    cached = []
    for path, dataset, model_key, sp in result_entries:
        try:
            data = load_predictions(path)
            predictions = data.get("predictions", [])
            metrics = compute_metrics(predictions)
            cached.append((path, dataset, model_key, sp, data, metrics))
        except Exception as e:
            print(f"  Warning: could not load or compute metrics for {path}: {e}")

    rows = build_comparison_rows(cached)
    comparison_report = format_comparison_report(rows)
    comparison_path = REPORTS_DIR / "benchmark_comparison.txt"
    with open(comparison_path, "w") as f:
        f.write(comparison_report)
    print("\n" + comparison_report)
    print(f"\nComparison saved to {comparison_path}")

    for path, dataset, model_key, sp, data, metrics in cached:
        try:
            report = generate_detail_report({**metrics, "predictions": data.get("predictions", [])})
            detail_path = REPORTS_DIR / f"baseline_analysis_{dataset}_{model_key}_sp{sp}.txt"
            with open(detail_path, "w") as f:
                f.write(report)
        except Exception as e:
            print(f"  Warning: could not generate detail report for {path}: {e}")

    per_dataset = build_per_dataset_category_data(cached)
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
        help="Only generate reports from existing prediction files (no model runs).",
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
        choices=list(BENCHMARK_MODEL_OPTIONS.keys()),
        help="Model keys to run (default: all, including API models gpt-5.2, gemini-3.1-pro).",
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
        help="Max evaluations in parallel. GPU runs: 1 GPU per worker (default: min(4, num_gpus)). API-only runs: default 4 workers, no GPU (more RAM). Use 2 for less RAM. Set 1 for sequential.",
    )
    parser.add_argument(
        "--filter-no-context",
        action="store_true",
        default=True,
        help="Exclude samples with no context_turns from baseline evaluation (default: True).",
    )
    parser.add_argument(
        "--no-filter-no-context",
        action="store_false",
        dest="filter_no_context",
        help="Do not filter; include samples with no context_turns.",
    )
    parser.add_argument(
        "--api-concurrency",
        type=int,
        default=32,
        metavar="N",
        help="Max concurrent API requests per job when using OpenAI/Gemini (default: 32).",
    )
    args = parser.parse_args()
    main(
        skip_run=args.skip_run,
        datasets=args.datasets,
        models=args.models,
        repeats=args.repeats,
        max_parallel=args.max_parallel,
        filter_no_context=args.filter_no_context,
        api_concurrency=args.api_concurrency,
    )
