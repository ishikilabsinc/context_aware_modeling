"""
Compute human evaluation stats from a human_eval_results_{name}.jsonl file.

Usage:
    python eval_stats/compute_human_eval_stats.py eval_stats/human_eval_results_kratika.jsonl

Output:
    eval_stats/human_eval_stats_{name}.json
"""

import json
import sys
import re
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

CATEGORY_MAP = {
    "S1": {"key": "SILENT_no_ref",  "positive_judgement": "SILENT"},
    "S2": {"key": "SILENT_ref",     "positive_judgement": "SILENT"},
    "I1": {"key": "SPEAK_implicit", "positive_judgement": "SPEAK"},
    "I2": {"key": "SPEAK_explicit", "positive_judgement": "SPEAK"},
}

# Reverse lookup: category field value → label
KEY_TO_LABEL = {v["key"]: label for label, v in CATEGORY_MAP.items()}


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_per_category(records: list[dict]) -> dict:
    buckets: dict[str, dict] = {
        label: {"category_key": info["key"], "total": 0, "correct": 0}
        for label, info in CATEGORY_MAP.items()
    }

    for r in records:
        cat_key = r.get("category")
        label = KEY_TO_LABEL.get(cat_key)
        if label is None:
            continue
        human = r.get("human_judgement")
        positive = CATEGORY_MAP[label]["positive_judgement"]
        buckets[label]["total"] += 1
        if human == positive:
            buckets[label]["correct"] += 1

    per_category = {}
    for label, b in buckets.items():
        total = b["total"]
        correct = b["correct"]
        per_category[label] = {
            "category_key": b["category_key"],
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total > 0 else None,
        }
    return per_category


def compute_overall(records: list[dict]) -> dict:
    y_true = [r["decision"] for r in records]
    y_pred = [r["human_judgement"] for r in records]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_human_eval_stats.py <path/to/human_eval_results_{name}.jsonl>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    match = re.match(r"human_eval_results_(.+)\.jsonl$", input_path.name)
    if not match:
        print(f"Filename must match pattern 'human_eval_results_{{name}}.jsonl', got: {input_path.name}")
        sys.exit(1)
    name = match.group(1)

    records = load_records(input_path)
    print(f"Loaded {len(records)} records from {input_path}")

    per_category = compute_per_category(records)
    overall = compute_overall(records)

    result = {
        "name": name,
        "total_samples": len(records),
        "per_category": per_category,
        "overall": overall,
    }

    output_path = input_path.parent / f"human_eval_stats_{name}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved stats to {output_path}")
    print("\n--- Results ---")
    print(f"Total samples: {len(records)}")
    print("\nPer-category accuracy:")
    for label, stats in per_category.items():
        print(f"  {label} ({stats['category_key']}): {stats['correct']}/{stats['total']} = {stats['accuracy']}")
    print("\nOverall:")
    print(f"  Accuracy:          {overall['accuracy']}")
    print(f"  F1-macro:          {overall['f1_macro']}")
    print(f"  Balanced Accuracy: {overall['balanced_accuracy']}")


if __name__ == "__main__":
    main()
