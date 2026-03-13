#!/usr/bin/env python3
"""
Prepare data: load samples from data/{dataset}/ and create train/val/test splits.

Run data/download.py first to populate data/{dataset}/ from HuggingFace.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import random
from collections import Counter


BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42

# Metadata/summary files that are not sample data
EXCLUDED_DATA_FILES = {'filtering_summary.json'}


def list_local_files(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        print("Run `python data/download.py` to download the dataset first.")
        return []

    files = []
    for ext in ['*.json', '*.jsonl']:
        files.extend(data_dir.glob(ext))

    files = [f for f in files if f.name not in EXCLUDED_DATA_FILES]
    files = sorted(files)
    print(f"Found {len(files)} files in {data_dir}")
    return files


def load_jsonl_file(file_path: Path) -> List[Dict]:
    samples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num} of {file_path}: {e}")
        return samples
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def load_json_file(file_path: Path) -> List[Dict]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                for key in ['samples', 'data', 'items']:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                return [data]
            return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def split_samples(samples: List[Dict], train_ratio: float, val_ratio: float,
                  test_ratio: float, seed: int = 42) -> tuple:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def save_samples_jsonl(samples: List[Dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def print_split_statistics(train_samples, val_samples, test_samples):
    def get_decision_counts(samples):
        return Counter(s.get('decision', 'UNKNOWN') for s in samples)

    total = len(train_samples) + len(val_samples) + len(test_samples)
    print("\n" + "=" * 70)
    print("DATA SPLIT STATISTICS")
    print("=" * 70)
    print(f"\nTotal samples: {total:,}")
    print(f"  Train: {len(train_samples):,} ({len(train_samples)/total*100:.1f}%)")
    print(f"  Val:   {len(val_samples):,} ({len(val_samples)/total*100:.1f}%)")
    print(f"  Test:  {len(test_samples):,} ({len(test_samples)/total*100:.1f}%)")

    for split_name, split_data in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        print(f"\n{split_name} set decisions:")
        for decision, count in get_decision_counts(split_data).most_common():
            print(f"  {decision}: {count:,} ({count/len(split_data)*100:.1f}%)")

    print("\n" + "=" * 70)


def main(dataset: str = 'ami'):
    DATA_DIR = BASE_DIR / 'data' / dataset
    TRAIN_DIR = DATA_DIR / 'train'
    VAL_DIR = DATA_DIR / 'val'
    TEST_DIR = DATA_DIR / 'test'

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"LOADING DATA FOR DATASET: {dataset.upper()}")
    print("LOADING DATA FROM LOCAL FILES")
    print("=" * 70)

    print(f"\nLoading files from {DATA_DIR}...")
    local_files = list_local_files(DATA_DIR)

    if not local_files:
        return

    all_samples = []
    for i, file_path in enumerate(local_files, 1):
        print(f"  [{i}/{len(local_files)}] Loading {file_path.name}...", end=' ')
        if file_path.suffix == '.jsonl':
            samples = load_jsonl_file(file_path)
        else:
            samples = load_json_file(file_path)
        all_samples.extend(samples)
        print(f"({len(samples)} samples)")

    print(f"\nTotal samples loaded: {len(all_samples):,}")

    if not all_samples:
        print("No samples loaded. Exiting.")
        return

    print(f"\nSplitting into train/val/test ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%})...")
    train_samples, val_samples, test_samples = split_samples(
        all_samples, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    print("\nSaving splits...")
    save_samples_jsonl(train_samples, TRAIN_DIR / 'train_samples.jsonl')
    save_samples_jsonl(val_samples, VAL_DIR / 'val_samples.jsonl')
    save_samples_jsonl(test_samples, TEST_DIR / 'test_samples.jsonl')

    print(f"  Train: {TRAIN_DIR / 'train_samples.jsonl'}")
    print(f"  Val:   {VAL_DIR / 'val_samples.jsonl'}")
    print(f"  Test:  {TEST_DIR / 'test_samples.jsonl'}")

    print_split_statistics(train_samples, val_samples, test_samples)

    print("\n" + "=" * 70)
    print("DATA LOADING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load and split dataset samples')
    parser.add_argument('--dataset', type=str, default='ami',
                        choices=['ami', 'friends', 'spgi'],
                        help='Dataset name (default: ami)')
    args = parser.parse_args()
    main(dataset=args.dataset)
