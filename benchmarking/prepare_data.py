#!/usr/bin/env python3
"""
Prepare data: load samples from data/{dataset}/ and create train/val/test splits.
Supports local files or S3 when USE_S3 is True.
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import random
from collections import Counter

try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

USE_S3 = False

BASE_DIR = Path(__file__).resolve().parents[1]

S3_BUCKET = 'ishiki-ml-datasets'
S3_PREFIX = 'AMI_samples/json_run/'

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42


def list_local_files(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return []
    
    files = []
    for ext in ['*.json', '*.jsonl']:
        files.extend(data_dir.glob(ext))
    
    files = sorted(files)
    print(f"Found {len(files)} files in {data_dir}")
    return files



def list_s3_files(bucket: str, prefix: str) -> List[str]:
    s3_client = boto3.client('s3')
    files = []
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.json') or key.endswith('.jsonl'):
                        files.append(key)
        
        print(f"Found {len(files)} files in s3://{bucket}/{prefix}")
        return files
    
    except Exception as e:
        print(f"Error listing S3 files: {e}")
        print("You may need to configure AWS credentials or use public access")
        return []


def download_s3_file(bucket: str, key: str, local_path: Path) -> bool:
    if not S3_AVAILABLE:
        print("Error: boto3 not available. Install with: pip install boto3")
        return False
    
    s3_client = boto3.client('s3')
    
    try:
        s3_client.download_file(bucket, key, str(local_path))
        return True
    except Exception as e:
        print(f"Error downloading {key}: {e}")
        return False


def load_jsonl_file(file_path: Path) -> List[Dict]:
    samples = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
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
            else:
                return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []



def split_samples(samples: List[Dict], train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42) -> tuple:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_samples = shuffled[:n_train]
    val_samples = shuffled[n_train:n_train + n_val]
    test_samples = shuffled[n_train + n_val:]
    
    return train_samples, val_samples, test_samples


def save_samples_jsonl(samples: List[Dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def print_split_statistics(train_samples: List[Dict], val_samples: List[Dict], test_samples: List[Dict]):
    def get_decision_counts(samples):
        decisions = [s.get('decision', 'UNKNOWN') for s in samples]
        return Counter(decisions)
    
    print("\n" + "="*70)
    print("DATA SPLIT STATISTICS")
    print("="*70)
    
    print(f"\nTotal samples: {len(train_samples) + len(val_samples) + len(test_samples):,}")
    print(f"  Train: {len(train_samples):,} ({len(train_samples)/(len(train_samples)+len(val_samples)+len(test_samples))*100:.1f}%)")
    print(f"  Val:   {len(val_samples):,} ({len(val_samples)/(len(train_samples)+len(val_samples)+len(test_samples))*100:.1f}%)")
    print(f"  Test:  {len(test_samples):,} ({len(test_samples)/(len(train_samples)+len(val_samples)+len(test_samples))*100:.1f}%)")
    
    train_decisions = get_decision_counts(train_samples)
    val_decisions = get_decision_counts(val_samples)
    test_decisions = get_decision_counts(test_samples)
    
    print("\nTrain set decisions:")
    for decision, count in train_decisions.most_common():
        print(f"  {decision}: {count:,} ({count/len(train_samples)*100:.1f}%)")
    
    print("\nVal set decisions:")
    for decision, count in val_decisions.most_common():
        print(f"  {decision}: {count:,} ({count/len(val_samples)*100:.1f}%)")
    
    print("\nTest set decisions:")
    for decision, count in test_decisions.most_common():
        print(f"  {decision}: {count:,} ({count/len(test_samples)*100:.1f}%)")
    
    print("\n" + "="*70)



def main(dataset: str = 'ami'):
    DATA_DIR = BASE_DIR / 'data' / dataset
    RAW_DATA_DIR = DATA_DIR
    TRAIN_DIR = DATA_DIR / 'train'
    VAL_DIR = DATA_DIR / 'val'
    TEST_DIR = DATA_DIR / 'test'
    
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(f"LOADING DATA FOR DATASET: {dataset.upper()}")
    if USE_S3:
        print("DOWNLOADING DATA FROM S3")
    else:
        print("LOADING DATA FROM LOCAL FILES")
    print("="*70)
    
    all_samples = []
    
    if USE_S3:
        if not S3_AVAILABLE:
            print("\nError: boto3 not available. Install with: pip install boto3")
            print("Or set USE_S3 = False to use local files.")
            return
        
        print(f"\nListing files in s3://{S3_BUCKET}/{S3_PREFIX}...")
        s3_files = list_s3_files(S3_BUCKET, S3_PREFIX)
        
        if not s3_files:
            print("\nNo files found. Please check:")
            print("  1. AWS credentials are configured")
            print("  2. S3 bucket and prefix are correct")
            print("  3. Network connectivity")
            return
        
        print(f"\nDownloading {len(s3_files)} files...")
        temp_dir = DATA_DIR / 'temp_download'
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        for i, s3_key in enumerate(s3_files, 1):
            filename = os.path.basename(s3_key)
            local_path = temp_dir / filename
            
            print(f"  [{i}/{len(s3_files)}] Downloading {filename}...", end=' ')
            
            if download_s3_file(S3_BUCKET, s3_key, local_path):
                if filename.endswith('.jsonl'):
                    samples = load_jsonl_file(local_path)
                else:
                    samples = load_json_file(local_path)
                
                all_samples.extend(samples)
                print(f"({len(samples)} samples)")
            else:
                print("Failed")
        
        import shutil
        shutil.rmtree(temp_dir)
    
    else:
        print(f"\nLoading files from {RAW_DATA_DIR}...")
        local_files = list_local_files(RAW_DATA_DIR)
        
        if not local_files:
            print(f"\nNo files found in {RAW_DATA_DIR}")
            print("Please ensure the data files are in the correct location.")
            return
        
        print(f"\nLoading {len(local_files)} files...")
        
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
    
    print("\n" + "="*70)
    print("DATA LOADING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load and split dataset samples')
    parser.add_argument('--dataset', type=str, default='ami', 
                       choices=['ami', 'friends', 'spgi'],
                       help='Dataset name (default: ami)')
    args = parser.parse_args()
    main(dataset=args.dataset)
