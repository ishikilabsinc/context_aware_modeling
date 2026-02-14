#!/usr/bin/env python3
"""
Validate Data Format

Validates that downloaded samples match the expected training format and generates statistics.

Expected format (from ami/stage5_format_training.py):
- Input sections: <|system|>, <|instruction|>, <|context|>, <|current|>
- Output format: <decision>, <confidence>, <reason>
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter, defaultdict


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'

# Expected sections in training format
REQUIRED_SECTIONS = ['system', 'instruction', 'context', 'current']
REQUIRED_OUTPUT_TAGS = ['decision', 'confidence', 'reason']

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.constants import SPEAK_CATEGORIES, SILENT_CATEGORIES, ALL_CATEGORIES
from utils.data_utils import load_samples


def validate_training_format(sample: Dict) -> Tuple[bool, List[str]]:
    errors = []
    
    if 'text' in sample:
        text = sample['text']
        
        for section in REQUIRED_SECTIONS:
            pattern = f'<\\|{section}\\|>.*?<\\|/{section}\\|>'
            if not re.search(pattern, text, re.DOTALL):
                errors.append(f"Missing required section: <|{section}|>")
        
        for tag in REQUIRED_OUTPUT_TAGS:
            pattern = f'<{tag}>.*?</{tag}>'
            if not re.search(pattern, text, re.DOTALL):
                errors.append(f"Missing required output tag: <{tag}>")
        
        decision_match = re.search(r'<decision>(.*?)</decision>', text, re.DOTALL)
        if decision_match:
            decision = decision_match.group(1).strip()
            if decision not in ['SPEAK', 'SILENT']:
                errors.append(f"Invalid decision value: {decision} (must be SPEAK or SILENT)")
        
        confidence_match = re.search(r'<confidence>(.*?)</confidence>', text, re.DOTALL)
        if confidence_match:
            confidence = confidence_match.group(1).strip()
            if confidence not in ['high', 'medium', 'low']:
                errors.append(f"Invalid confidence value: {confidence} (must be high, medium, or low)")
    
    elif 'decision' in sample and ('context' in sample or 'context_turns' in sample):
        if 'decision' not in sample:
            errors.append("Missing 'decision' field")
        elif sample['decision'] not in ['SPEAK', 'SILENT']:
            errors.append(f"Invalid decision: {sample['decision']}")
        
        if 'confidence' in sample and sample['confidence'] not in ['high', 'medium', 'low']:
            errors.append(f"Invalid confidence: {sample.get('confidence')}")
        
        if 'context' not in sample and 'context_turns' not in sample:
            errors.append("Missing 'context' or 'context_turns' field")
        
        if 'current_turn' not in sample:
            errors.append("Missing 'current_turn' field")
    
    else:
        errors.append("Sample format not recognized (missing 'text' field or intermediate format fields)")
    
    return len(errors) == 0, errors


def extract_category(sample: Dict) -> str:
    if 'category' in sample:
        return sample['category']
    elif 'text' in sample:
        return 'UNKNOWN'
    else:
        return 'UNKNOWN'


def extract_decision(sample: Dict) -> str:
    if 'decision' in sample:
        return sample['decision']
    elif 'text' in sample:
        match = re.search(r'<decision>(.*?)</decision>', sample['text'], re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return 'UNKNOWN'


def generate_statistics(samples: List[Dict]) -> Dict:
    stats = {
        'total_samples': len(samples),
        'decisions': Counter(),
        'categories': Counter(),
        'confidence_levels': Counter(),
        'validation_results': {
            'valid': 0,
            'invalid': 0,
            'errors': defaultdict(int)
        }
    }
    
    for sample in samples:
        is_valid, errors = validate_training_format(sample)
        
        if is_valid:
            stats['validation_results']['valid'] += 1
            
            # Extract decision
            decision = extract_decision(sample)
            stats['decisions'][decision] += 1
            
            # Extract category
            category = extract_category(sample)
            stats['categories'][category] += 1
            
            # Extract confidence
            if 'confidence' in sample:
                stats['confidence_levels'][sample['confidence']] += 1
            elif 'text' in sample:
                match = re.search(r'<confidence>(.*?)</confidence>', sample['text'], re.DOTALL)
                if match:
                    stats['confidence_levels'][match.group(1).strip()] += 1
        else:
            stats['validation_results']['invalid'] += 1
            for error in errors:
                stats['validation_results']['errors'][error] += 1
    
    return stats


def print_statistics(stats: Dict, dataset_name: str = "Dataset"):
    print("\n" + "="*70)
    print(f"{dataset_name.upper()} STATISTICS")
    print("="*70)
    
    print(f"\nTotal samples: {stats['total_samples']:,}")
    
    # Validation results
    valid = stats['validation_results']['valid']
    invalid = stats['validation_results']['invalid']
    valid_pct = (valid / stats['total_samples'] * 100) if stats['total_samples'] > 0 else 0
    
    print(f"\nValidation:")
    print(f"  Valid:   {valid:,} ({valid_pct:.1f}%)")
    print(f"  Invalid: {invalid:,} ({100-valid_pct:.1f}%)")
    
    if stats['validation_results']['errors']:
        print(f"\n  Common errors:")
        for error, count in sorted(stats['validation_results']['errors'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    - {error}: {count}")
    
    # Decision distribution
    if stats['decisions']:
        print(f"\nDecision distribution:")
        total_decisions = sum(stats['decisions'].values())
        for decision, count in stats['decisions'].most_common():
            pct = (count / total_decisions * 100) if total_decisions > 0 else 0
            print(f"  {decision}: {count:,} ({pct:.1f}%)")
    
    # Category distribution
    if stats['categories']:
        print(f"\nCategory distribution:")
        total_categories = sum(stats['categories'].values())
        
        # Group by SPEAK vs SILENT
        speak_cats = {k: v for k, v in stats['categories'].items() if k in SPEAK_CATEGORIES}
        silent_cats = {k: v for k, v in stats['categories'].items() if k in SILENT_CATEGORIES}
        
        if speak_cats:
            print(f"  SPEAK categories:")
            for cat in SPEAK_CATEGORIES:
                count = speak_cats.get(cat, 0)
                pct = (count / total_categories * 100) if total_categories > 0 else 0
                print(f"    {cat}: {count:,} ({pct:.1f}%)")
        
        if silent_cats:
            print(f"  SILENT categories:")
            for cat in SILENT_CATEGORIES:
                count = silent_cats.get(cat, 0)
                pct = (count / total_categories * 100) if total_categories > 0 else 0
                print(f"    {cat}: {count:,} ({pct:.1f}%)")
        
        unknown = stats['categories'].get('UNKNOWN', 0)
        if unknown > 0:
            pct = (unknown / total_categories * 100) if total_categories > 0 else 0
            print(f"    UNKNOWN: {unknown:,} ({pct:.1f}%)")
    
    # Confidence distribution
    if stats['confidence_levels']:
        print(f"\nConfidence distribution:")
        total_conf = sum(stats['confidence_levels'].values())
        for conf, count in stats['confidence_levels'].most_common():
            pct = (count / total_conf * 100) if total_conf > 0 else 0
            print(f"  {conf}: {count:,} ({pct:.1f}%)")
    
    print("\n" + "="*70)



def main():
    """Main execution function"""
    print("="*70)
    print("VALIDATING DATA FORMAT")
    print("="*70)
    
    # Check for data files
    train_file = DATA_DIR / 'train' / 'train_samples.jsonl'
    val_file = DATA_DIR / 'val' / 'val_samples.jsonl'
    test_file = DATA_DIR / 'test' / 'test_samples.jsonl'
    
    files_to_check = [
        ('Train', train_file),
        ('Val', val_file),
        ('Test', test_file)
    ]
    
    all_stats = {}
    
    for dataset_name, file_path in files_to_check:
        if not file_path.exists():
            print(f"\nWarning: {file_path} not found. Skipping {dataset_name} set.")
            continue
        
        print(f"\nValidating {dataset_name} set: {file_path}")
        samples = load_samples(file_path)
        
        if not samples:
            print(f"  No samples found in {file_path}")
            continue
        
        print(f"  Loaded {len(samples):,} samples")
        
        # Generate statistics
        stats = generate_statistics(samples)
        all_stats[dataset_name] = stats
        
        # Print statistics
        print_statistics(stats, dataset_name)
    
    # Summary
    if all_stats:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        total_samples = sum(s['total_samples'] for s in all_stats.values())
        total_valid = sum(s['validation_results']['valid'] for s in all_stats.values())
        total_invalid = sum(s['validation_results']['invalid'] for s in all_stats.values())
        
        print(f"\nTotal samples across all sets: {total_samples:,}")
        print(f"  Valid:   {total_valid:,} ({total_valid/total_samples*100:.1f}%)")
        print(f"  Invalid: {total_invalid:,} ({total_invalid/total_samples*100:.1f}%)")
        
        # Overall decision distribution
        all_decisions = Counter()
        for stats in all_stats.values():
            all_decisions.update(stats['decisions'])
        
        if all_decisions:
            print(f"\nOverall decision distribution:")
            total = sum(all_decisions.values())
            for decision, count in all_decisions.most_common():
                print(f"  {decision}: {count:,} ({count/total*100:.1f}%)")
        
        print("\n" + "="*70)
        print("VALIDATION COMPLETE")
        print("="*70)
    else:
        print("\nNo data files found. Please run download_data.py first.")


if __name__ == '__main__':
    main()
