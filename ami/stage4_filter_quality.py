#!/usr/bin/env python3
"""
Stage 4: Filter and Balance High-Quality Samples

Goal:
    Filter stage4 output for high-quality, balanced training data:
    - Remove filler words and low-quality utterances
    - Balance SPEAK/SILENT ratio (50/50)
    - Balance subcategories (I1-I3, S1-S5) to avoid skew
    - Sort by confidence and select best samples

Input:
    - json_dumps/stage4_categorized_samples.json from Stage 3+4

Output:
    - stage45_filtered_samples.json - high-quality balanced samples for Stage 5
"""

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Set
from collections import Counter, defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

JSON_DUMPS_DIR = Path('json_dumps')
INPUT_FILE = JSON_DUMPS_DIR / 'stage3_categorized_samples.json'

# Directory to dump individual filtered samples as separate JSON files
JSON_RUN_DIR = Path('json_run')

# Directory and filename for aggregated JSONL output
OUTPUT_DIR = 'data_final'
OUTPUT_FILE = 'stage4_filtered_samples.jsonl'  # JSONL format

# Target ratio for SPEAK/SILENT
SPEAK_RATIO = 0.5  # 50% SPEAK, 50% SILENT

# Filler words/patterns to remove (case-insensitive)
FILLER_PATTERNS = [
    r'^\s*mm+\s*[-.]?\s*hm+\s*\.?\s*$',  # Mm-hmm, Mmhmm, etc.
    r'^\s*uh+\s*[-.]?\s*huh+\s*\.?\s*$',  # Uh-huh
    r'^\s*um+\s*\.?\s*$',                  # Um
    r'^\s*uh+\s*\.?\s*$',                  # Uh
    r'^\s*er+\s*\.?\s*$',                  # Er
    r'^\s*ah+\s*\.?\s*$',                  # Ah
    r'^\s*oh+\s*\.?\s*$',                  # Oh
    r'^\s*hm+\s*\.?\s*$',                  # Hmm
    r'^\s*\.\s*$',                         # Just punctuation
]

# Minimum text length (characters) after cleaning
MIN_TEXT_LENGTH = 3

# Maximum samples to keep (None = keep all that pass filters)
MAX_SAMPLES = None  # Can be set to limit dataset size, e.g., 1000000

# Subcategory balance threshold (max ratio any category can be)
# E.g., 0.4 means no category can be more than 40% of its class
MAX_CATEGORY_RATIO = 0.35

# ============================================================================
# CONFIDENCE SCORING
# ============================================================================

def confidence_to_score(confidence_str: str) -> int:
    """
    Convert confidence string to numeric score for sorting
    
    Args:
        confidence_str: "high", "medium", or "low"
        
    Returns:
        Numeric score (3=high, 2=medium, 1=low)
    """
    confidence_map = {
        'high': 3,
        'medium': 2,
        'low': 1
    }
    return confidence_map.get(confidence_str.lower(), 0)


# ============================================================================
# QUALITY FILTERS
# ============================================================================

def is_filler_text(text: str) -> bool:
    """
    Check if text is a filler word/phrase that should be removed
    
    Args:
        text: The utterance text
        
    Returns:
        True if text is filler and should be removed
    """
    text_clean = text.strip().lower()
    
    # Check against filler patterns
    for pattern in FILLER_PATTERNS:
        if re.match(pattern, text_clean, re.IGNORECASE):
            return True
    
    return False


def is_low_quality(sample: Dict) -> bool:
    """
    Check if sample is low quality and should be filtered out
    
    Criteria:
    - Filler words
    - Too short
    - Empty or whitespace only
    - Repetitive punctuation
    
    Args:
        sample: Sample dict with 'current_turn' containing 'text'
        
    Returns:
        True if sample should be removed
    """
    text = sample['current_turn']['text']
    
    # Empty or whitespace only
    if not text or not text.strip():
        return True
    
    # Filler words
    if is_filler_text(text):
        return True
    
    # Too short (after removing punctuation and whitespace)
    text_alphanum = re.sub(r'[^\w\s]', '', text)
    if len(text_alphanum.strip()) < MIN_TEXT_LENGTH:
        return True
    
    # Mostly punctuation (>50% non-alphanumeric)
    if len(text_alphanum) < len(text) * 0.5:
        return True
    
    return False


# ============================================================================
# DEDUPLICATION
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for similarity comparison
    
    Args:
        text: Original text
        
    Returns:
        Normalized text (lowercase, no extra whitespace, no punctuation)
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


def create_sample_signature(sample: Dict, context_window: int = 3) -> str:
    """
    Create unique signature for a sample based on text + context + label
    
    Args:
        sample: Sample dict
        context_window: Number of recent context turns to include (default: 3)
        
    Returns:
        Unique signature string
    """
    # Normalize current turn text
    current_text = normalize_text(sample['current_turn']['text'])
    
    # Normalize recent context (last N turns)
    context_turns = sample.get('context_turns', [])
    recent_context = context_turns[-context_window:] if len(context_turns) > context_window else context_turns
    
    context_signature = []
    for turn in recent_context:
        speaker = turn['speaker']
        text = normalize_text(turn['text'])
        context_signature.append(f"{speaker}:{text}")
    
    context_str = "|".join(context_signature)
    
    # Include decision label
    decision = sample['decision']
    
    # Create composite signature
    signature = f"{current_text}:::{context_str}:::{decision}"
    
    return signature


def deduplicate_samples(samples: List[Dict]) -> List[Dict]:
    """
    Remove duplicate samples based on text + context + label
    
    Strategy:
    1. Create signature from: current_text + recent_context + decision_label
    2. Keep only one sample per unique signature
    3. When duplicates found, keep sample with higher confidence
    
    This preserves:
    - Same text with different context
    - Same text with different labels (SPEAK vs SILENT)
    - Only removes truly identical samples
    
    Args:
        samples: List of samples
        
    Returns:
        Deduplicated list of samples
    """
    if not samples:
        return []
    
    print(f"\n  Starting samples: {len(samples):,}")
    print(f"  Deduplication strategy: Text + Context (last 3 turns) + Label")
    
    # Sort by confidence (high to low) to keep better samples
    samples_sorted = sorted(
        samples, 
        key=lambda s: confidence_to_score(s['confidence']), 
        reverse=True
    )
    
    # Track seen signatures
    seen_signatures: Set[str] = set()
    
    deduplicated = []
    duplicates_removed = 0
    
    for sample in samples_sorted:
        # Create unique signature
        signature = create_sample_signature(sample)
        
        # Check if we've seen this exact combination before
        if signature in seen_signatures:
            duplicates_removed += 1
            continue
        
        # Keep this sample
        seen_signatures.add(signature)
        deduplicated.append(sample)
    
    print(f"  Removed {duplicates_removed:,} duplicates (same text + context + label)")
    print(f"  Remaining samples: {len(deduplicated):,}")
    
    return deduplicated


# ============================================================================
# BALANCING
# ============================================================================

def balance_by_category(samples: List[Dict], 
                       target_count: int,
                       max_category_ratio: float = MAX_CATEGORY_RATIO) -> List[Dict]:
    """
    Balance samples by category (I1-I3 or S1-S5) to avoid skew
    
    Strategy:
    1. Sort by confidence score
    2. Calculate target per category (ensure no category dominates)
    3. Take top N from each category proportionally
    
    Args:
        samples: List of samples (all SPEAK or all SILENT)
        target_count: Total number of samples to keep
        max_category_ratio: Max fraction any single category can be
        
    Returns:
        Balanced list of samples
    """
    if not samples:
        return []
    
    # Group by category
    by_category = defaultdict(list)
    for sample in samples:
        category = sample['category']
        by_category[category].append(sample)
    
    num_categories = len(by_category)
    
    # Calculate target per category
    # Ensure no category exceeds max_category_ratio
    max_per_category = int(target_count * max_category_ratio)
    min_per_category = max(1, target_count // (num_categories * 2))  # At least some from each
    
    # Distribute target_count across categories proportionally
    # but cap at max_per_category
    category_counts = {cat: len(samples) for cat, samples in by_category.items()}
    total_available = sum(category_counts.values())
    
    targets = {}
    for cat, available in category_counts.items():
        # Proportional allocation
        proportion = available / total_available
        target = int(target_count * proportion)
        
        # Apply constraints
        target = max(min_per_category, min(target, max_per_category))
        target = min(target, available)  # Can't take more than available
        
        targets[cat] = target
    
    # Adjust to hit exact target_count
    current_total = sum(targets.values())
    if current_total < target_count:
        # Add more to categories with room
        for cat in sorted(targets.keys(), key=lambda c: category_counts[c], reverse=True):
            if current_total >= target_count:
                break
            available = category_counts[cat]
            current = targets[cat]
            can_add = min(available - current, target_count - current_total)
            targets[cat] += can_add
            current_total += can_add
    elif current_total > target_count:
        # Remove from largest categories
        for cat in sorted(targets.keys(), key=lambda c: targets[c], reverse=True):
            if current_total <= target_count:
                break
            can_remove = min(targets[cat] - min_per_category, current_total - target_count)
            targets[cat] -= can_remove
            current_total -= can_remove
    
    # Select top samples from each category
    balanced = []
    for cat, target in targets.items():
        cat_samples = by_category[cat]
        
        # Sort by confidence (high to low)
        cat_samples.sort(key=lambda s: confidence_to_score(s['confidence']), reverse=True)
        
        # Take top N
        balanced.extend(cat_samples[:target])
    
    return balanced


def balance_speak_silent(samples: List[Dict], 
                        speak_ratio: float = SPEAK_RATIO) -> List[Dict]:
    """
    Balance SPEAK and SILENT samples
    
    Args:
        samples: All samples
        speak_ratio: Target ratio for SPEAK samples (0.5 = 50/50)
        
    Returns:
        Balanced list of samples
    """
    # Separate by decision
    speak_samples = [s for s in samples if s['decision'] == 'SPEAK']
    silent_samples = [s for s in samples if s['decision'] == 'SILENT']
    
    print(f"  Before balancing: {len(speak_samples):,} SPEAK, {len(silent_samples):,} SILENT")
    
    # Determine target counts
    total_available = len(speak_samples) + len(silent_samples)
    
    if MAX_SAMPLES and total_available > MAX_SAMPLES:
        total_target = MAX_SAMPLES
    else:
        # Balance to the minority class to avoid throwing away too much data
        minority_count = min(len(speak_samples), len(silent_samples))
        total_target = minority_count * 2  # 50/50 split
    
    speak_target = int(total_target * speak_ratio)
    silent_target = total_target - speak_target
    
    # Cap at available
    speak_target = min(speak_target, len(speak_samples))
    silent_target = min(silent_target, len(silent_samples))
    
    print(f"  Target: {speak_target:,} SPEAK, {silent_target:,} SILENT")
    
    # Balance each class by subcategories
    print(f"\n  Balancing SPEAK categories (I1-I3)...")
    balanced_speak = balance_by_category(speak_samples, speak_target)
    
    print(f"  Balancing SILENT categories (S1-S5)...")
    balanced_silent = balance_by_category(silent_samples, silent_target)
    
    # Combine
    balanced = balanced_speak + balanced_silent
    
    print(f"\n  After balancing: {len(balanced_speak):,} SPEAK, {len(balanced_silent):,} SILENT")
    print(f"  Total samples: {len(balanced):,}")
    
    return balanced


# ============================================================================
# STATISTICS
# ============================================================================

def print_statistics(samples: List[Dict], title: str = "Dataset Statistics"):
    """Print detailed statistics about the dataset"""
    print("\n" + "="*70)
    print(title)
    print("="*70)
    
    print(f"\nTotal samples: {len(samples):,}")
    
    # Decision distribution
    decisions = Counter(s['decision'] for s in samples)
    print(f"\nDecision distribution:")
    for decision, count in decisions.most_common():
        pct = count / len(samples) * 100
        print(f"  {decision}: {count:,} ({pct:.1f}%)")
    
    # Category distribution
    categories = Counter(s['category'] for s in samples)
    print(f"\nCategory distribution:")
    
    # SPEAK categories (I1-I3)
    speak_cats = {k: v for k, v in categories.items() if k.startswith('I')}
    if speak_cats:
        speak_total = sum(speak_cats.values())
        print(f"\n  SPEAK categories ({speak_total:,} total):")
        for cat in sorted(speak_cats.keys()):
            count = speak_cats[cat]
            pct = count / speak_total * 100
            print(f"    {cat}: {count:,} ({pct:.1f}%)")
    
    # SILENT categories (S1-S5)
    silent_cats = {k: v for k, v in categories.items() if k.startswith('S')}
    if silent_cats:
        silent_total = sum(silent_cats.values())
        print(f"\n  SILENT categories ({silent_total:,} total):")
        for cat in sorted(silent_cats.keys()):
            count = silent_cats[cat]
            pct = count / silent_total * 100
            print(f"    {cat}: {count:,} ({pct:.1f}%)")
    
    # Confidence distribution
    confidences = Counter(s['confidence'] for s in samples)
    print(f"\nConfidence distribution:")
    for conf in ['high', 'medium', 'low']:
        count = confidences.get(conf, 0)
        pct = count / len(samples) * 100 if len(samples) > 0 else 0
        print(f"  {conf}: {count:,} ({pct:.1f}%)")
    
    print("="*70)


def print_examples(samples: List[Dict], num_examples: int = 5):
    """Print example samples"""
    print("\n" + "="*70)
    print("EXAMPLE FILTERED SAMPLES")
    print("="*70 + "\n")
    
    for i, sample in enumerate(samples[:num_examples]):
        print(f"Sample {i+1}:")
        print(f"  Decision: {sample['decision']}")
        print(f"  Category: {sample['category']}")
        print(f"  Confidence: {sample['confidence']}")
        print(f"  Current turn: [{sample['current_turn']['speaker']}] {sample['current_turn']['text'][:100]}...")
        print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("="*70)
    print("STAGE 4: FILTER AND BALANCE HIGH-QUALITY SAMPLES")
    print("="*70)
    
    # Ensure json_dumps directory exists
    JSON_DUMPS_DIR.mkdir(exist_ok=True)
    
    # Load samples
    print(f"\nLoading samples from {INPUT_FILE}...")
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print(f"Please ensure Stage 3+4 has been run and output exists in json_dumps/")
        return
    
    with open(INPUT_FILE, 'r') as f:
        all_samples = json.load(f)
    
    print(f"✓ Loaded {len(all_samples):,} samples")
    
    # Print initial statistics
    print_statistics(all_samples, "BEFORE FILTERING")
    
    # Filter low quality samples
    print("\n" + "="*70)
    print("FILTERING LOW-QUALITY SAMPLES")
    print("="*70)
    
    filtered_samples = []
    filtered_out_count = 0
    
    for sample in all_samples:
        if not is_low_quality(sample):
            filtered_samples.append(sample)
        else:
            filtered_out_count += 1
    
    print(f"\n✓ Filtered out {filtered_out_count:,} low-quality samples")
    print(f"✓ Remaining: {len(filtered_samples):,} samples")
    
    # Deduplicate samples
    print("\n" + "="*70)
    print("REMOVING EXACT DUPLICATES")
    print("="*70)
    
    deduplicated_samples = deduplicate_samples(filtered_samples)
    
    # Balance SPEAK/SILENT and subcategories
    print("\n" + "="*70)
    print("BALANCING SPEAK/SILENT AND SUBCATEGORIES")
    print("="*70)
    
    balanced_samples = balance_speak_silent(deduplicated_samples, speak_ratio=SPEAK_RATIO)
    
    # Print final statistics
    print_statistics(balanced_samples, "AFTER FILTERING, DEDUPLICATION, AND BALANCING")
    
    # Print examples
    print_examples(balanced_samples)
    
    # Save output
    print("\n" + "="*70)
    print("SAVING FILTERED DATA")
    print("="*70)
    
    # Create output directories
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    print(f"\n✓ Created output directory: {OUTPUT_DIR}/")

    JSON_RUN_DIR.mkdir(exist_ok=True)
    print(f"✓ Created per-sample output directory: {JSON_RUN_DIR}/")
    
    # Save as JSONL (one JSON object per line)
    output_path = output_dir / OUTPUT_FILE
    print(f"\nSaving filtered samples to {output_path}...")
    
    with open(output_path, 'w') as f:
        for sample in balanced_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"✓ Saved {len(balanced_samples):,} samples in JSONL format")

    # Also save each filtered sample as an individual JSON file in JSON_RUN_DIR
    print(f"\nSaving individual filtered samples to {JSON_RUN_DIR}/ ...")
    for idx, sample in enumerate(balanced_samples, start=1):
        sample_path = JSON_RUN_DIR / f"{idx}.json"
        with open(sample_path, 'w') as sf:
            json.dump(sample, sf, indent=2)
    print(f"✓ Saved {len(balanced_samples):,} individual sample files")
    
    # Save summary statistics
    summary_path = output_dir / 'filtering_summary.json'
    summary = {
        'total_samples': len(balanced_samples),
        'speak_samples': len([s for s in balanced_samples if s['decision'] == 'SPEAK']),
        'silent_samples': len([s for s in balanced_samples if s['decision'] == 'SILENT']),
        'category_distribution': dict(Counter(s['category'] for s in balanced_samples)),
        'confidence_distribution': dict(Counter(s['confidence'] for s in balanced_samples)),
        'filtering_config': {
            'speak_ratio': SPEAK_RATIO,
            'max_category_ratio': MAX_CATEGORY_RATIO,
            'min_text_length': MIN_TEXT_LENGTH,
            'deduplication': 'text + context (3 turns) + label'
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved filtering summary to {summary_path}")
    
    # Print output location
    print(f"\nOutput files:")
    print(f"  {output_path} - Filtered samples (JSONL)")
    print(f"  {summary_path} - Summary statistics (JSON)")
    
    print("\n" + "="*70)
    print("STAGE 4 COMPLETE")
    print("="*70)
    print(f"\nNext step: Use filtered data from {OUTPUT_DIR}/ folder")
    print(f"  Stage 5 will read: {output_path}")


if __name__ == '__main__':
    main()
