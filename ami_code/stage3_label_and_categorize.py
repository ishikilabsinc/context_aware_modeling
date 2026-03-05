#!/usr/bin/env python3
"""
Stage 3+4: Label SPEAK/SILENT and Assign Categories (Merged)

Goal:
    For each decision point:
    1. Assign label: SPEAK or SILENT based on ground truth (target_spoke_next)
    2. Assign confidence level (high/medium/low)
    3. Assign category (4 total: SPEAK_explicit, SPEAK_implicit, SILENT_no_ref, SILENT_ref)

Input:
    - json_dumps/stage2_decision_points.json from Stage 2

Output:
    - json_dumps/stage3_categorized_samples.json - labeled and categorized samples for Stage 4

Labeling Logic:
    Uses ground truth (target_spoke_next) directly:
        - SPEAK if target_spoke_next = true (target actually spoke)
        - SILENT if target_spoke_next = false (target did not speak)
    
    All samples get 'high' confidence since they're based on ground truth.
    Context (addressing, addressees) is used for reasoning but not decision.

Source Metadata:
    Each decision point includes metadata about its origin:
        - dp['source']: 'explicit' (from Stage 1) or 'gemini_inferred' (from Stage 1b)
        - dp['is_explicit']: True if from explicit annotations, False if AI-inferred
        - dp['inference_confidence']: 0-10 scale (10 = ground truth, lower = AI confidence)
    
    You can filter or analyze by source:
        explicit_samples = [dp for dp in decision_points if dp.get('is_explicit', True)]
        ai_samples = [dp for dp in decision_points if not dp.get('is_explicit', True)]
"""

import json
import re
from pathlib import Path
from typing import Dict, Tuple
from collections import Counter

from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

JSON_DUMPS_DIR = Path('json_dumps')
INPUT_FILE = JSON_DUMPS_DIR / 'stage2_decision_points.json'
OUTPUT_FILE = JSON_DUMPS_DIR / 'stage3_categorized_samples.json'

# Category descriptions (4 categories)
CATEGORY_NAMES = {
    'SPEAK_explicit': 'Target was directly addressed',
    'SPEAK_implicit': 'Target spoke but not from direct address',
    'SILENT_no_ref': 'No reference to target',
    'SILENT_ref': 'Target mentioned or in context, but did not speak',
}

# ============================================================================
# LABELING FUNCTION (Stage 3)
# ============================================================================

def label_decision_point(dp: Dict) -> Tuple[str, str, str]:
    """
    Label a decision point as SPEAK or SILENT based on ground truth
    
    Uses target_spoke_next (ground truth) as primary signal for bucketing.
    Confidence is based on ground truth alignment and addressing context.
    
    Args:
        dp: Decision point from Stage 2
        
    Returns:
        (decision, confidence, reason) tuple
    """
    target_is_addressed = dp['target_is_addressed']
    target_spoke_next = dp['target_spoke_next']  # Ground truth
    target_speaker = dp['target_speaker']
    current_speaker = dp['current_turn']['speaker']
    
    # Use ground truth directly for decision
    decision = 'SPEAK' if target_spoke_next else 'SILENT'
    
    # Assign confidence/bucket based on ground truth alignment and context
    if target_spoke_next:
        # Ground truth: target spoke
        if target_is_addressed:
            # Explicitly addressed AND spoke → highest quality bucket
            confidence = 'high'
            reason = f"{target_speaker} was explicitly addressed by {current_speaker} and spoke (ground truth)"
        else:
            # Spoke but not explicitly addressed → high quality (ground truth positive)
            confidence = 'high'
            reason = f"{target_speaker} spoke next in conversation (ground truth)"
    else:
        # Ground truth: target did not speak
        if dp['addressees_in_current']:
            # Someone else was addressed → high quality negative example
            confidence = 'high'
            addressees_str = ', '.join(dp['addressees_in_current'])
            reason = f"{current_speaker} addressed {addressees_str}, not {target_speaker} (ground truth: did not speak)"
        else:
            # No explicit addressing, target didn't speak → high quality negative
            confidence = 'high'
            reason = f"{target_speaker} did not speak next (ground truth)"
    
    return decision, confidence, reason


# ============================================================================
# CATEGORIZATION HELPER FUNCTIONS (Stage 4)
# ============================================================================

def check_target_mentioned(current_text: str, target_speaker: str) -> bool:
    """
    Check if target speaker is mentioned in the current turn text
    
    Looks for patterns like "Speaker A", "A said", "tell A", etc.
    """
    text_lower = current_text.lower()
    target_lower = target_speaker.lower()
    
    # Look for patterns where speaker is mentioned
    patterns = [
        f"\\b{target_lower}\\b",
        f"speaker {target_lower}",
        f"{target_lower} said",
        f"tell {target_lower}",
        f"ask {target_lower}"
    ]
    
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False


def check_incomplete_sentence(text: str) -> bool:
    """
    Check if sentence appears incomplete
    
    Indicators: trailing ellipsis, hesitation markers (um, uh, hmm), very short
    """
    text = text.strip()
    
    # Check for incomplete markers
    incomplete_markers = ['...', 'um', 'uh', 'hmm']
    
    # Ends with ellipsis
    if text.endswith('...'):
        return True
    
    # Very short utterances (likely incomplete)
    if len(text.split()) <= 2:
        return True
    
    # Ends with hesitation markers
    last_words = text.lower().split()[-2:]
    for marker in incomplete_markers:
        if marker in last_words:
            return True
    
    return False


def was_target_in_recent_exchange(context_turns: list, target_speaker: str, window: int = 3) -> bool:
    """
    Check if target was active in recent turns
    
    Args:
        context_turns: List of previous turns
        target_speaker: Speaker to check for
        window: Number of recent turns to check
        
    Returns:
        True if target spoke in last 'window' turns
    """
    if not context_turns:
        return False
    
    recent_speakers = [turn['speaker'] for turn in context_turns[-window:]]
    return target_speaker in recent_speakers


# ============================================================================
# CATEGORY ASSIGNMENT (Stage 4)
# ============================================================================

def assign_category(sample: Dict) -> str:
    """
    Assign category to a labeled sample using decision tree logic.
    Returns one of: SPEAK_explicit, SPEAK_implicit, SILENT_no_ref, SILENT_ref
    """
    decision = sample['decision']
    target_speaker = sample['target_speaker']
    target_is_addressed = sample['target_is_addressed']
    current_text = sample['current_turn']['text']
    context_turns = sample['context_turns']
    
    # SPEAK categories
    if decision == 'SPEAK':
        if target_is_addressed:
            return 'SPEAK_explicit'
        # I2 or I3 -> SPEAK_implicit
        return 'SPEAK_implicit'
    
    # SILENT categories: S2 (mentioned), S3 (recent exchange) -> SILENT_ref; else SILENT_no_ref
    else:
        target_mentioned = check_target_mentioned(current_text, target_speaker)
        if target_mentioned and not target_is_addressed:
            return 'SILENT_ref'
        if was_target_in_recent_exchange(context_turns, target_speaker, window=2):
            return 'SILENT_ref'
        return 'SILENT_no_ref'


# ============================================================================
# STATISTICS AND EXAMPLES
# ============================================================================

def print_statistics(categorized_samples: list):
    """Print statistics about categorized samples"""
    print("\n" + "="*70)
    print("STAGE 3+4 STATISTICS")
    print("="*70)
    
    total_samples = len(categorized_samples)
    
    # Source distribution (if available)
    if total_samples > 0 and 'source' in categorized_samples[0]:
        sources = [s.get('source', 'unknown') for s in categorized_samples]
        source_counts = Counter(sources)
        print(f"\nSource distribution:")
        for source, count in source_counts.most_common():
            print(f"  {source}: {count:,} ({count/total_samples*100:.1f}%)")
        
        explicit_count = sum(1 for s in categorized_samples if s.get('is_explicit', True))
        inferred_count = total_samples - explicit_count
        if inferred_count > 0:
            print(f"\nAnnotation type:")
            print(f"  Explicit annotations: {explicit_count:,} ({explicit_count/total_samples*100:.1f}%)")
            print(f"  AI-inferred: {inferred_count:,} ({inferred_count/total_samples*100:.1f}%)")
    
    # Decision counts
    decisions = [s['decision'] for s in categorized_samples]
    decision_counts = Counter(decisions)
    print(f"\nDecisions:")
    for decision, count in decision_counts.most_common():
        print(f"  {decision}: {count:,} ({count/total_samples*100:.1f}%)")
    
    # Confidence distribution
    confidences = [s['confidence'] for s in categorized_samples]
    confidence_counts = Counter(confidences)
    print(f"\nConfidence levels:")
    for conf, count in confidence_counts.most_common():
        print(f"  {conf}: {count:,} ({count/total_samples*100:.1f}%)")
    
    # Category distribution
    categories = [s['category'] for s in categorized_samples]
    category_counts = Counter(categories)
    
    print(f"\nCategory distribution:")
    for cat in ['SPEAK_explicit', 'SPEAK_implicit', 'SILENT_no_ref', 'SILENT_ref']:
        count = category_counts.get(cat, 0)
        if count > 0:
            print(f"  {cat} ({CATEGORY_NAMES[cat]}): {count:,} ({count/total_samples*100:.1f}%)")
    
    speak_count = sum(category_counts.get(cat, 0) for cat in ['SPEAK_explicit', 'SPEAK_implicit'])
    silent_count = sum(category_counts.get(cat, 0) for cat in ['SILENT_no_ref', 'SILENT_ref'])
    print(f"\nOverall: SPEAK {speak_count:,}, SILENT {silent_count:,}")
    
    # Category × Confidence
    print(f"\nCategory × Confidence:")
    for cat in sorted(category_counts.keys()):
        cat_samples = [s for s in categorized_samples if s['category'] == cat]
        conf_counts = Counter(s['confidence'] for s in cat_samples)
        print(f"  {cat}: high={conf_counts.get('high', 0)}, medium={conf_counts.get('medium', 0)}, low={conf_counts.get('low', 0)}")
    
    # Accuracy check: Does addressing predict speaking?
    addressed_and_spoke = sum(1 for s in categorized_samples 
                              if s['target_is_addressed'] and s['target_spoke_next'])
    addressed_total = sum(1 for s in categorized_samples if s['target_is_addressed'])
    
    print(f"\nGround truth validation:")
    if addressed_total > 0:
        print(f"  When addressed, target spoke: {addressed_and_spoke}/{addressed_total} ({addressed_and_spoke/addressed_total*100:.1f}%)")
    else:
        print(f"  No explicitly addressed samples found")
    
    print(f"\n" + "="*70)


def print_examples(categorized_samples: list):
    """Print examples from each category"""
    print("\n" + "="*70)
    print("EXAMPLES BY CATEGORY")
    print("="*70 + "\n")
    
    categories = ['SPEAK_explicit', 'SPEAK_implicit', 'SILENT_no_ref', 'SILENT_ref']
    
    for cat in categories:
        examples = [s for s in categorized_samples if s['category'] == cat][:2]
        
        if examples:
            print(f"\n{cat}: {CATEGORY_NAMES[cat]}")
            print("-" * 70)
            
            for i, sample in enumerate(examples, 1):
                print(f"{i}. Target: {sample['target_speaker']}")
                print(f"   Current: [{sample['current_turn']['speaker']}]: {sample['current_turn']['text'][:60]}...")
                print(f"   Decision: {sample['decision']}")
                print(f"   Confidence: {sample['confidence']}")
                print(f"   Reason: {sample['reason']}")
                print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    # Ensure json_dumps directory exists
    JSON_DUMPS_DIR.mkdir(exist_ok=True)
    
    # Load decision points from Stage 2
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print(f"Please ensure Stage 2 has been run and output exists in json_dumps/")
        return
    
    with open(INPUT_FILE, 'r') as f:
        decision_points = json.load(f)
    
    print(f"Loaded {len(decision_points):,} decision points from Stage 2")
    print("Labeling and categorizing decision points...")
    print("="*70)
    
    # Label and categorize all decision points in one pass
    categorized_samples = []
    
    for dp in tqdm(decision_points, desc="Label & categorize", unit=" sample"):
        # Stage 3: Label (SPEAK/SILENT) and assign confidence/reason
        decision, confidence, reason = label_decision_point(dp)
        
        # Create labeled sample
        labeled_sample = dp.copy()
        labeled_sample['decision'] = decision
        labeled_sample['confidence'] = confidence
        labeled_sample['reason'] = reason
        
        # Stage 4: Assign category
        category = assign_category(labeled_sample)
        labeled_sample['category'] = category
        
        categorized_samples.append(labeled_sample)
    
    print("="*70)
    print(f"\nLabeled and categorized {len(categorized_samples):,} samples")
    
    # Print examples and statistics
    print_examples(categorized_samples)
    print_statistics(categorized_samples)
    
    # Save categorized samples for next stage
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(categorized_samples, f, indent=2)
    
    print(f"\n✓ Saved {len(categorized_samples):,} categorized samples to {OUTPUT_FILE}")
    print(f"  Output location: {OUTPUT_FILE.absolute()}")
    print(f"\nReady for Stage 4: Filter and Balance")


if __name__ == '__main__':
    main()
