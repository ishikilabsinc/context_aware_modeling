#!/usr/bin/env python3
"""
Stage 6: Statistics & Validation

Goal:
    Generate comprehensive statistics and validation samples for the training dataset.
    
    Analyze:
    - Dataset size and composition
    - SPEAK/SILENT ratio
    - Category distribution
    - Confidence distribution
    - Sample quality
    - Duplicates

Input:
    - training_data_intermediate.jsonl
    - training_data_formatted.jsonl

Output:
    - data_statistics.json - comprehensive statistics
    - sample_examples.txt - example samples from each category
    - Console output with analysis
"""

import json
import random
from collections import Counter, defaultdict
from typing import List, Dict

# ============================================================================
# CONFIGURATION
# ============================================================================

INTERMEDIATE_FILE = 'training_data_intermediate.jsonl'
STATS_OUTPUT = 'data_statistics.json'
EXAMPLES_OUTPUT = 'sample_examples.txt'

CATEGORY_NAMES = {
    'I1': 'SPEAK - Direct address',
    'I2': 'SPEAK - Context follow-up',
    'I3': 'SPEAK - Implicit redirect',
    'S1': 'SILENT - No reference',
    'S2': 'SILENT - Mentioned but not addressed',
    'S3': 'SILENT - Context shifted away',
    'S4': 'SILENT - Incomplete sentence',
    'S5': 'SILENT - Explicit context switch'
}

# ============================================================================
# STATISTICS FUNCTIONS
# ============================================================================

def print_overall_statistics(samples: List[Dict]):
    """Print overall dataset statistics"""
    print("\n" + "="*70)
    print("OVERALL DATASET STATISTICS")
    print("="*70)
    
    total_samples = len(samples)
    print(f"\nTotal training samples: {total_samples:,}")
    
    # Unique meetings
    unique_meetings = set(s['meeting_id'] for s in samples)
    print(f"Unique meetings: {len(unique_meetings)}")
    print(f"  {', '.join(sorted(unique_meetings))}")
    
    # Unique sequences
    unique_sequences = set(s['sequence_id'] for s in samples)
    print(f"\nUnique sequences: {len(unique_sequences)}")
    print(f"Average samples per sequence: {total_samples / len(unique_sequences):.1f}")
    
    # Speakers
    all_speakers_sets = [set(s['all_speakers']) for s in samples]
    all_unique_speakers = set()
    for speakers in all_speakers_sets:
        all_unique_speakers.update(speakers)
    print(f"\nUnique speakers: {len(all_unique_speakers)}")
    print(f"  {', '.join(sorted(all_unique_speakers))}")


def print_decision_distribution(samples: List[Dict]):
    """Print decision distribution"""
    print("\n" + "="*70)
    print("DECISION DISTRIBUTION")
    print("="*70)
    
    total_samples = len(samples)
    decisions = [s['decision'] for s in samples]
    decision_counts = Counter(decisions)
    
    print("\nSPEAK vs SILENT:")
    for decision, count in decision_counts.most_common():
        percentage = count / total_samples * 100
        print(f"  {decision}: {count:,} ({percentage:.1f}%)")
    
    speak_count = decision_counts['SPEAK']
    silent_count = decision_counts['SILENT']
    ratio = silent_count / speak_count if speak_count > 0 else 0
    print(f"\nSILENT:SPEAK ratio: {ratio:.2f}:1")


def print_category_distribution(samples: List[Dict]):
    """Print category distribution"""
    print("\n" + "="*70)
    print("CATEGORY DISTRIBUTION")
    print("="*70)
    
    total_samples = len(samples)
    categories = [s['category'] for s in samples]
    category_counts = Counter(categories)
    
    print("\nSPEAK Categories (I1-I3):")
    for cat in ['I1', 'I2', 'I3']:
        count = category_counts.get(cat, 0)
        percentage = count / total_samples * 100
        print(f"  {cat} - {CATEGORY_NAMES[cat]}: {count:,} ({percentage:.1f}%)")
    
    print("\nSILENT Categories (S1-S5):")
    for cat in ['S1', 'S2', 'S3', 'S4', 'S5']:
        count = category_counts.get(cat, 0)
        percentage = count / total_samples * 100
        print(f"  {cat} - {CATEGORY_NAMES[cat]}: {count:,} ({percentage:.1f}%)")


def print_confidence_distribution(samples: List[Dict]):
    """Print confidence distribution"""
    print("\n" + "="*70)
    print("CONFIDENCE DISTRIBUTION")
    print("="*70)
    
    total_samples = len(samples)
    confidences = [s['confidence'] for s in samples]
    confidence_counts = Counter(confidences)
    
    print("\nOverall:")
    for conf in ['high', 'medium', 'low']:
        count = confidence_counts.get(conf, 0)
        percentage = count / total_samples * 100
        print(f"  {conf}: {count:,} ({percentage:.1f}%)")
    
    # Confidence by decision
    print("\nBy Decision:")
    for decision in ['SPEAK', 'SILENT']:
        decision_samples = [s for s in samples if s['decision'] == decision]
        print(f"  {decision}:")
        for conf in ['high', 'medium', 'low']:
            count = sum(1 for s in decision_samples if s['confidence'] == conf)
            if len(decision_samples) > 0:
                percentage = count / len(decision_samples) * 100
                print(f"    {conf}: {count:,} ({percentage:.1f}%)")


def print_context_statistics(samples: List[Dict]):
    """Print context length statistics"""
    print("\n" + "="*70)
    print("CONTEXT LENGTH STATISTICS")
    print("="*70)
    
    # Count turns in context
    context_turn_counts = [len(s['context'].split('\n')) if s['context'] != '(No previous context)' else 0 
                           for s in samples]
    
    print(f"\nContext turns per sample:")
    print(f"  Average: {sum(context_turn_counts) / len(context_turn_counts):.1f}")
    print(f"  Min: {min(context_turn_counts)}")
    print(f"  Max: {max(context_turn_counts)}")
    print(f"  Samples with no context: {sum(1 for c in context_turn_counts if c == 0):,}")
    
    # Context length distribution
    context_length_dist = Counter(context_turn_counts)
    print(f"\nDistribution:")
    for length in sorted(context_length_dist.keys())[:10]:
        count = context_length_dist[length]
        print(f"  {length} turns: {count:,} samples")


def print_duplicate_analysis(samples: List[Dict]):
    """Check for duplicates"""
    print("\n" + "="*70)
    print("DUPLICATE ANALYSIS")
    print("="*70)
    
    # Check for duplicate decision points
    decision_point_ids = [s['decision_point_id'] for s in samples]
    unique_ids = set(decision_point_ids)
    
    print(f"\nDecision point IDs:")
    print(f"  Total: {len(decision_point_ids):,}")
    print(f"  Unique: {len(unique_ids):,}")
    print(f"  Duplicates: {len(decision_point_ids) - len(unique_ids):,}")
    
    # Check for duplicate context+current combinations
    context_current_pairs = [(s['context'], s['current_turn']) for s in samples]
    unique_pairs = set(context_current_pairs)
    
    print(f"\nContext+Current combinations:")
    print(f"  Total: {len(context_current_pairs):,}")
    print(f"  Unique: {len(unique_pairs):,}")
    print(f"  Duplicates: {len(context_current_pairs) - len(unique_pairs):,}")


def print_samples_per_meeting(samples: List[Dict]):
    """Print samples per meeting"""
    print("\n" + "="*70)
    print("SAMPLES PER MEETING")
    print("="*70)
    
    meeting_counts = Counter(s['meeting_id'] for s in samples)
    total_samples = len(samples)
    
    print(f"\nSamples by meeting:")
    for meeting, count in meeting_counts.most_common():
        percentage = count / total_samples * 100
        print(f"  {meeting}: {count:,} ({percentage:.1f}%)")


def generate_example_samples(samples: List[Dict], examples_per_category: int = 5):
    """Generate example samples for each category"""
    print("\n" + "="*70)
    print("GENERATING EXAMPLE SAMPLES")
    print("="*70)
    
    category_examples = {}
    
    for cat in ['I1', 'I2', 'I3', 'S1', 'S2', 'S3', 'S4', 'S5']:
        cat_samples = [s for s in samples if s['category'] == cat]
        if cat_samples:
            # Randomly sample
            num_examples = min(examples_per_category, len(cat_samples))
            category_examples[cat] = random.sample(cat_samples, num_examples)
            print(f"  {cat}: {num_examples} examples")
        else:
            print(f"  {cat}: No samples found")
            category_examples[cat] = []
    
    return category_examples


def save_example_samples(category_examples: Dict, filename: str):
    """Save example samples to file"""
    with open(filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TRAINING DATASET EXAMPLE SAMPLES\n")
        f.write("="*70 + "\n\n")
        
        for cat in ['I1', 'I2', 'I3', 'S1', 'S2', 'S3', 'S4', 'S5']:
            f.write(f"\n{'='*70}\n")
            f.write(f"{cat}: {CATEGORY_NAMES[cat]}\n")
            f.write(f"{'='*70}\n\n")
            
            examples = category_examples.get(cat, [])
            if not examples:
                f.write("No samples in this category.\n")
                continue
            
            for i, sample in enumerate(examples, 1):
                f.write(f"Example {i}:\n")
                f.write(f"-" * 70 + "\n")
                f.write(f"Sample ID: {sample['sample_id']}\n")
                f.write(f"Meeting: {sample['meeting_id']}\n")
                f.write(f"Target Speaker: {sample['target_speaker']}\n")
                f.write(f"\nContext:\n{sample['context']}\n")
                f.write(f"\nCurrent Turn:\n{sample['current_turn']}\n")
                f.write(f"\nDecision: {sample['decision']}\n")
                f.write(f"Confidence: {sample['confidence']}\n")
                f.write(f"Reason: {sample['reason']}\n")
                f.write(f"\n")


def compile_statistics(samples: List[Dict]) -> Dict:
    """Compile all statistics into JSON"""
    total_samples = len(samples)
    
    # Collect all data
    unique_meetings = set(s['meeting_id'] for s in samples)
    unique_sequences = set(s['sequence_id'] for s in samples)
    all_speakers = set()
    for s in samples:
        all_speakers.update(s['all_speakers'])
    
    decisions = [s['decision'] for s in samples]
    decision_counts = Counter(decisions)
    
    categories = [s['category'] for s in samples]
    category_counts = Counter(categories)
    
    confidences = [s['confidence'] for s in samples]
    confidence_counts = Counter(confidences)
    
    context_turn_counts = [len(s['context'].split('\n')) if s['context'] != '(No previous context)' else 0 
                           for s in samples]
    
    meeting_counts = Counter(s['meeting_id'] for s in samples)
    
    # Decision point IDs
    decision_point_ids = [s['decision_point_id'] for s in samples]
    unique_ids = set(decision_point_ids)
    
    # Context+current pairs
    context_current_pairs = [(s['context'], s['current_turn']) for s in samples]
    unique_pairs = set(context_current_pairs)
    
    # Compile into dict
    statistics = {
        'dataset_overview': {
            'total_samples': total_samples,
            'unique_meetings': len(unique_meetings),
            'meetings': sorted(list(unique_meetings)),
            'unique_sequences': len(unique_sequences),
            'unique_speakers': len(all_speakers),
            'speakers': sorted(list(all_speakers)),
            'samples_per_sequence': total_samples / len(unique_sequences)
        },
        'decision_distribution': {
            'SPEAK': decision_counts['SPEAK'],
            'SILENT': decision_counts['SILENT'],
            'speak_percentage': decision_counts['SPEAK'] / total_samples * 100,
            'silent_percentage': decision_counts['SILENT'] / total_samples * 100,
            'silent_to_speak_ratio': decision_counts['SILENT'] / decision_counts['SPEAK']
        },
        'category_distribution': {
            cat: {
                'count': category_counts.get(cat, 0),
                'percentage': category_counts.get(cat, 0) / total_samples * 100,
                'name': CATEGORY_NAMES[cat]
            }
            for cat in ['I1', 'I2', 'I3', 'S1', 'S2', 'S3', 'S4', 'S5']
        },
        'confidence_distribution': {
            'overall': {
                conf: {
                    'count': confidence_counts.get(conf, 0),
                    'percentage': confidence_counts.get(conf, 0) / total_samples * 100
                }
                for conf in ['high', 'medium', 'low']
            },
            'by_decision': {
                decision: {
                    conf: sum(1 for s in samples 
                             if s['decision'] == decision and s['confidence'] == conf)
                    for conf in ['high', 'medium', 'low']
                }
                for decision in ['SPEAK', 'SILENT']
            }
        },
        'context_statistics': {
            'average_turns': sum(context_turn_counts) / len(context_turn_counts),
            'min_turns': min(context_turn_counts),
            'max_turns': max(context_turn_counts),
            'samples_with_no_context': sum(1 for c in context_turn_counts if c == 0)
        },
        'duplicate_analysis': {
            'unique_decision_points': len(unique_ids),
            'duplicate_decision_points': len(decision_point_ids) - len(unique_ids),
            'unique_context_current_pairs': len(unique_pairs),
            'duplicate_context_current_pairs': len(context_current_pairs) - len(unique_pairs)
        },
        'samples_per_meeting': dict(meeting_counts)
    }
    
    return statistics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    # Load intermediate samples
    samples = []
    with open(INTERMEDIATE_FILE, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"Loaded {len(samples):,} samples from intermediate JSONL")
    
    # Print all statistics
    print_overall_statistics(samples)
    print_decision_distribution(samples)
    print_category_distribution(samples)
    print_confidence_distribution(samples)
    print_context_statistics(samples)
    print_duplicate_analysis(samples)
    print_samples_per_meeting(samples)
    
    # Generate and save example samples
    category_examples = generate_example_samples(samples)
    save_example_samples(category_examples, EXAMPLES_OUTPUT)
    print(f"\n✓ Saved example samples to {EXAMPLES_OUTPUT}")
    
    # Compile and save statistics
    statistics = compile_statistics(samples)
    with open(STATS_OUTPUT, 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"✓ Saved statistics to {STATS_OUTPUT}")
    
    # Final summary
    print("\n" + "="*70)
    print("DATASET GENERATION COMPLETE")
    print("="*70)
    
    print(f"\nOutput Files:")
    print(f"  1. training_data_intermediate.jsonl - {len(samples):,} samples (human-readable)")
    print(f"  2. training_data_formatted.jsonl - {len(samples):,} samples (training format)")
    print(f"  3. {STATS_OUTPUT} - comprehensive statistics")
    print(f"  4. {EXAMPLES_OUTPUT} - example samples from each category")
    
    print(f"\nDataset Summary:")
    print(f"  Total Samples: {len(samples):,}")
    decisions = [s['decision'] for s in samples]
    decision_counts = Counter(decisions)
    print(f"  SPEAK: {decision_counts['SPEAK']:,} ({decision_counts['SPEAK']/len(samples)*100:.1f}%)")
    print(f"  SILENT: {decision_counts['SILENT']:,} ({decision_counts['SILENT']/len(samples)*100:.1f}%)")
    print(f"  Categories: I1-I3 (SPEAK), S1-S5 (SILENT)")
    unique_meetings = set(s['meeting_id'] for s in samples)
    all_speakers = set()
    for s in samples:
        all_speakers.update(s['all_speakers'])
    print(f"  Meetings: {len(unique_meetings)}")
    print(f"  Speakers: {len(all_speakers)}")
    
    print(f"\n" + "="*70)
    print("Ready for model training!")
    print("="*70)


if __name__ == '__main__':
    main()
