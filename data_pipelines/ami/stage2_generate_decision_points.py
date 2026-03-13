#!/usr/bin/env python3
"""
Stage 2: Generate Decision Points

Goal:
    From each conversation sequence, create multiple decision points (moments where
    we decide SPEAK/SILENT). For each turn in the conversation, generate decision 
    points for each speaker: "Should they respond after this turn?"

Input:
    - stage1_sequences.json from Stage 1

Output:
    - stage2_decision_points.json - decision points for Stage 3
    
Process:
    1. Build complete timeline of all turns (context + addressing + response + continuation)
    2. At each turn, for each speaker (except current speaker):
       - Create a decision point: "Should this speaker respond after this turn?"
       - Record context, current turn, addressing info, and ground truth
"""

import json
from pathlib import Path
from typing import List, Dict, Set
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory for all intermediate JSON dumps across stages
JSON_DUMPS_DIR = Path(__file__).parent / 'json_dumps'
JSON_DUMPS_DIR.mkdir(exist_ok=True)

INPUT_FILE = JSON_DUMPS_DIR / 'stage1_combined_sequences.json'
OUTPUT_FILE = JSON_DUMPS_DIR / 'stage2_decision_points.json'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_speakers_in_sequence(sequence: Dict) -> Set[str]:
    """
    Extract all unique speakers from a sequence
    
    Returns:
        Set of speaker identifiers (e.g., {'A', 'B', 'C', 'D'})
    """
    speakers = set()
    
    # From context
    for turn in sequence['context']:
        speakers.add(turn['speaker'])
    
    # From addressing turn
    speakers.add(sequence['addressing_turn']['speaker'])
    speakers.update(sequence['addressing_turn']['addressees'])
    
    # From response
    speakers.add(sequence['response']['speaker'])
    
    # From continuation
    for turn in sequence['continuation']:
        speakers.add(turn['speaker'])
    
    return speakers


def generate_decision_points_from_sequence(sequence: Dict) -> List[Dict]:
    """
    Generate decision points from a conversation sequence
    
    For each turn, create decision points for all speakers (except current speaker):
    "Should speaker X respond after this turn?"
    
    Args:
        sequence: Conversation sequence from Stage 1
        
    Returns:
        List of decision points
    """
    decision_points = []
    
    sequence_id = sequence['sequence_id']
    meeting_id = sequence['meeting_id']
    
    # Get all speakers in this conversation
    all_speakers = get_all_speakers_in_sequence(sequence)
    
    # Build the full conversation timeline
    all_turns = []
    
    # Add context turns
    for turn in sequence['context']:
        all_turns.append({
            'speaker': turn['speaker'],
            'text': turn['text'],
            'addressees': [],  # Context turns don't have explicit addressing
            'is_addressing': False,
            'turn_type': 'context'
        })
    
    # Add addressing turn (explicit addressee annotations)
    all_turns.append({
        'speaker': sequence['addressing_turn']['speaker'],
        'text': sequence['addressing_turn']['text'],
        'addressees': sequence['addressing_turn']['addressees'],
        'is_addressing': True,
        'turn_type': 'addressing'
    })
    
    # Add response turn
    all_turns.append({
        'speaker': sequence['response']['speaker'],
        'text': sequence['response']['text'],
        'addressees': [],
        'is_addressing': False,
        'turn_type': 'response'
    })
    
    # Add continuation turns
    for turn in sequence['continuation']:
        all_turns.append({
            'speaker': turn['speaker'],
            'text': turn['text'],
            'addressees': [],
            'is_addressing': False,
            'turn_type': 'continuation'
        })
    
    # Generate decision points at each turn
    for turn_idx, current_turn in enumerate(all_turns):
        # Build context up to this turn
        context_turns = all_turns[:turn_idx]
        
        # For each speaker, create a decision point
        for target_speaker in all_speakers:
            # Skip if target is the current speaker (can't interrupt self)
            if target_speaker == current_turn['speaker']:
                continue
            
            # Check if target is addressed in current turn
            target_is_addressed = target_speaker in current_turn['addressees']
            
            # Check if target actually spoke next in the original sequence (GROUND TRUTH)
            # We look ahead one turn to see who spoke next
            target_spoke_next = False
            if turn_idx + 1 < len(all_turns):
                next_turn = all_turns[turn_idx + 1]
                target_spoke_next = (next_turn['speaker'] == target_speaker)
            
            # Create decision point
            decision_point = {
                'decision_point_id': f"{sequence_id}_turn{turn_idx}_target{target_speaker}",
                'sequence_id': sequence_id,
                'meeting_id': meeting_id,
                'target_speaker': target_speaker,
                'all_speakers': sorted(list(all_speakers)),
                'turn_index': turn_idx,
                'turn_type': current_turn['turn_type'],
                
                # Context: all turns before current
                'context_turns': [
                    {'speaker': t['speaker'], 'text': t['text']} 
                    for t in context_turns
                ],
                
                # Current turn after which we make decision
                'current_turn': {
                    'speaker': current_turn['speaker'],
                    'text': current_turn['text']
                },
                
                # Addressing info (from AMI annotations)
                'addressees_in_current': current_turn['addressees'],
                'target_is_addressed': target_is_addressed,
                
                # Ground truth from recorded conversation
                'target_spoke_next': target_spoke_next,
            }
            
            decision_points.append(decision_point)
    
    return decision_points


def print_statistics(all_decision_points: List[Dict]):
    """Print statistics about generated decision points"""
    print("\n" + "="*70)
    print("STAGE 2 STATISTICS")
    print("="*70)
    
    # Turn types
    turn_types = [dp['turn_type'] for dp in all_decision_points]
    turn_type_counts = Counter(turn_types)
    print(f"\nDecision points by turn type:")
    for turn_type, count in turn_type_counts.most_common():
        print(f"  {turn_type}: {count} ({count/len(all_decision_points)*100:.1f}%)")
    
    # Target addressed vs not
    addressed_count = sum(1 for dp in all_decision_points if dp['target_is_addressed'])
    not_addressed_count = len(all_decision_points) - addressed_count
    print(f"\nAddressing status:")
    print(f"  Target addressed: {addressed_count} ({addressed_count/len(all_decision_points)*100:.1f}%)")
    print(f"  Target NOT addressed: {not_addressed_count} ({not_addressed_count/len(all_decision_points)*100:.1f}%)")
    
    # Target spoke next vs didn't
    spoke_next_count = sum(1 for dp in all_decision_points if dp['target_spoke_next'])
    didnt_speak_count = len(all_decision_points) - spoke_next_count
    print(f"\nTarget response (ground truth):") 
    print(f"  Target spoke next: {spoke_next_count} ({spoke_next_count/len(all_decision_points)*100:.1f}%)")
    print(f"  Target didn't speak: {didnt_speak_count} ({didnt_speak_count/len(all_decision_points)*100:.1f}%)")
    
    # Context lengths
    context_lengths = [len(dp['context_turns']) for dp in all_decision_points]
    print(f"\nContext length:")
    print(f"  Average: {sum(context_lengths)/len(context_lengths):.1f} turns")
    print(f"  Range: {min(context_lengths)} - {max(context_lengths)} turns")
    
    # Speakers per decision point
    speaker_counts = [len(dp['all_speakers']) for dp in all_decision_points]
    print(f"\nSpeakers per conversation:")
    print(f"  Average: {sum(speaker_counts)/len(speaker_counts):.1f}")
    print(f"  Range: {min(speaker_counts)} - {max(speaker_counts)}")
    
    print(f"\n" + "="*70)


def print_examples(all_decision_points: List[Dict], num_examples: int = 5):
    """Print example decision points"""
    print("\n" + "="*70)
    print("EXAMPLE DECISION POINTS")
    print("="*70 + "\n")
    
    for i, dp in enumerate(all_decision_points[:num_examples]):
        print(f"Decision Point {i+1}: {dp['decision_point_id']}")
        print(f"Meeting: {dp['meeting_id']}")
        print(f"Target Speaker: {dp['target_speaker']}")
        print(f"Turn Type: {dp['turn_type']}")
        print(f"All Speakers: {', '.join(dp['all_speakers'])}")
        
        print(f"\nContext ({len(dp['context_turns'])} turns):")
        for turn in dp['context_turns'][-2:]:
            print(f"  [{turn['speaker']}]: {turn['text'][:60]}...")
        
        print(f"\nCurrent Turn:")
        print(f"  [{dp['current_turn']['speaker']}]: {dp['current_turn']['text'][:60]}...")
        
        print(f"\nDecision Info:")
        print(f"  Addressees in current: {dp['addressees_in_current'] if dp['addressees_in_current'] else 'None'}")
        print(f"  Target is addressed: {dp['target_is_addressed']}")
        print(f"  Target spoke next: {dp['target_spoke_next']}")
        
        print("\n" + "-"*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    # Load sequences from Stage 1
    with open(INPUT_FILE, 'r') as f:
        sequences = json.load(f)
    
    print(f"Loaded {len(sequences)} sequences from Stage 1")
    print("Generating decision points from sequences...")
    print("="*70)
    
    # Generate decision points from all sequences
    all_decision_points = []
    
    for i, sequence in enumerate(sequences):
        dps = generate_decision_points_from_sequence(sequence)
        all_decision_points.extend(dps)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(sequences)} sequences...")
    
    print("="*70)
    print(f"\nGenerated {len(all_decision_points):,} decision points from {len(sequences)} sequences")
    print(f"Average: {len(all_decision_points)/len(sequences):.1f} decision points per sequence")
    
    # Print examples and statistics
    print_examples(all_decision_points)
    print_statistics(all_decision_points)
    
    # Save decision points for next stage
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_decision_points, f, indent=2)
    
    print(f"\n✓ Saved {len(all_decision_points):,} decision points to {OUTPUT_FILE}")
    print(f"\nReady for Stage 3: Label SPEAK/SILENT")


if __name__ == '__main__':
    main()
