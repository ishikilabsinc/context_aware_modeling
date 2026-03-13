#!/usr/bin/env python3
"""
Stage 2: Generate Decision Points (Friends)

From each conversation sequence (from Stage 1), create decision points:
"Should speaker X respond after this turn?" for each turn and each speaker.

Input: json_dumps/stage1_combined_sequences.json
Output: json_dumps/stage2_decision_points.json
"""

import json
from pathlib import Path
from typing import List, Dict, Set

from tqdm import tqdm
from config import JSON_DUMPS_DIR

JSON_DUMPS_DIR.mkdir(exist_ok=True)

INPUT_FILE = JSON_DUMPS_DIR / 'stage1_combined_sequences.json'
OUTPUT_FILE = JSON_DUMPS_DIR / 'stage2_decision_points.json'


def get_all_speakers_in_sequence(sequence: Dict) -> Set[str]:
    speakers = set()
    for turn in sequence['context']:
        speakers.add(turn['speaker'])
    speakers.add(sequence['addressing_turn']['speaker'])
    speakers.update(sequence['addressing_turn']['addressees'])
    speakers.add(sequence['response']['speaker'])
    for turn in sequence['continuation']:
        speakers.add(turn['speaker'])
    return speakers


def generate_decision_points_from_sequence(sequence: Dict) -> List[Dict]:
    decision_points = []
    sequence_id = sequence['sequence_id']
    meeting_id = sequence['meeting_id']
    source = sequence.get('source', 'unknown')
    addressing_turn = sequence.get('addressing_turn', {})
    is_explicit = addressing_turn.get('is_explicit', True)
    inference_confidence = addressing_turn.get('inference_confidence', 10)

    all_speakers = get_all_speakers_in_sequence(sequence)

    all_turns = []
    for turn in sequence['context']:
        all_turns.append({
            'speaker': turn['speaker'],
            'text': turn['text'],
            'addressees': [],
            'is_addressing': False,
            'turn_type': 'context'
        })
    all_turns.append({
        'speaker': sequence['addressing_turn']['speaker'],
        'text': sequence['addressing_turn']['text'],
        'addressees': sequence['addressing_turn']['addressees'],
        'is_addressing': True,
        'turn_type': 'addressing'
    })
    all_turns.append({
        'speaker': sequence['response']['speaker'],
        'text': sequence['response']['text'],
        'addressees': [],
        'is_addressing': False,
        'turn_type': 'response'
    })
    for turn in sequence['continuation']:
        all_turns.append({
            'speaker': turn['speaker'],
            'text': turn['text'],
            'addressees': [],
            'is_addressing': False,
            'turn_type': 'continuation'
        })

    for turn_idx, current_turn in enumerate(all_turns):
        context_turns = all_turns[:turn_idx]
        for target_speaker in all_speakers:
            if target_speaker == current_turn['speaker']:
                continue
            target_is_addressed = target_speaker in current_turn['addressees']
            target_spoke_next = False
            if turn_idx + 1 < len(all_turns):
                target_spoke_next = (all_turns[turn_idx + 1]['speaker'] == target_speaker)

            decision_points.append({
                'decision_point_id': f"{sequence_id}_turn{turn_idx}_target{target_speaker}",
                'sequence_id': sequence_id,
                'meeting_id': meeting_id,
                'target_speaker': target_speaker,
                'all_speakers': sorted(list(all_speakers)),
                'turn_index': turn_idx,
                'turn_type': current_turn['turn_type'],
                'source': source,
                'is_explicit': is_explicit,
                'inference_confidence': inference_confidence,
                'context_turns': [{'speaker': t['speaker'], 'text': t['text']} for t in context_turns],
                'current_turn': {'speaker': current_turn['speaker'], 'text': current_turn['text']},
                'addressees_in_current': current_turn['addressees'],
                'target_is_addressed': target_is_addressed,
                'target_spoke_next': target_spoke_next,
            })

    return decision_points


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found. Run stage1_extract_sequences.py first.")
        return

    with open(INPUT_FILE, 'r') as f:
        sequences = json.load(f)

    print(f"Loaded {len(sequences)} sequences from Stage 1")
    all_decision_points = []
    for seq in tqdm(sequences, desc="Decision points", unit="seq"):
        all_decision_points.extend(generate_decision_points_from_sequence(seq))

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_decision_points, f, indent=2)

    print(f"✓ Saved {len(all_decision_points):,} decision points to {OUTPUT_FILE}")
    print("Next: run stage3_label_and_categorize.py")


if __name__ == '__main__':
    main()
