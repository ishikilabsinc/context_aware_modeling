#!/usr/bin/env python3
"""
Stage 1b: Build sequences from turns (heuristic only, no API).

For each turn i with next turn i+1, treat next speaker as addressee:
  context = turns[max(0, i-N):i]
  addressing_turn = turn i, addressees = [ speaker(turn i+1) ]
  response = turn i+1
  continuation = turns[i+2 : i+2+K]

Input: json_dumps/stage1_turns_by_call.json
Output: json_dumps/stage1_combined_sequences.json (same schema as ICSI/AMI for Stage 2)
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

from config import (
    JSON_DUMPS_DIR,
    STAGE1_MAX_CONTEXT_TURNS,
    STAGE1B_MAX_CONTINUATION_TURNS,
    STAGE1B_HEURISTIC_CONFIDENCE,
)

INPUT_FILE = JSON_DUMPS_DIR / 'stage1_turns_by_call.json'
OUTPUT_FILE = JSON_DUMPS_DIR / 'stage1_combined_sequences.json'


def build_sequences_for_call(call_id: str, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build heuristic sequences: for each turn i, addressee = speaker of turn i+1."""
    sequences = []
    n = len(turns)
    for i in range(n - 1):
        next_speaker = turns[i + 1]['speaker']
        context_start = max(0, i - STAGE1_MAX_CONTEXT_TURNS)
        context = [{'speaker': t['speaker'], 'text': t['text']} for t in turns[context_start:i]]
        continuation_end = min(n, i + 2 + STAGE1B_MAX_CONTINUATION_TURNS)
        continuation = [{'speaker': t['speaker'], 'text': t['text']} for t in turns[i + 2 : continuation_end]]

        seq = {
            'meeting_id': call_id,
            'sequence_id': f"{call_id}_seq{i}",
            'source': 'spgi_heuristic',
            'context': context,
            'addressing_turn': {
                'speaker': turns[i]['speaker'],
                'addressees': [next_speaker],
                'text': turns[i]['text'],
                'is_explicit': False,
                'inference_confidence': STAGE1B_HEURISTIC_CONFIDENCE,
            },
            'response': {
                'speaker': turns[i + 1]['speaker'],
                'text': turns[i + 1]['text'],
            },
            'continuation': continuation,
        }
        sequences.append(seq)
    return sequences


def main():
    JSON_DUMPS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found. Run stage1_extract_turns.py first.")
        return

    with open(INPUT_FILE, 'r') as f:
        turns_by_call = json.load(f)

    all_sequences = []
    items = [(cid, t) for cid, t in turns_by_call.items() if len(t) >= 2]
    for call_id, turns in tqdm(items, desc="Building sequences", unit="call"):
        all_sequences.extend(build_sequences_for_call(call_id, turns))

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_sequences, f, indent=2)

    print(f"✓ Saved {len(all_sequences):,} sequences to {OUTPUT_FILE}")
    print("Next: run stage2_generate_decision_points.py")


if __name__ == '__main__':
    main()
