#!/usr/bin/env python3
"""
Stage 2: Generate Decision Points (SPGI) — flat turn-stream approach.

Reads the turn stream directly from Stage 1 and generates one decision point
per (turn, target_speaker) pair.  No intermediate "sequence" abstraction.

Addressing signal (who the current turn is directed at) is derived from
conversational dynamics visible in the PAST only:
  - Primary addressee = the most recent speaker before the current one
    (the natural "reply-to" partner in a conversation).
  - Recently-active speakers (last K turns) feed into SILENT_ref downstream.

This avoids the future-leaking heuristic (next_speaker = addressee) used
by the legacy stage1b+stage2 pair.

Input:  json_dumps/stage1_turns_by_call.json
Output: json_dumps/stage2_decision_points.jsonl
"""

import json
from pathlib import Path
from typing import List, Dict, Set

from tqdm import tqdm

from config import (
    JSON_DUMPS_DIR,
    STAGE1_MAX_CONTEXT_TURNS,
    STAGE1B_HEURISTIC_CONFIDENCE,
    STAGE2_MIN_SPEAKERS_PER_CALL,
    STAGE2_MIN_CONTEXT_TURNS,
)

INPUT_FILE = JSON_DUMPS_DIR / 'stage1_turns_by_call.json'
OUTPUT_FILE = JSON_DUMPS_DIR / 'stage2_decision_points.jsonl'

FILLER_PATTERNS_SIMPLE = {
    'yes', 'yeah', 'yep', 'no', 'nope', 'okay', 'ok',
    'thank you', 'thanks', 'mm-hmm', 'uh-huh', 'right',
    'sure', 'alright', 'got it',
}


def is_trivial_turn(text: str) -> bool:
    """True for very short backchannel / filler turns."""
    cleaned = text.strip().lower().rstrip('.!?,')
    return cleaned in FILLER_PATTERNS_SIMPLE or len(cleaned.split()) < 2


def find_primary_addressee(turns: List[Dict], current_idx: int) -> List[str]:
    """
    Return the most recent speaker before turns[current_idx] who is
    different from the current speaker.  This is the "reply-to" partner.
    """
    current_speaker = turns[current_idx]['speaker']
    for j in range(current_idx - 1, -1, -1):
        if turns[j]['speaker'] != current_speaker:
            return [turns[j]['speaker']]
    return []


def generate_decision_points_for_call(
    call_id: str,
    turns: List[Dict],
) -> List[Dict]:
    """Yield decision points from a single call's turn stream."""

    all_speakers: Set[str] = {t['speaker'] for t in turns}

    if len(all_speakers) < STAGE2_MIN_SPEAKERS_PER_CALL:
        return []

    decision_points: List[Dict] = []
    sorted_speakers = sorted(all_speakers)

    for i in range(len(turns) - 1):
        if i < STAGE2_MIN_CONTEXT_TURNS:
            continue

        current_turn = turns[i]
        current_speaker = current_turn['speaker']
        current_text = current_turn['text']

        if is_trivial_turn(current_text):
            continue

        next_speaker = turns[i + 1]['speaker']

        context_start = max(0, i - STAGE1_MAX_CONTEXT_TURNS)
        context_turns = [
            {'speaker': t['speaker'], 'text': t['text']}
            for t in turns[context_start:i]
        ]

        addressees = find_primary_addressee(turns, i)

        for target_speaker in sorted_speakers:
            if target_speaker == current_speaker:
                continue

            target_is_addressed = target_speaker in addressees
            target_spoke_next = (next_speaker == target_speaker)

            decision_points.append({
                'decision_point_id': f"{call_id}_t{i}_target{target_speaker}",
                'sequence_id': f"{call_id}_t{i}",
                'meeting_id': call_id,
                'target_speaker': target_speaker,
                'all_speakers': sorted_speakers,
                'turn_index': i,
                'turn_type': 'addressing',
                'source': 'spgi_heuristic',
                'is_explicit': False,
                'inference_confidence': STAGE1B_HEURISTIC_CONFIDENCE,
                'context_turns': context_turns,
                'current_turn': {
                    'speaker': current_speaker,
                    'text': current_text,
                },
                'addressees_in_current': addressees,
                'target_is_addressed': target_is_addressed,
                'target_spoke_next': target_spoke_next,
            })

    return decision_points


def main():
    JSON_DUMPS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found. Run stage1_extract_turns.py first.")
        return

    print(f"Loading turns from {INPUT_FILE} …")
    with open(INPUT_FILE, 'r') as f:
        turns_by_call = json.load(f)

    calls = [
        (cid, t) for cid, t in turns_by_call.items() if len(t) >= 2
    ]
    print(f"Loaded {len(calls)} calls")

    total_dps = 0
    skipped_calls = 0

    with open(OUTPUT_FILE, 'w') as fout:
        for call_id, turns in tqdm(calls, desc="Decision points", unit="call"):
            dps = generate_decision_points_for_call(call_id, turns)
            if not dps:
                skipped_calls += 1
                continue
            for dp in dps:
                fout.write(json.dumps(dp) + '\n')
                total_dps += 1

    print(f"\n✓ Saved {total_dps:,} decision points to {OUTPUT_FILE}")
    print(f"  Skipped {skipped_calls} calls (< {STAGE2_MIN_SPEAKERS_PER_CALL} speakers)")
    print("Next: run stage3_label_and_categorize.py")


if __name__ == '__main__':
    main()
