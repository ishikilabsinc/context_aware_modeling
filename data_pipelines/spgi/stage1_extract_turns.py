#!/usr/bin/env python3
"""
Stage 1: Extract turns from SPGI word-level segment JSONs.

Loads spgi_dataset/{call_id}/{0,1,2,...}.json, merges segments in order,
sorts by time, groups consecutive words by speaker into turns.
Output: json_dumps/stage1_turns_by_call.json for Stage 1b.

Input: spgi/spgi_dataset/{call_id}/{0,1,2,...}.json
Output: json_dumps/stage1_turns_by_call.json  -> { call_id: [ { speaker, text, start_time, end_time }, ... ] }
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

from config import (
    SPGI_DATASET_DIR,
    JSON_DUMPS_DIR,
    STAGE1_MAX_CALLS,
    STAGE1_TURN_GAP_SECONDS,
)


def load_segment_words(segment_path: Path) -> List[Dict[str, Any]]:
    """Load one segment JSON and return list of { word, start_time, end_time, speaker }."""
    with open(segment_path, 'r') as f:
        data = json.load(f)
    words = []
    for k, v in data.items():
        if isinstance(v, dict) and 'word' in v and 'speaker' in v:
            words.append({
                'word': v['word'],
                'start_time': float(v.get('start_time', 0)),
                'end_time': float(v.get('end_time', 0)),
                'speaker': str(v['speaker']),
            })
    return words


def extract_turns_for_call(call_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all segment JSONs for one call in order (0.json, 1.json, ...),
    build global timeline, group by speaker into turns.
    """
    segment_files = sorted(
        call_dir.glob('*.json'),
        key=lambda p: int(p.stem) if p.stem.isdigit() else -1
    )
    if not segment_files:
        return []

    all_words = []
    segment_start = 0.0

    for seg_path in segment_files:
        seg_words = load_segment_words(seg_path)
        for w in seg_words:
            all_words.append({
                'word': w['word'],
                'start_time': segment_start + w['start_time'],
                'end_time': segment_start + w['end_time'],
                'speaker': w['speaker'],
            })
        if seg_words:
            segment_start += max(x['end_time'] for x in seg_words)

    all_words.sort(key=lambda x: (x['start_time'], x['speaker']))

    turns = []
    current_speaker = None
    current_words = []
    current_start = None
    current_end = None

    for w in all_words:
        gap_ok = True
        if STAGE1_TURN_GAP_SECONDS is not None and current_end is not None:
            gap_ok = (w['start_time'] - current_end) <= STAGE1_TURN_GAP_SECONDS

        if w['speaker'] == current_speaker and gap_ok:
            current_words.append(w['word'])
            current_end = w['end_time']
        else:
            if current_words:
                turns.append({
                    'speaker': current_speaker,
                    'text': ' '.join(current_words).strip(),
                    'start_time': current_start,
                    'end_time': current_end,
                })
            current_speaker = w['speaker']
            current_words = [w['word']]
            current_start = w['start_time']
            current_end = w['end_time']

    if current_words:
        turns.append({
            'speaker': current_speaker,
            'text': ' '.join(current_words).strip(),
            'start_time': current_start,
            'end_time': current_end,
        })

    return turns


def main():
    parser = argparse.ArgumentParser(description='SPGI Stage 1: Extract turns from word-level segments')
    parser.add_argument('--max-calls', type=int, default=STAGE1_MAX_CALLS, help='Cap number of calls (default from config)')
    args = parser.parse_args()

    if not SPGI_DATASET_DIR.exists():
        print(f"ERROR: Dataset not found at {SPGI_DATASET_DIR}")
        return

    JSON_DUMPS_DIR.mkdir(parents=True, exist_ok=True)

    call_dirs = sorted([d for d in SPGI_DATASET_DIR.iterdir() if d.is_dir()])
    if args.max_calls is not None:
        call_dirs = call_dirs[: args.max_calls]

    print(f"Processing {len(call_dirs)} calls from {SPGI_DATASET_DIR}")

    turns_by_call = {}
    for call_dir in tqdm(call_dirs, desc="Extracting turns", unit="call"):
        call_id = call_dir.name
        turns = extract_turns_for_call(call_dir)
        if turns:
            turns_by_call[call_id] = turns

    out_file = JSON_DUMPS_DIR / 'stage1_turns_by_call.json'
    with open(out_file, 'w') as f:
        json.dump(turns_by_call, f, indent=2)

    total_turns = sum(len(t) for t in turns_by_call.values())
    print(f"✓ Saved {len(turns_by_call)} calls, {total_turns:,} total turns to {out_file}")
    print("Next: run stage1b_build_sequences.py")


if __name__ == '__main__':
    main()
