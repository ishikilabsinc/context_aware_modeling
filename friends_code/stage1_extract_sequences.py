#!/usr/bin/env python3
"""
Stage 1: Extract sequences from Friends-MMC metadata (explicit addressing).

Loads JSON metadata, detects explicit name mention, builds
context → addressing turn → response → continuation, deduplicates,
and outputs ICSI-compatible stage1 schema for Stage 2.

Input: friends_mmc/{num_turns}_turns/{split}-metadata.json
Output: json_dumps/stage1_combined_sequences.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm
from config import (
    FRIENDS_MMC_DIR,
    FRIENDS_NUM_TURNS,
    FRIENDS_SPLITS,
    JSON_DUMPS_DIR,
    STAGE1_MAX_SEQUENCES_PER_SPLIT,
)

JSON_DUMPS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = JSON_DUMPS_DIR / 'stage1_combined_sequences.json'

FRIENDS_NAMES = [
    'monica', 'rachel', 'phoebe', 'ross', 'chandler', 'joey',
    'gunther', 'carol', 'susan', 'richard', 'janice', 'mike',
    'emily', 'julio', 'pete', 'bonnie', 'ginger', 'frank jr',
]


def detect_explicit_naming(text: str, speaker_names: List[str]) -> Optional[str]:
    """Detect explicit name mention in text."""
    text_lower = text.lower()
    for name in speaker_names:
        name_lower = name.lower()
        patterns = [
            rf"\b{re.escape(name_lower)}\s*[,!?]",
            rf"\b{re.escape(name_lower)}\s+",
            rf",\s*{re.escape(name_lower)}\s*[,!?]",
        ]
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return name
    return None


def extract_sequences_from_metadata(
    friends_mmc_dir: Path,
    num_turns: int,
    split: str,
    max_sequences: Optional[int] = None,
) -> List[Dict]:
    """Filter one metadata file for explicit addressing and return normalized sequences."""
    metadata_file = friends_mmc_dir / f'{num_turns}_turns' / f'{split}-metadata.json'
    if not metadata_file.exists():
        print(f"File not found: {metadata_file}")
        return []

    with open(metadata_file, 'r') as f:
        all_conversations = json.load(f)

    filtered_sequences = []
    for conv_idx, conversation in enumerate(all_conversations):
        if len(conversation) < 2:
            continue

        all_speakers = set(turn['speaker'] for turn in conversation)
        speaker_names = list(all_speakers) + FRIENDS_NAMES

        explicit_addressing_idx = None
        addressee_name = None
        addressing_speaker = None

        for i in range(len(conversation)):
            current_turn = conversation[i]
            current_text = current_turn.get('content', '')

            explicitly_named = detect_explicit_naming(current_text, speaker_names)

            if explicitly_named and explicitly_named != current_turn['speaker']:
                explicit_addressing_idx = i
                addressee_name = explicitly_named
                addressing_speaker = current_turn['speaker']
                break

        if explicit_addressing_idx is None:
            continue

        i = explicit_addressing_idx
        current_turn = conversation[i]
        current_text = current_turn.get('content', '')

        context_start = max(0, i - 2)
        context = conversation[context_start:i]

        response = None
        response_idx = None
        for j in range(i + 1, len(conversation)):
            if conversation[j]['speaker'] == addressee_name:
                response = conversation[j]
                response_idx = j
                break

        if not response:
            continue

        continuation = []
        participants = {addressing_speaker, addressee_name}
        for k in range(response_idx + 1, len(conversation)):
            next_turn = conversation[k]
            next_text = next_turn.get('content', '')

            new_named = detect_explicit_naming(next_text, speaker_names)
            if new_named and new_named != addressee_name and new_named != next_turn['speaker']:
                break

            if next_turn['speaker'] in participants:
                continuation.append(next_turn)
            else:
                if len(continuation) > 0:
                    break

        conversation_id = f"{split}_{num_turns}turns_{conv_idx}"
        # ICSI-compatible schema: addressees list, text in turns, sequence_id, meeting_id
        filtered_sequences.append({
            'sequence_id': conversation_id,
            'meeting_id': conversation_id,
            'source': 'friends_explicit',
            'context': [{'speaker': c['speaker'], 'text': c['content']} for c in context],
            'addressing_turn': {
                'speaker': addressing_speaker,
                'addressees': [addressee_name],
                'text': current_text,
                'is_explicit': True,
                'inference_confidence': 10,
            },
            'response': {
                'speaker': response['speaker'],
                'text': response['content'],
            },
            'continuation': [{'speaker': c['speaker'], 'text': c['content']} for c in continuation],
        })

        if max_sequences and len(filtered_sequences) >= max_sequences:
            break

    # Deduplicate by (addressing text, response text), keep longer continuation
    seen = {}
    deduplicated = []
    for seq in filtered_sequences:
        key = (seq['addressing_turn']['text'], seq['response']['text'])
        if key not in seen:
            seen[key] = seq
            deduplicated.append(seq)
        else:
            existing = seen[key]
            if len(seq.get('continuation', [])) > len(existing.get('continuation', [])):
                idx = deduplicated.index(existing)
                deduplicated[idx] = seq
                seen[key] = seq

    return deduplicated


def main():
    parser = argparse.ArgumentParser(description='Friends Stage 1: Extract sequences from metadata')
    parser.add_argument('--splits', nargs='+', default=FRIENDS_SPLITS, help='Splits to process')
    parser.add_argument('--num-turns', nargs='+', type=int, default=FRIENDS_NUM_TURNS,
                        help='Turn windows (e.g. 5 8)')
    parser.add_argument('--max-sequences', type=int, default=STAGE1_MAX_SEQUENCES_PER_SPLIT,
                        help='Cap sequences per (split, num_turns) for debugging')
    args = parser.parse_args()

    if not FRIENDS_MMC_DIR.exists():
        print(f"ERROR: Friends-MMC dir not found: {FRIENDS_MMC_DIR}")
        return

    all_sequences = []
    pairs = [(n, s) for n in args.num_turns for s in args.splits]
    for num_turns, split in tqdm(pairs, desc="Extracting sequences", unit="file"):
        seqs = extract_sequences_from_metadata(
            FRIENDS_MMC_DIR,
            num_turns,
            split,
            max_sequences=args.max_sequences,
        )
        all_sequences.extend(seqs)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_sequences, f, indent=2)

    print(f"\n✓ Saved {len(all_sequences)} sequences to {OUTPUT_FILE}")
    print("Next: run stage2_generate_decision_points.py")


if __name__ == '__main__':
    main()
