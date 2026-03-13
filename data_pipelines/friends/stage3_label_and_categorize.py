#!/usr/bin/env python3
"""
Stage 3: Label SPEAK/SILENT and Assign Categories (Friends)

For each decision point:
- Assign SPEAK or SILENT based on target_spoke_next (ground truth)
- Assign category (SPEAK_explicit, SPEAK_implicit, SILENT_no_ref, SILENT_ref)

Input: json_dumps/stage2_decision_points.json
Output: json_dumps/stage3_categorized_samples.json
"""

import json
import re
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm
from config import JSON_DUMPS_DIR

JSON_DUMPS_DIR.mkdir(exist_ok=True)
INPUT_FILE = JSON_DUMPS_DIR / 'stage2_decision_points.json'
OUTPUT_FILE = JSON_DUMPS_DIR / 'stage3_categorized_samples.json'


def label_decision_point(dp: Dict) -> Tuple[str, str, str]:
    target_is_addressed = dp['target_is_addressed']
    target_spoke_next = dp['target_spoke_next']
    target_speaker = dp['target_speaker']
    current_speaker = dp['current_turn']['speaker']
    decision = 'SPEAK' if target_spoke_next else 'SILENT'

    if target_spoke_next:
        if target_is_addressed:
            return decision, 'high', f"{target_speaker} was addressed by {current_speaker} and spoke (ground truth)"
        return decision, 'high', f"{target_speaker} spoke next (ground truth)"
    else:
        if dp['addressees_in_current']:
            addressees_str = ', '.join(dp['addressees_in_current'])
            return decision, 'high', f"{current_speaker} addressed {addressees_str}, not {target_speaker} (ground truth)"
        return decision, 'high', f"{target_speaker} did not speak next (ground truth)"


def check_target_mentioned(current_text: str, target_speaker: str) -> bool:
    text_lower = current_text.lower()
    target_lower = target_speaker.lower()
    patterns = [
        rf"\b{re.escape(target_lower)}\b",
        f"speaker {target_lower}",
        f"{target_lower} said",
        f"tell {target_lower}",
        f"ask {target_lower}",
    ]
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def check_incomplete_sentence(text: str) -> bool:
    text = text.strip()
    if text.endswith('...'):
        return True
    if len(text.split()) <= 2:
        return True
    incomplete_markers = ['...', 'um', 'uh', 'hmm']
    last_words = text.lower().split()[-2:]
    for marker in incomplete_markers:
        if marker in last_words:
            return True
    return False


def was_target_in_recent_exchange(context_turns: list, target_speaker: str, window: int = 3) -> bool:
    if not context_turns:
        return False
    recent_speakers = [turn['speaker'] for turn in context_turns[-window:]]
    return target_speaker in recent_speakers


def assign_category(sample: Dict) -> str:
    decision = sample['decision']
    target_speaker = sample['target_speaker']
    target_is_addressed = sample['target_is_addressed']
    current_text = sample['current_turn']['text']
    context_turns = sample['context_turns']

    if decision == 'SPEAK':
        if target_is_addressed:
            return 'SPEAK_explicit'
        return 'SPEAK_implicit'
    else:
        target_mentioned = check_target_mentioned(current_text, target_speaker)
        if target_mentioned and not target_is_addressed:
            return 'SILENT_ref'
        if was_target_in_recent_exchange(context_turns, target_speaker, window=2):
            return 'SILENT_ref'
        return 'SILENT_no_ref'


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found. Run stage2_generate_decision_points.py first.")
        return

    with open(INPUT_FILE, 'r') as f:
        decision_points = json.load(f)

    print(f"Loaded {len(decision_points):,} decision points")
    categorized = []
    for dp in tqdm(decision_points, desc="Label & categorize", unit=" sample"):
        decision, confidence, reason = label_decision_point(dp)
        sample = dp.copy()
        sample['decision'] = decision
        sample['confidence'] = confidence
        sample['reason'] = reason
        sample['category'] = assign_category(sample)
        categorized.append(sample)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(categorized, f, indent=2)

    print(f"✓ Saved {len(categorized):,} categorized samples to {OUTPUT_FILE}")
    print("Next: run stage4_filter_quality.py")


if __name__ == '__main__':
    main()
