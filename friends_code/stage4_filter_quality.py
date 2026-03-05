#!/usr/bin/env python3
"""
Stage 4: Filter and Balance High-Quality Samples (Friends)

Filter stage3 output: remove fillers, balance SPEAK/SILENT, deduplicate.

Input: json_dumps/stage3_categorized_samples.json
Output: data_final/stage4_filtered_samples.jsonl
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set
from collections import Counter, defaultdict

from tqdm import tqdm
from config import (
    JSON_DUMPS_DIR,
    OUTPUT_DIR,
    JSON_RUN_DIR,
    STAGE4_SPEAK_RATIO,
    STAGE4_MIN_TEXT_LENGTH,
    STAGE4_MAX_SAMPLES,
    STAGE4_MAX_CATEGORY_RATIO,
    STAGE4_ENABLE_DEDUPLICATION,
    STAGE4_DEDUP_NGRAM_N,
    STAGE4_DEDUP_SIMILARITY_THRESHOLD,
    STAGE4_DEDUP_SAME_MEETING_ONLY,
    STAGE4_DEDUP_CONTEXT_WINDOW,
    STAGE4_REMOVE_NO_CONTEXT,
    STAGE4_REMOVE_SHORT_QUERY,
    STAGE4_MIN_QUERY_WORDS,
    STAGE4_GEMINI_CONFIDENCE_FILTER,
)

INPUT_FILE = JSON_DUMPS_DIR / 'stage3_categorized_samples.json'
OUTPUT_FILE = 'stage4_filtered_samples.jsonl'

FILLER_PATTERNS = [
    r'^\s*mm+\s*[-.]?\s*hm+\s*\.?\s*$',
    r'^\s*uh+\s*[-.]?\s*huh+\s*\.?\s*$',
    r'^\s*um+\s*\.?\s*$',
    r'^\s*uh+\s*\.?\s*$',
    r'^\s*er+\s*\.?\s*$',
    r'^\s*ah+\s*\.?\s*$',
    r'^\s*oh+\s*\.?\s*$',
    r'^\s*hm+\s*\.?\s*$',
    r'^\s*\.\s*$',
]


def confidence_to_score(confidence_str: str) -> int:
    return {'high': 3, 'medium': 2, 'low': 1}.get(confidence_str.lower(), 0)


def is_filler_text(text: str) -> bool:
    t = text.strip().lower()
    for pattern in FILLER_PATTERNS:
        if re.match(pattern, t, re.IGNORECASE):
            return True
    return False


def is_low_quality(sample: Dict) -> bool:
    text = sample['current_turn']['text']
    if not text or not text.strip():
        return True
    if is_filler_text(text):
        return True
    text_alphanum = re.sub(r'[^\w\s]', '', text)
    if len(text_alphanum.strip()) < STAGE4_MIN_TEXT_LENGTH:
        return True
    if len(text_alphanum) < len(text) * 0.5:
        return True
    return False


def normalize_text(text: str) -> str:
    return ' '.join(re.sub(r'[^\w\s]', '', text.lower()).split())


def create_sample_signature(sample: Dict, context_window: int = 3) -> str:
    current_text = normalize_text(sample['current_turn']['text'])
    context_turns = sample.get('context_turns', [])
    recent = context_turns[-context_window:] if len(context_turns) > context_window else context_turns
    context_str = "|".join(f"{t['speaker']}:{normalize_text(t['text'])}" for t in recent)
    return f"{current_text}:::{context_str}:::{sample['decision']}"


def get_word_ngrams(text: str, n: int) -> Set[tuple]:
    """Word-level n-grams from normalized text (hashable tuples)."""
    norm = normalize_text(text)
    words = norm.split()
    if len(words) < n:
        return set((tuple(words),)) if words else set()
    return set(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def sample_to_ngram_set(sample: Dict, context_window: int, n: int) -> Set[tuple]:
    """N-grams from last context_window context turns + current turn (for similarity)."""
    parts = []
    context_turns = sample.get('context_turns', [])
    recent = context_turns[-context_window:] if len(context_turns) > context_window else context_turns
    for t in recent:
        parts.append(normalize_text(t['text']))
    parts.append(normalize_text(sample['current_turn']['text']))
    combined = " ".join(parts)
    return get_word_ngrams(combined, n)


def jaccard_similarity(a: Set[tuple], b: Set[tuple]) -> float:
    """Jaccard similarity between two n-gram sets; 0 if both empty."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _meeting_key(sample: Dict) -> str:
    return str(sample.get('meeting_id', ''))


def deduplicate_samples(samples: List[Dict]) -> List[Dict]:
    if not samples:
        return []
    sorted_samples = sorted(samples, key=lambda s: confidence_to_score(s['confidence']), reverse=True)
    context_window = min(5, STAGE4_DEDUP_CONTEXT_WINDOW)  # last 5 context + current only
    n = STAGE4_DEDUP_NGRAM_N
    thresh = STAGE4_DEDUP_SIMILARITY_THRESHOLD
    same_meeting_only = STAGE4_DEDUP_SAME_MEETING_ONLY
    if thresh >= 1.0:
        seen: Set[str] = set()
        deduped = []
        for s in sorted_samples:
            sig = create_sample_signature(s)
            if sig in seen:
                continue
            seen.add(sig)
            deduped.append(s)
        return deduped
    deduped = []
    kept_ngrams: Dict[tuple, List[Set[tuple]]] = defaultdict(list)
    for s in sorted_samples:
        ngrams = sample_to_ngram_set(s, context_window, n)
        decision = s['decision']
        key = (_meeting_key(s), decision) if same_meeting_only else (decision,)
        if any(jaccard_similarity(ngrams, k) >= thresh for k in kept_ngrams[key]):
            continue
        kept_ngrams[key].append(ngrams)
        deduped.append(s)
    return deduped


def has_context(sample: Dict) -> bool:
    return bool(sample.get('context_turns', []))


def add_filter_fields(sample: Dict) -> Dict:
    """Add fields so consumers can filter from the JSONL (num_context_turns, num_query_words, etc.)."""
    text = sample.get('current_turn', {}).get('text', '') or ''
    context_turns = sample.get('context_turns', [])
    words = re.findall(r'\w+', text)
    out = dict(sample)
    out['num_context_turns'] = len(context_turns)
    out['num_query_words'] = len(words)
    out['current_turn_text_length'] = len(text.strip())
    out['is_filler'] = is_filler_text(text)
    return out


def is_short_query(sample: Dict) -> bool:
    words = re.findall(r'\w+', sample['current_turn']['text'])
    return len(words) < STAGE4_MIN_QUERY_WORDS


def balance_by_category(samples: List[Dict], target_count: int,
                        max_category_ratio: float = STAGE4_MAX_CATEGORY_RATIO) -> List[Dict]:
    if not samples:
        return []
    by_category = defaultdict(list)
    for s in samples:
        by_category[s['category']].append(s)
    category_counts = {c: len(lst) for c, lst in by_category.items()}
    total = sum(category_counts.values())
    max_per = int(target_count * max_category_ratio)
    min_per = max(1, target_count // (len(by_category) * 2))

    targets = {}
    for cat, avail in category_counts.items():
        prop = avail / total
        t = max(min_per, min(int(target_count * prop), max_per, avail))
        targets[cat] = t

    curr = sum(targets.values())
    if curr < target_count:
        for cat in sorted(targets, key=lambda c: category_counts[c], reverse=True):
            if curr >= target_count:
                break
            add = min(category_counts[cat] - targets[cat], target_count - curr)
            targets[cat] += add
            curr += add
    elif curr > target_count:
        for cat in sorted(targets, key=lambda c: targets[c], reverse=True):
            if curr <= target_count:
                break
            rm = min(targets[cat] - min_per, curr - target_count)
            targets[cat] -= rm
            curr -= rm

    result = []
    for cat, t in targets.items():
        lst = sorted(by_category[cat], key=lambda s: confidence_to_score(s['confidence']), reverse=True)
        result.extend(lst[:t])
    return result


def balance_speak_silent(samples: List[Dict], speak_ratio: float = STAGE4_SPEAK_RATIO) -> List[Dict]:
    speak = [s for s in samples if s['decision'] == 'SPEAK']
    silent = [s for s in samples if s['decision'] == 'SILENT']
    total = len(speak) + len(silent)
    if STAGE4_MAX_SAMPLES and total > STAGE4_MAX_SAMPLES:
        target = STAGE4_MAX_SAMPLES
    else:
        target = min(len(speak), len(silent)) * 2
    sp_target = min(int(target * speak_ratio), len(speak))
    sl_target = min(target - sp_target, len(silent))
    balanced_speak = balance_by_category(speak, sp_target)
    balanced_silent = balance_by_category(silent, sl_target)
    return balanced_speak + balanced_silent


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found. Run stage3_label_and_categorize.py first.")
        return

    with open(INPUT_FILE, 'r') as f:
        all_samples = json.load(f)

    print(f"Loaded {len(all_samples):,} samples")
    filtered = [s for s in all_samples if not is_low_quality(s)]
    print(f"After quality filter: {len(filtered):,}")

    advanced = []
    for s in tqdm(filtered, desc="Applying filters", unit=" sample"):
        if STAGE4_REMOVE_NO_CONTEXT and not has_context(s):
            continue
        if STAGE4_REMOVE_SHORT_QUERY and is_short_query(s):
            continue
        if STAGE4_GEMINI_CONFIDENCE_FILTER is not None and s.get('source') == 'gemini_inferred':
            if s.get('inference_confidence', 0) < STAGE4_GEMINI_CONFIDENCE_FILTER:
                continue
        advanced.append(s)

    if STAGE4_ENABLE_DEDUPLICATION:
        advanced = deduplicate_samples(advanced)
        print(f"After deduplication: {len(advanced):,}")

    balanced = balance_speak_silent(advanced)
    print(f"After balancing: {len(balanced):,}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    JSON_RUN_DIR.mkdir(exist_ok=True)

    output_path = OUTPUT_DIR / OUTPUT_FILE
    with open(output_path, 'w') as f:
        for s in tqdm(balanced, desc="Writing JSONL", unit=" sample"):
            f.write(json.dumps(add_filter_fields(s)) + '\n')
    print(f"✓ Saved {len(balanced):,} samples to {output_path} (with num_context_turns, num_query_words, etc.)")

    for idx, s in enumerate(tqdm(balanced, desc="Writing json_run", unit=" file"), start=1):
        with open(JSON_RUN_DIR / f"{idx}.json", 'w') as f:
            json.dump(add_filter_fields(s), f, indent=2)

    summary = {
        'total_samples': len(balanced),
        'speak_samples': sum(1 for s in balanced if s['decision'] == 'SPEAK'),
        'silent_samples': sum(1 for s in balanced if s['decision'] == 'SILENT'),
        'category_distribution': dict(Counter(s['category'] for s in balanced)),
        'filter_from_jsonl': {
            'num_context_turns': 'Filter with num_context_turns >= 1 to require context',
            'num_query_words': f'Filter with num_query_words >= N (e.g. {STAGE4_MIN_QUERY_WORDS}) for min query length',
            'current_turn_text_length': 'Filter by min character length',
            'is_filler': 'Exclude rows with is_filler true',
            'source': "Filter by origin (e.g. 'gemini_inferred' vs 'friends_explicit').",
            'inference_confidence': "For Gemini-inferred rows: filter with inference_confidence >= N (0-10). E.g. >= 6 for higher confidence.",
        },
    }
    with open(OUTPUT_DIR / 'filtering_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("Stage 4 complete.")


if __name__ == '__main__':
    main()
