#!/usr/bin/env python3
"""
Evaluate category quality: sample 25 per category from the Stage 4 output,
verify each sample's category assignment against the raw fields, and dump
the 100 samples to a human-readable JSON file for manual inspection.

Output:
  data_final/eval_100_samples.json   — 100 samples grouped by category
  (also prints a terminal summary with pass/fail per sample)
"""

import json
import random
from collections import defaultdict
from pathlib import Path

INPUT_FILE = Path(__file__).parent / "data_final" / "stage4_filtered_samples.jsonl"
OUTPUT_FILE = Path(__file__).parent / "data_final" / "eval_100_samples.json"

CATEGORIES = ["SPEAK_explicit", "SPEAK_implicit", "SILENT_ref", "SILENT_no_ref"]
SAMPLES_PER_CATEGORY = 25
SEED = 42


def load_by_category(path: Path) -> dict[str, list[dict]]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            by_cat[s.get("category", "unknown")].append(s)
    return by_cat


def verify_category(sample: dict) -> dict:
    """
    Re-derive the expected category from the raw fields and compare
    with the assigned category.  Returns a verdict dict.
    """
    decision = sample["decision"]
    target = sample["target_speaker"]
    target_is_addressed = sample["target_is_addressed"]
    target_spoke_next = sample["target_spoke_next"]
    context_turns = sample.get("context_turns", [])
    assigned = sample["category"]

    issues = []

    # ── Ground truth consistency ──
    expected_decision = "SPEAK" if target_spoke_next else "SILENT"
    if decision != expected_decision:
        issues.append(f"decision={decision} but target_spoke_next={target_spoke_next}")

    # ── Category re-derivation ──
    if decision == "SPEAK":
        expected_cat = "SPEAK_explicit" if target_is_addressed else "SPEAK_implicit"
    else:
        recent_speakers = [t["speaker"] for t in context_turns[-2:]]
        target_recent = target in recent_speakers
        if target_is_addressed or target_recent:
            expected_cat = "SILENT_ref"
        else:
            expected_cat = "SILENT_no_ref"

    if assigned != expected_cat:
        issues.append(f"assigned={assigned} but re-derived={expected_cat}")

    # ── Addressing sanity ──
    addressees = sample.get("addressees_in_current", [])
    if target_is_addressed and target not in addressees:
        issues.append("target_is_addressed=True but target not in addressees_in_current")
    if not target_is_addressed and target in addressees:
        issues.append("target_is_addressed=False but target IS in addressees_in_current")

    # ── Engaged-silence check (should have been filtered in stage4) ──
    if decision == "SILENT" and not target_is_addressed:
        recent_5 = context_turns[-5:]
        target_in_recent_5 = any(t["speaker"] == target for t in recent_5)
        total_target = sum(1 for t in context_turns if t["speaker"] == target)
        if not target_in_recent_5 and total_target < 2:
            issues.append("trivial silence: target not recent and <2 total turns in context")

    return {
        "pass": len(issues) == 0,
        "expected_category": expected_cat,
        "issues": issues,
    }


def format_sample_for_display(sample: dict, idx: int, verdict: dict) -> dict:
    """Create a compact, human-readable version of the sample for the JSON dump."""
    context_turns = sample.get("context_turns", [])
    context_display = [
        f"[{t['speaker']}]: {t['text'][:120]}{'…' if len(t['text']) > 120 else ''}"
        for t in context_turns[-5:]
    ]
    return {
        "sample_index": idx,
        "decision_point_id": sample["decision_point_id"],
        "meeting_id": sample["meeting_id"],
        "target_speaker": sample["target_speaker"],
        "all_speakers": sample["all_speakers"],
        "category": sample["category"],
        "decision": sample["decision"],
        "target_is_addressed": sample["target_is_addressed"],
        "target_spoke_next": sample["target_spoke_next"],
        "addressees_in_current": sample.get("addressees_in_current", []),
        "current_turn": {
            "speaker": sample["current_turn"]["speaker"],
            "text": sample["current_turn"]["text"][:300],
        },
        "context_last_5": context_display,
        "num_context_turns": len(context_turns),
        "verification": verdict,
    }


def main():
    print(f"Loading samples from {INPUT_FILE} …")
    by_cat = load_by_category(INPUT_FILE)

    print(f"\nCategory counts in full dataset:")
    for cat in CATEGORIES:
        print(f"  {cat}: {len(by_cat[cat]):,}")

    rng = random.Random(SEED)
    output = {}
    total_pass = 0
    total_fail = 0

    for cat in CATEGORIES:
        pool = by_cat[cat]
        n = min(SAMPLES_PER_CATEGORY, len(pool))
        chosen = rng.sample(pool, n)

        cat_pass = 0
        cat_fail = 0
        cat_display = []

        print(f"\n{'=' * 70}")
        print(f"  {cat}  ({n} samples)")
        print(f"{'=' * 70}")

        for i, sample in enumerate(chosen):
            verdict = verify_category(sample)
            status = "PASS" if verdict["pass"] else "FAIL"

            if verdict["pass"]:
                cat_pass += 1
            else:
                cat_fail += 1
                print(f"  [{i+1}] FAIL — {sample['decision_point_id']}")
                for issue in verdict["issues"]:
                    print(f"        {issue}")

            cat_display.append(format_sample_for_display(sample, i + 1, verdict))

        print(f"  Result: {cat_pass} pass, {cat_fail} fail")
        total_pass += cat_pass
        total_fail += cat_fail
        output[cat] = cat_display

    print(f"\n{'=' * 70}")
    print(f"OVERALL: {total_pass} pass, {total_fail} fail out of {total_pass + total_fail}")
    print(f"{'=' * 70}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Saved {sum(len(v) for v in output.values())} samples to {OUTPUT_FILE}")
    print(f"  Open the file to manually inspect conversation context + category assignment.")


if __name__ == "__main__":
    main()
