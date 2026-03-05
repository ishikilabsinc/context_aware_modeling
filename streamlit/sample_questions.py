"""
Samples up to 100 questions per category for human evaluation.
Source: Friends test dataset only.
Rules:
  - Skip samples with no context turns (at least 1 required)
  - Up to 100 per category; if fewer available, take all
  - Output: sampled_100.jsonl  (may contain up to 400 samples total)

Run standalone:
    python streamlit/sample_questions.py
or import and call sample_questions() from app.py.
"""

import json
import random
from collections import defaultdict, Counter
from pathlib import Path

BASE_DIR = Path(__file__).parent

FRIENDS_FILE = BASE_DIR / "test_json" / "test_samples_friends.jsonl"

CATEGORIES = ["SPEAK_explicit", "SPEAK_implicit", "SILENT_no_ref", "SILENT_ref"]
MAX_PER_CATEGORY = 100
OUTPUT_PATH = BASE_DIR / "sampled_100.jsonl"


def sample_questions(seed: int = 42, output_path: Path = OUTPUT_PATH) -> list[dict]:
    """
    Load Friends test JSONL, group by category, draw up to MAX_PER_CATEGORY
    samples per category (all available if fewer than MAX_PER_CATEGORY).
    Skips samples with no context turns or empty current turn.

    Returns the list of sampled dicts (with 'dataset' field added).
    """
    by_cat: dict[str, list[dict]] = defaultdict(list)

    with open(FRIENDS_FILE) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            # Drop samples with no context
            if not sample.get("context_turns"):
                continue
            # Drop samples with empty current turn
            if not sample.get("current_turn", {}).get("text", "").strip():
                continue
            sample["dataset"] = "Friends"
            cat = sample.get("category", "unknown")
            if cat in CATEGORIES:
                by_cat[cat].append(sample)

    rng = random.Random(seed)
    selected: list[dict] = []

    for cat in CATEGORIES:
        pool = by_cat[cat]
        n = min(MAX_PER_CATEGORY, len(pool))
        chosen = rng.sample(pool, n)
        selected.extend(chosen)
        print(f"  {cat}: {n} samples (available: {len(pool)})")

    rng.shuffle(selected)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        for s in selected:
            fh.write(json.dumps(s) + "\n")

    print(f"\nSaved {len(selected)} samples to {output_path}")
    return selected


if __name__ == "__main__":
    samples = sample_questions()
    cats = Counter(s["category"] for s in samples)
    print("\nCategory distribution:", dict(cats))
    print("Total:", sum(cats.values()))
