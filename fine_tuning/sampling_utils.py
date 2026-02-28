"""Shared stratified subsampling for SPGI (used by data_loader and generate_reasoning). No heavy deps."""

import random
from typing import Dict, List

# Target sizes for equal sampling (dataset=all): AMI and Friends kept as-is, SPGI subsampled to this.
EQUAL_SAMPLING_SPGI_TARGET = 11_000
EQUAL_SAMPLING_SEED = 42


def stratified_subsample_spgi(
    samples: List[Dict],
    target_total: int = EQUAL_SAMPLING_SPGI_TARGET,
    seed: int = EQUAL_SAMPLING_SEED,
) -> List[Dict]:
    """Subsample SPGI to target_total preserving 50/50 SPEAK/SILENT and category proportions.
    Categories: SPEAK_explicit, SPEAK_implicit, SILENT_ref, SILENT_no_ref.
    """
    if len(samples) <= target_total:
        return list(samples)
    rng = random.Random(seed)
    speak = [s for s in samples if s.get("decision") == "SPEAK"]
    silent = [s for s in samples if s.get("decision") == "SILENT"]
    n_speak_target = target_total // 2
    n_silent_target = target_total - n_speak_target

    def _stratified_sample(pool: List[Dict], n_want: int) -> List[Dict]:
        if not pool or n_want <= 0:
            return []
        if n_want >= len(pool):
            return list(pool)
        total = len(pool)
        by_cat: Dict[str, List[Dict]] = {}
        for s in pool:
            cat = s.get("category") or "unknown"
            by_cat.setdefault(cat, []).append(s)
        # Proportional target per category, capped by available
        sampled: List[Dict] = []
        for cat, group in by_cat.items():
            n_cat = min(len(group), max(0, int(round(n_want * len(group) / total))))
            if n_cat > 0:
                sampled.extend(rng.sample(group, n_cat))
        # Fix size: add or remove to reach n_want (preserves approximate balance)
        if len(sampled) < n_want:
            remaining = [s for s in pool if s not in sampled]
            sampled.extend(rng.sample(remaining, min(n_want - len(sampled), len(remaining))))
        elif len(sampled) > n_want:
            sampled = rng.sample(sampled, n_want)
        return sampled

    if len(speak) >= n_speak_target and len(silent) >= n_silent_target:
        sampled_speak = _stratified_sample(speak, n_speak_target)
        sampled_silent = _stratified_sample(silent, n_silent_target)
    elif len(speak) < n_speak_target:
        sampled_speak = list(speak)
        sampled_silent = _stratified_sample(silent, min(len(silent), target_total - len(sampled_speak)))
    else:
        sampled_silent = list(silent)
        sampled_speak = _stratified_sample(speak, min(len(speak), target_total - len(sampled_silent)))
    result = sampled_speak + sampled_silent
    rng.shuffle(result)
    return result[:target_total]
