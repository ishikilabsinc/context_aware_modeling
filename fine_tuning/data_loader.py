#!/usr/bin/env python3
"""Load and format training/val data for LoRA fine-tuning. Supports two modes: decision_only (response = <decision>SPEAK|SILENT</decision>) and cot (response = <reasoning>...</reasoning> + <decision>...</decision> + <confidence>...). CoT uses train_samples_with_reasoning.jsonl when present (see generate_reasoning.py)."""

import sys
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

try:
    from .config import TRAIN_FILE, VAL_FILE, DATASET, BASE_DIR
except ImportError:
    from config import TRAIN_FILE, VAL_FILE, DATASET, BASE_DIR

TRAINING_MODE_DECISION_ONLY = "decision_only"
TRAINING_MODE_COT = "cot"
TRAINING_MODES = [TRAINING_MODE_DECISION_ONLY, TRAINING_MODE_COT]

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _get_system_prompt_for_training() -> str:
    from benchmarking.evaluate_baseline import SYSTEM_PROMPT
    return SYSTEM_PROMPT.strip()


def estimate_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False, max_length=10000, truncation=True))


def _build_context_str_benchmark_style(
    context_turns: List[Dict],
    current_turn: Dict,
    tokenizer,
    max_tokens: int,
) -> str:
    all_turns = context_turns + ([current_turn] if current_turn else [])
    if not all_turns:
        return "(No previous context)"
    if max_tokens <= 0:
        return "(No previous context)"
    selected = []
    current_tokens = 0
    for turn in reversed(all_turns):
        line = f"Speaker {turn.get('speaker', '?')}: {turn.get('text', '')}"
        if current_turn and turn == current_turn:
            line += "  [MOST RECENT - after this there was a pause]"
        turn_tokens = estimate_tokens(line + "\n", tokenizer)
        if current_tokens + turn_tokens <= max_tokens:
            selected.insert(0, (turn, line))
            current_tokens += turn_tokens
        else:
            break
    if not selected:
        return "(No previous context)"
    return "\n".join(line for _, line in selected)


def create_training_prompt(sample: Dict, tokenizer, max_length: int = None, training_mode: str = TRAINING_MODE_DECISION_ONLY) -> str:
    if training_mode not in TRAINING_MODES:
        raise ValueError(f"training_mode must be one of {TRAINING_MODES}, got {training_mode!r}")

    system_prompt = _get_system_prompt_for_training()
    context_turns = sample.get("context_turns", [])
    current_turn = sample.get("current_turn", {})
    target_speaker = sample.get("target_speaker", "?")

    instruction = (
        f"You are playing the role of Speaker {target_speaker}. The conversation history above shows all utterances including the most recent one (marked as [MOST RECENT]). "
        f"After that most recent utterance, there was a pause. Decide if you (Speaker {target_speaker}) should START TALKING or STAY SILENT now."
    )
    if current_turn:
        current_str = f"Speaker {current_turn.get('speaker', '?')}: {current_turn.get('text', '')}"
    else:
        current_str = "(No current utterance)"

    if training_mode == TRAINING_MODE_DECISION_ONLY:
        reply_format = (
            "Reply with your decision in this exact format: <decision>SPEAK</decision> or <decision>SILENT</decision>"
        )
        output_part = "<decision>SPEAK</decision>"
    else:
        reply_format = (
            "Reply with: <reasoning>one sentence explaining why</reasoning>\n<decision>SPEAK or SILENT</decision>\n<confidence>high, medium, or low</confidence>"
        )
        output_part = "<reasoning>Placeholder.</reasoning>\n<decision>SPEAK</decision>\n<confidence>high</confidence>"

    output_tokens = estimate_tokens(output_part, tokenizer)
    system_tokens = estimate_tokens(system_prompt, tokenizer)
    instruction_tokens = estimate_tokens(instruction, tokenizer)
    current_block = f"MOST RECENT UTTERANCE (the previous utterance that just occurred): {current_str}"
    current_tokens = estimate_tokens(current_block, tokenizer)
    reply_tokens = estimate_tokens(reply_format, tokenizer)
    reserved = system_tokens + instruction_tokens + current_tokens + reply_tokens + output_tokens + 150
    available_context = 10**6 if (max_length is None or max_length <= 0) else max(0, max_length - reserved)
    context_str = _build_context_str_benchmark_style(
        context_turns, current_turn, tokenizer, available_context
    )

    prompt = f"""<|system|>{system_prompt}<|/system|>
<|instruction|>{instruction}<|/instruction|>
<|context|>{context_str}<|/context|>
<|current|>MOST RECENT UTTERANCE (the previous utterance that just occurred): {current_str}<|/current|>
{reply_format}"""
    return prompt


class TurnTakingDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer: AutoTokenizer, max_length: int = 2048, debug: bool = False, training_mode: str = TRAINING_MODE_DECISION_ONLY):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.debug = debug
        self.training_mode = training_mode
        self.no_truncation = max_length is None or max_length <= 0

    def __len__(self):
        return len(self.samples)

    def _getitem_decision_only(self, idx):
        sample = self.samples[idx]
        decision = sample.get("decision", "SILENT")
        if decision not in ("SPEAK", "SILENT"):
            decision = "SILENT" if str(decision).upper() == "SILENT" else "SPEAK"
        response_str = "<decision>SPEAK</decision>" if decision == "SPEAK" else "<decision>SILENT</decision>"
        prompt = create_training_prompt(sample, self.tokenizer, self.max_length, training_mode=TRAINING_MODE_DECISION_ONLY)
        prompt_content = prompt.rstrip()
        full_text = prompt_content + "\n" + response_str
        response_start_char = len(prompt_content) + 1

        if self.no_truncation:
            encoding = self.tokenizer(
                full_text,
                truncation=False,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            input_ids = encoding["input_ids"].flatten().clone()
            attention_mask = encoding["attention_mask"].flatten().clone()
            offset_mapping = encoding.get("offset_mapping")
            offset_mapping = offset_mapping.flatten(0, 1) if offset_mapping is not None else None
        else:
            old_side = getattr(self.tokenizer, "truncation_side", "right")
            self.tokenizer.truncation_side = "left"
            try:
                encoding = self.tokenizer(
                    full_text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                    return_offsets_mapping=True,
                )
            finally:
                self.tokenizer.truncation_side = old_side
            input_ids = encoding["input_ids"].flatten().clone()
            attention_mask = encoding["attention_mask"].flatten().clone()
            offset_mapping = encoding.get("offset_mapping")
            offset_mapping = offset_mapping.flatten(0, 1) if offset_mapping is not None else None

        labels = torch.full_like(input_ids, -100, dtype=torch.long)
        if offset_mapping is not None:
            for i in range(offset_mapping.shape[0]):
                s, e = offset_mapping[i].tolist()
                if s == 0 and e == 0:
                    continue
                if s >= response_start_char:
                    labels[i] = input_ids[i].item()
        else:
            n_prompt = len(self.tokenizer.encode(prompt_content + "\n", add_special_tokens=True))
            response_start_idx = min(n_prompt, input_ids.shape[0])
            labels[response_start_idx:] = input_ids[response_start_idx:].clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _getitem_cot(self, idx):
        sample = self.samples[idx]
        decision = sample.get("decision", "SILENT")
        if decision not in ("SPEAK", "SILENT"):
            decision = "SILENT" if str(decision).upper() == "SILENT" else "SPEAK"
        reasoning = sample.get("reasoning", "No reasoning provided.").strip()
        if not reasoning:
            reasoning = "No reasoning provided."
        confidence = sample.get("confidence", "medium")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"
        response_str = f"<reasoning>{reasoning}</reasoning>\n<decision>{decision}</decision>\n<confidence>{confidence}</confidence>"
        prompt = create_training_prompt(sample, self.tokenizer, self.max_length, training_mode=TRAINING_MODE_COT)
        prompt_content = prompt.rstrip()
        full_text = prompt_content + "\n" + response_str
        response_start_char = len(prompt_content) + 1

        if self.no_truncation:
            encoding = self.tokenizer(
                full_text,
                truncation=False,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            input_ids = encoding["input_ids"].flatten().clone()
            attention_mask = encoding["attention_mask"].flatten().clone()
            offset_mapping = encoding.get("offset_mapping")
            offset_mapping = offset_mapping.flatten(0, 1) if offset_mapping is not None else None
        else:
            old_side = getattr(self.tokenizer, "truncation_side", "right")
            self.tokenizer.truncation_side = "left"
            try:
                encoding = self.tokenizer(
                    full_text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                    return_offsets_mapping=True,
                )
            finally:
                self.tokenizer.truncation_side = old_side
            input_ids = encoding["input_ids"].flatten().clone()
            attention_mask = encoding["attention_mask"].flatten().clone()
            offset_mapping = encoding.get("offset_mapping")
            offset_mapping = offset_mapping.flatten(0, 1) if offset_mapping is not None else None

        labels = torch.full_like(input_ids, -100, dtype=torch.long)
        if offset_mapping is not None:
            for i in range(offset_mapping.shape[0]):
                s, e = offset_mapping[i].tolist()
                if s == 0 and e == 0:
                    continue
                if s >= response_start_char:
                    labels[i] = input_ids[i].item()
        else:
            n_prompt = len(self.tokenizer.encode(prompt_content + "\n", add_special_tokens=True))
            response_start_idx = min(n_prompt, input_ids.shape[0])
            labels[response_start_idx:] = input_ids[response_start_idx:].clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __getitem__(self, idx):
        if self.training_mode == TRAINING_MODE_DECISION_ONLY:
            return self._getitem_decision_only(idx)
        if self.training_mode == TRAINING_MODE_COT:
            return self._getitem_cot(idx)
        raise ValueError(f"training_mode must be one of {TRAINING_MODES}, got {self.training_mode!r}")


def verify_response_boundary(
    tokenizer: AutoTokenizer,
    dataset: "TurnTakingDataset",
    num_samples: int = 3,
) -> tuple:
    errors = []
    for idx in range(min(num_samples, len(dataset))):
        item = dataset[idx]
        labels = item["labels"]
        input_ids = item["input_ids"]
        label_positions = (labels != -100).nonzero(as_tuple=True)[0]
        if label_positions.numel() == 0:
            errors.append(f"Sample {idx}: no unmasked tokens")
            continue
        start_pos = int(label_positions[0].item())
        # Decode first few response tokens to check they form "<decision>..."
        num_tokens = min(12, labels.shape[0] - start_pos)
        token_ids = [int(input_ids[start_pos + i].item()) for i in range(num_tokens)]
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        decoded_stripped = decoded.strip()
        if not (decoded_stripped.startswith("<decision>") or decoded_stripped.startswith("<decision") or decoded_stripped.startswith("<")):
            errors.append(
                f"Sample {idx}: first unmasked token(s) at pos {start_pos} decode to {decoded!r}; "
                f"expected to start with '<' (start of <decision>)"
            )
    return (len(errors) == 0, errors)


def _write_sample_inputs_outputs(
    tokenizer,
    train_dataset: "TurnTakingDataset",
    val_dataset: "TurnTakingDataset",
    train_samples: List[Dict],
    val_samples: List[Dict],
    max_length: int,
    filepath: Path,
    n_per_split: int = 4,
    training_mode: str = TRAINING_MODE_DECISION_ONLY,
) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    no_truncation = max_length is None or max_length <= 0
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"training_mode = {training_mode}\n")
        if no_truncation:
            f.write("max_length = None (no truncation; full examples)\n")
        else:
            f.write(f"max_length = {max_length}\n")
        f.write("Each sample shows: full tokenized input (what the model sees), target (response tokens; prompt tokens masked with -100), and whether the prompt was truncated.\n")
        f.write("=" * 80 + "\n\n")
        for name, dataset, samples in [
            ("TRAIN", train_dataset, train_samples),
            ("VAL", val_dataset, val_samples),
        ]:
            n = min(n_per_split, len(samples))
            for i in range(n):
                sample = samples[i]
                prompt_text = create_training_prompt(sample, tokenizer, max_length, training_mode=training_mode)
                full_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
                n_tokens_before = full_enc["input_ids"].shape[1]
                truncated = False if no_truncation else (n_tokens_before > max_length)
                item = dataset[i]
                input_ids = item["input_ids"]
                labels = item["labels"]
                full_text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
                label_positions = (labels != -100).nonzero(as_tuple=True)[0]
                if label_positions.numel() > 0:
                    start_pos = label_positions[0].item()
                    end_pos = label_positions[-1].item()
                    response_tokens = [int(labels[i].item()) for i in range(start_pos, end_pos + 1)]
                    response_str = tokenizer.decode(response_tokens)
                    label_info = f"Positions {start_pos}-{end_pos}: {response_str!r} ({label_positions.numel()} response tokens)"
                    n_show = min(8, label_positions.numel())
                    first_few = [int(input_ids[start_pos + k].item()) for k in range(n_show)]
                    first_few_decoded = tokenizer.decode(first_few, skip_special_tokens=False)
                    boundary_note = f"Boundary check: first unmasked token(s) decoded: {first_few_decoded!r} (expected: start of '<decision>')"
                else:
                    label_info = "No label position (all -100); answer block may be truncated"
                    boundary_note = "Boundary check: N/A (no unmasked tokens)"
                decision = sample.get("decision", "?")
                f.write(f"\n{'='*80}\n")
                f.write(f"{name} sample {i}  |  decision={decision}  |  tokens_used={len(input_ids)}  |  tokens_before_truncation={n_tokens_before}  |  truncated={truncated}\n")
                f.write(f"{'='*80}\n")
                f.write(f"TARGET (response tokens): {label_info}\n")
                f.write(f"{boundary_note}\n")
                f.write(f"\nFULL MODEL INPUT (decoded, {len(input_ids)} tokens):\n")
                f.write("-" * 80 + "\n")
                f.write(full_text)
                f.write("\n" + "-" * 80 + "\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("BOUNDARY VERIFICATION (first unmasked token should be '<' of <decision>)\n")
        f.write("-" * 80 + "\n")
        for name, dataset in [("TRAIN", train_dataset), ("VAL", val_dataset)]:
            ok, errs = verify_response_boundary(tokenizer, dataset, num_samples=3)
            if ok:
                f.write(f"{name}: OK (3 samples checked)\n")
            else:
                for e in errs:
                    f.write(f"{name}: {e}\n")
        f.write("=" * 80 + "\n")
    print(f"[debug] Full model input/output for {n_per_split} train + {n_per_split} val samples written to: {filepath}")


import random
from torch.utils.data import Sampler

from utils.data_utils import load_samples, filter_samples_with_context

ALL_DATASETS = ["ami", "friends", "spgi"]

# Target sizes for equal sampling (dataset=all): AMI and Friends kept as-is, SPGI subsampled to this.
EQUAL_SAMPLING_SPGI_TARGET = 11_000
EQUAL_SAMPLING_SEED = 42


def _stratified_subsample_spgi(
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


class BalancedBatchSampler(Sampler):
    def __init__(
        self,
        speak_indices: List[int],
        silent_indices: List[int],
        batch_size: int,
        shuffle: bool = True,
        silent_ratio_in_batch: float = 0.5,
    ):
        self.speak_indices = speak_indices
        self.silent_indices = silent_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        n_silent = max(1, min(batch_size - 1, round(batch_size * silent_ratio_in_batch)))
        n_speak = batch_size - n_silent
        self.n_speak = n_speak
        self.n_silent = n_silent
        self.n_batches = min(len(speak_indices) // n_speak, len(silent_indices) // n_silent)

    def __iter__(self):
        speak = list(self.speak_indices)
        silent = list(self.silent_indices)
        if self.shuffle:
            random.shuffle(speak)
            random.shuffle(silent)
        n_speak, n_silent = self.n_speak, self.n_silent
        for i in range(self.n_batches):
            batch = speak[i * n_speak : (i + 1) * n_speak] + silent[i * n_silent : (i + 1) * n_silent]
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


class DistributedBalancedBatchSampler(Sampler):
    def __init__(
        self,
        speak_indices: List[int],
        silent_indices: List[int],
        batch_size: int,
        rank: int,
        world_size: int,
        shuffle: bool = True,
        silent_ratio_in_batch: float = 0.5,
    ):
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        n_silent = max(1, min(batch_size - 1, round(batch_size * silent_ratio_in_batch)))
        n_speak = batch_size - n_silent
        self.n_speak = n_speak
        self.n_silent = n_silent
        self.my_speak = [speak_indices[i] for i in range(rank, len(speak_indices), world_size)]
        self.my_silent = [silent_indices[i] for i in range(rank, len(silent_indices), world_size)]
        self.n_batches = min(len(self.my_speak) // n_speak, len(self.my_silent) // n_silent)

    def __iter__(self):
        speak = list(self.my_speak)
        silent = list(self.my_silent)
        if self.shuffle:
            random.shuffle(speak)
            random.shuffle(silent)
        n_speak, n_silent = self.n_speak, self.n_silent
        for i in range(self.n_batches):
            batch = speak[i * n_speak : (i + 1) * n_speak] + silent[i * n_silent : (i + 1) * n_silent]
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


def prepare_datasets(
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
    debug: bool = False,
    batch_size: int = None,
    use_balanced_batches: bool = True,
    rank: int = None,
    world_size: int = None,
    silent_ratio_in_batch: float = 0.5,
    train_fraction: float = 1.0,
    val_fraction: float = 1.0,
    debug_sample_io_path: str = None,
    training_mode: str = TRAINING_MODE_DECISION_ONLY,
    equal_sampling: bool = False,
    filter_no_context: bool = True,
):
    if DATASET == "all":
        print("Loading training data from all datasets (ami, friends, spgi)...")
        train_parts: Dict[str, List[Dict]] = {}
        for name in ALL_DATASETS:
            if training_mode == TRAINING_MODE_COT:
                path = BASE_DIR / "data" / name / "train" / "train_samples_with_reasoning.jsonl"
                if not path.exists():
                    path = BASE_DIR / "data" / name / "train" / "train_samples.jsonl"
                    if path.exists() and (rank is None or rank == 0):
                        print(f"  {name}: train_samples_with_reasoning.jsonl not found; using train_samples.jsonl (reasoning will be placeholder)")
            else:
                path = BASE_DIR / "data" / name / "train" / "train_samples.jsonl"
            if path.exists():
                part = load_samples(path)
                if training_mode == TRAINING_MODE_COT:
                    for s in part:
                        if "reasoning" not in s:
                            s["reasoning"] = "No reasoning provided."
                        if s.get("confidence") not in ("high", "medium", "low"):
                            s["confidence"] = s.get("confidence") or "medium"
                train_parts[name] = part
                print(f"  {name}: {len(part):,} samples")
            else:
                print(f"  {name}: (file not found, skipping)")
                train_parts[name] = []
        if equal_sampling and len(train_parts.get("spgi", [])) > EQUAL_SAMPLING_SPGI_TARGET:
            spgi_train = _stratified_subsample_spgi(
                train_parts["spgi"], target_total=EQUAL_SAMPLING_SPGI_TARGET, seed=EQUAL_SAMPLING_SEED
            )
            train_parts["spgi"] = spgi_train
            if rank is None or rank == 0:
                print(f"  Equal sampling: SPGI train subsampled to {len(spgi_train):,} (stratified 50/50 SPEAK/SILENT, category-proportional)")
        train_samples = []
        for name in ALL_DATASETS:
            train_samples.extend(train_parts.get(name, []))
        print(f"  Total training samples: {len(train_samples):,}")

        print("Loading validation data from all datasets...")
        val_samples = []
        for name in ALL_DATASETS:
            path = BASE_DIR / "data" / name / "val" / "val_samples.jsonl"
            if path.exists():
                part = load_samples(path)
                if training_mode == TRAINING_MODE_COT:
                    for s in part:
                        if "reasoning" not in s:
                            s["reasoning"] = "No reasoning provided."
                        if s.get("confidence") not in ("high", "medium", "low"):
                            s["confidence"] = s.get("confidence") or "medium"
                val_samples.extend(part)
                print(f"  {name}: {len(part):,} samples")
            else:
                print(f"  {name}: (file not found, skipping)")
        print(f"  Total validation samples: {len(val_samples):,}")
    else:
        print("Loading training data...")
        if training_mode == TRAINING_MODE_COT:
            cot_train = BASE_DIR / "data" / DATASET / "train" / "train_samples_with_reasoning.jsonl"
            train_path = cot_train if cot_train.exists() else TRAIN_FILE
            if train_path == TRAIN_FILE and cot_train != TRAIN_FILE and (rank is None or rank == 0):
                print(f"  train_samples_with_reasoning.jsonl not found; using train_samples.jsonl (reasoning will be placeholder)")
            train_samples = load_samples(train_path)
            if training_mode == TRAINING_MODE_COT:
                for s in train_samples:
                    if "reasoning" not in s:
                        s["reasoning"] = "No reasoning provided."
                    if s.get("confidence") not in ("high", "medium", "low"):
                        s["confidence"] = s.get("confidence") or "medium"
        else:
            train_samples = load_samples(TRAIN_FILE)
        print(f"  Loaded {len(train_samples):,} training samples")
        print("Loading validation data...")
        if training_mode == TRAINING_MODE_COT:
            cot_val = BASE_DIR / "data" / DATASET / "val" / "val_samples_with_reasoning.jsonl"
            val_path = cot_val if cot_val.exists() else VAL_FILE
            val_samples = load_samples(val_path)
            for s in val_samples:
                if "reasoning" not in s:
                    s["reasoning"] = "No reasoning provided."
                if s.get("confidence") not in ("high", "medium", "low"):
                    s["confidence"] = s.get("confidence") or "medium"
        else:
            val_samples = load_samples(VAL_FILE)
        print(f"  Loaded {len(val_samples):,} validation samples")

    if filter_no_context:
        n_train_before, n_val_before = len(train_samples), len(val_samples)
        train_samples = filter_samples_with_context(train_samples)
        val_samples = filter_samples_with_context(val_samples)
        if rank is None or rank == 0:
            print(f"  Filtered to samples with context_turns: train {len(train_samples):,} (removed {n_train_before - len(train_samples):,}), val {len(val_samples):,} (removed {n_val_before - len(val_samples):,})")

    # Optionally use a subset for faster pipeline debugging (reproducible with seed 42)
    if train_fraction < 1.0 and len(train_samples) > 0:
        n_train = max(1, int(len(train_samples) * train_fraction))
        rng = random.Random(42)
        train_samples = rng.sample(train_samples, min(n_train, len(train_samples)))
        if rank is None or rank == 0:
            print(f"  Using subset: {len(train_samples):,} train samples ({train_fraction:.0%})")
    if val_fraction < 1.0 and len(val_samples) > 0:
        n_val = max(1, int(len(val_samples) * val_fraction))
        rng = random.Random(43)  # different seed so val subset is independent
        val_samples = rng.sample(val_samples, min(n_val, len(val_samples)))
        if rank is None or rank == 0:
            print(f"  Using subset: {len(val_samples):,} val samples ({val_fraction:.0%})")

    print("Creating datasets...")
    print(f"  Training mode: {training_mode}")
    train_dataset = TurnTakingDataset(train_samples, tokenizer, max_length, debug=debug, training_mode=training_mode)
    val_dataset = TurnTakingDataset(val_samples, tokenizer, max_length, debug=debug, training_mode=training_mode)
    print(f"  Train dataset: {len(train_dataset):,} samples")
    print(f"  Val dataset: {len(val_dataset):,} samples")

    if debug and (rank is None or rank == 0) and debug_sample_io_path is not None:
        _write_sample_inputs_outputs(
            tokenizer, train_dataset, val_dataset, train_samples, val_samples,
            max_length, Path(debug_sample_io_path), n_per_split=4, training_mode=training_mode,
        )

    batch_sampler = None
    if use_balanced_batches and batch_size is not None and batch_size >= 2:
        speak_idx = [i for i, s in enumerate(train_samples) if s.get("decision") == "SPEAK"]
        silent_idx = [i for i, s in enumerate(train_samples) if s.get("decision") == "SILENT"]
        n_silent = max(1, min(batch_size - 1, round(batch_size * silent_ratio_in_batch)))
        n_speak = batch_size - n_silent
        if len(speak_idx) >= n_speak and len(silent_idx) >= n_silent:
            if rank is not None and world_size is not None:
                batch_sampler = DistributedBalancedBatchSampler(
                    speak_idx, silent_idx, batch_size, rank, world_size, shuffle=True,
                    silent_ratio_in_batch=silent_ratio_in_batch,
                )
                n_batches = len(batch_sampler)
                if rank == 0:
                    pct = int(round(100 * silent_ratio_in_batch))
                    print(f"  Distributed balanced batch sampler (world_size={world_size}): {n_batches:,} batches/rank (~{pct}% SILENT / {100-pct}% SPEAK per batch)")
            else:
                batch_sampler = BalancedBatchSampler(
                    speak_idx, silent_idx, batch_size, shuffle=True,
                    silent_ratio_in_batch=silent_ratio_in_batch,
                )
                n_batches = len(batch_sampler)
                pct = int(round(100 * silent_ratio_in_batch))
                print(f"  Balanced batch sampler: {n_batches:,} batches (~{pct}% SILENT / {100-pct}% SPEAK per batch)")
        else:
            print("  Skipping balanced batches (not enough SPEAK or SILENT samples); using default shuffling")

    return train_dataset, val_dataset, batch_sampler



def make_data_collator(tokenizer, max_length_cap: int = 8192):
    def data_collator(features: List[Dict]):
        import torch
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        lengths = [f["input_ids"].shape[0] for f in features]
        same_length = len(set(lengths)) == 1
        if same_length:
            batch = {
                "input_ids": torch.stack([f["input_ids"] for f in features]),
                "attention_mask": torch.stack([f["attention_mask"] for f in features]),
                "labels": torch.stack([f["labels"] for f in features]),
            }
            return batch

        max_len = min(max(lengths), max_length_cap)
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        for f in features:
            seq_len = f["input_ids"].shape[0]
            if seq_len > max_len:
                start = seq_len - max_len
                inp = f["input_ids"][start:]
                att = f["attention_mask"][start:]
                lab = f["labels"][start:]
            else:
                inp = f["input_ids"].squeeze(0) if f["input_ids"].dim() > 1 else f["input_ids"]
                att = f["attention_mask"].squeeze(0) if f["attention_mask"].dim() > 1 else f["attention_mask"]
                lab = f["labels"].squeeze(0) if f["labels"].dim() > 1 else f["labels"]
            pad_len = max_len - inp.shape[0]
            padded_input_ids.append(
                torch.cat([inp, torch.full((pad_len,), pad_id, dtype=inp.dtype)])
            )
            padded_attention_mask.append(
                torch.cat([att, torch.zeros(pad_len, dtype=att.dtype)])
            )
            padded_labels.append(
                torch.cat([lab, torch.full((pad_len,), -100, dtype=lab.dtype)])
            )
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }
        return batch

    return data_collator
