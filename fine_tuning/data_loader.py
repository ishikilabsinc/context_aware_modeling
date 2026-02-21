#!/usr/bin/env python3
"""
Load and format training/val data for LoRA fine-tuning. Uses config for paths.
Prompt format matches benchmarking/evaluate_baseline.py for fair train/eval comparison.
"""

import sys
from pathlib import Path
from typing import List, Dict
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import TRAIN_FILE, VAL_FILE, DATASET, BASE_DIR

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _get_system_prompt_for_training() -> str:
    from benchmarking.evaluate_baseline import _get_system_prompt_content
    return _get_system_prompt_content()


def estimate_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False, max_length=10000, truncation=True))


def _build_context_str_benchmark_style(
    context_turns: List[Dict],
    current_turn: Dict,
    tokenizer,
    max_tokens: int,
) -> str:
    """
    Build context string in the same format as benchmarking/evaluate_baseline.py
    (all_turns with last one marked [MOST RECENT - after this there was a pause]).
    Selects turns from the end to fit within max_tokens.
    """
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


def create_training_prompt(sample: Dict, tokenizer, max_length: int) -> str:
    """Build prompt in same format as benchmark format_sample_for_inference."""
    system_prompt = _get_system_prompt_for_training()
    context_turns = sample.get("context_turns", [])
    current_turn = sample.get("current_turn", {})
    target_speaker = sample.get("target_speaker", "?")
    decision = sample.get("decision", "UNKNOWN")
    if decision not in ("SPEAK", "SILENT"):
        decision = "SILENT" if decision.upper() == "SILENT" else "SPEAK"
    confidence = sample.get("confidence", "medium")
    reason = sample.get("reason", "")

    instruction = (
        f"You are playing the role of Speaker {target_speaker}. The conversation history above shows all utterances including the most recent one (marked as [MOST RECENT]). "
        "After that most recent utterance, there was a pause. Decide if you (Speaker {target_speaker}) should START TALKING or STAY SILENT now."
    )
    if current_turn:
        current_str = f"Speaker {current_turn.get('speaker', '?')}: {current_turn.get('text', '')}"
    else:
        current_str = "(No current utterance)"

    reply_format = (
        "Reply with your decision in this exact format: <reasoning>One sentence: ACTIVE PARTICIPANT or BYSTANDER, and who is addressed.</reasoning> "
        "<decision>SPEAK</decision> or <decision>SILENT</decision> <confidence>high</confidence> or <confidence>medium</confidence> or <confidence>low</confidence>"
    )
    output_part = f"<reasoning>{reason}</reasoning> <decision>{decision}</decision> <confidence>{confidence}</confidence>"

    system_tokens = estimate_tokens(system_prompt, tokenizer)
    instruction_tokens = estimate_tokens(instruction, tokenizer)
    current_block = f"MOST RECENT UTTERANCE (the previous utterance that just occurred): {current_str}"
    current_tokens = estimate_tokens(current_block, tokenizer)
    reply_tokens = estimate_tokens(reply_format, tokenizer)
    output_tokens = estimate_tokens(output_part, tokenizer)
    reserved = system_tokens + instruction_tokens + current_tokens + reply_tokens + output_tokens + 150
    available_context = max(0, max_length - reserved)
    context_str = _build_context_str_benchmark_style(
        context_turns, current_turn, tokenizer, available_context
    )

    prompt = f"""<|system|>{system_prompt}<|/system|>
<|instruction|>{instruction}<|/instruction|>
<|context|>{context_str}<|/context|>
<|current|>MOST RECENT UTTERANCE (the previous utterance that just occurred): {current_str}<|/current|>
{reply_format}
<reasoning>{reason}</reasoning> <decision>{decision}</decision> <confidence>{confidence}</confidence>"""
    return prompt


class TurnTakingDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer: AutoTokenizer, max_length: int = 2048, debug: bool = False):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.debug = debug
        self.debug_count = 0
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        prompt = create_training_prompt(sample, self.tokenizer, self.max_length)
        
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        if self.debug and self.debug_count < 3:
            decoded = self.tokenizer.decode(encoding["input_ids"][0], skip_special_tokens=False)
            print(f"[DEBUG] sample {idx} decision={sample.get('decision')} len={encoding['input_ids'].shape[1]} tokens first300={decoded[:300]!r}")
            self.debug_count += 1
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten(),
        }



import random
from torch.utils.data import Sampler

from utils.data_utils import load_samples

ALL_DATASETS = ["ami", "friends", "spgi"]


class BalancedBatchSampler(Sampler):
    """
    Yields batches of indices with roughly 50% SPEAK and 50% SILENT per batch
    so each batch has representation from both decisions.
    """
    def __init__(self, speak_indices: List[int], silent_indices: List[int], batch_size: int, shuffle: bool = True):
        self.speak_indices = speak_indices
        self.silent_indices = silent_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        half = max(1, batch_size // 2)
        self.half = half
        self.n_batches = min(len(speak_indices) // half, len(silent_indices) // half)

    def __iter__(self):
        speak = list(self.speak_indices)
        silent = list(self.silent_indices)
        if self.shuffle:
            random.shuffle(speak)
            random.shuffle(silent)
        half = self.half
        for i in range(self.n_batches):
            batch = speak[i * half : (i + 1) * half] + silent[i * half : (i + 1) * half]
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


class DistributedBalancedBatchSampler(Sampler):
    """
    Distributed version of BalancedBatchSampler: each rank gets a partition of
    SPEAK/SILENT indices and builds ~50/50 batches from that partition so
    no sample is seen by two ranks and each rank still gets balanced batches.
    """
    def __init__(
        self,
        speak_indices: List[int],
        silent_indices: List[int],
        batch_size: int,
        rank: int,
        world_size: int,
        shuffle: bool = True,
    ):
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        # Partition indices across ranks: this rank gets indices at position rank, rank+world_size, ...
        self.my_speak = [speak_indices[i] for i in range(rank, len(speak_indices), world_size)]
        self.my_silent = [silent_indices[i] for i in range(rank, len(silent_indices), world_size)]
        half = max(1, batch_size // 2)
        self.half = half
        self.n_batches = min(len(self.my_speak) // half, len(self.my_silent) // half)

    def __iter__(self):
        speak = list(self.my_speak)
        silent = list(self.my_silent)
        if self.shuffle:
            random.shuffle(speak)
            random.shuffle(silent)
        half = self.half
        for i in range(self.n_batches):
            batch = speak[i * half : (i + 1) * half] + silent[i * half : (i + 1) * half]
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
):
    if DATASET == "all":
        print("Loading training data from all datasets (ami, friends, spgi)...")
        train_samples = []
        for name in ALL_DATASETS:
            path = BASE_DIR / "data" / name / "train" / "train_samples.jsonl"
            if path.exists():
                part = load_samples(path)
                train_samples.extend(part)
                print(f"  {name}: {len(part):,} samples")
            else:
                print(f"  {name}: (file not found, skipping)")
        print(f"  Total training samples: {len(train_samples):,}")

        print("Loading validation data from all datasets...")
        val_samples = []
        for name in ALL_DATASETS:
            path = BASE_DIR / "data" / name / "val" / "val_samples.jsonl"
            if path.exists():
                part = load_samples(path)
                val_samples.extend(part)
                print(f"  {name}: {len(part):,} samples")
            else:
                print(f"  {name}: (file not found, skipping)")
        print(f"  Total validation samples: {len(val_samples):,}")
    else:
        print("Loading training data...")
        train_samples = load_samples(TRAIN_FILE)
        print(f"  Loaded {len(train_samples):,} training samples")
        print("Loading validation data...")
        val_samples = load_samples(VAL_FILE)
        print(f"  Loaded {len(val_samples):,} validation samples")

    print("Creating datasets...")
    train_dataset = TurnTakingDataset(train_samples, tokenizer, max_length, debug=debug)
    val_dataset = TurnTakingDataset(val_samples, tokenizer, max_length, debug=debug)
    print(f"  Train dataset: {len(train_dataset):,} samples")
    print(f"  Val dataset: {len(val_dataset):,} samples")

    batch_sampler = None
    if use_balanced_batches and batch_size is not None and batch_size >= 2:
        speak_idx = [i for i, s in enumerate(train_samples) if s.get("decision") == "SPEAK"]
        silent_idx = [i for i, s in enumerate(train_samples) if s.get("decision") == "SILENT"]
        half = max(1, batch_size // 2)
        if len(speak_idx) >= half and len(silent_idx) >= half:
            if rank is not None and world_size is not None:
                batch_sampler = DistributedBalancedBatchSampler(
                    speak_idx, silent_idx, batch_size, rank, world_size, shuffle=True
                )
                n_batches = len(batch_sampler)
                if rank == 0:
                    print(f"  Distributed balanced batch sampler (world_size={world_size}): {n_batches:,} batches/rank (~50% SPEAK / 50% SILENT per batch)")
            else:
                batch_sampler = BalancedBatchSampler(speak_idx, silent_idx, batch_size, shuffle=True)
                n_batches = len(batch_sampler)
                print(f"  Balanced batch sampler: {n_batches:,} batches (~50% SPEAK / 50% SILENT per batch)")
        else:
            print("  Skipping balanced batches (not enough SPEAK or SILENT samples); using default shuffling")

    return train_dataset, val_dataset, batch_sampler



def data_collator(features: List[Dict]):
    import torch
    
    batch = {}
    batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
    batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
    batch["labels"] = torch.stack([f["labels"] for f in features])
    
    return batch
