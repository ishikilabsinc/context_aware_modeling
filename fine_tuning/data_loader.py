#!/usr/bin/env python3
"""
Load and format training/val data for LoRA fine-tuning. Uses config for paths.
Prompt format matches benchmarking/evaluate_baseline.py. Training objective: causal LM
with loss only on the decision token (SPEAK or SILENT) in the answer block.
"""

import sys
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import TRAIN_FILE, VAL_FILE, DATASET, BASE_DIR

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _get_system_prompt_for_training() -> str:
    """Same SYSTEM_PROMPT as benchmarking; used once so more tokens remain for context."""
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
    """Build context string in benchmarking format; selects turns from the end to fit within max_tokens when max_tokens > 0."""
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


def create_training_prompt(sample: Dict, tokenizer, max_length: int = None) -> str:
    """Build prompt in same format as benchmark format_sample_for_inference.
    If max_length is 0 or None, use full context (no truncation)."""
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
    # No truncation: use full context when max_length is 0 or None
    available_context = 10**6 if (max_length is None or max_length <= 0) else max(0, max_length - reserved)
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
        # No truncation: use full example (variable-length batches; collator will pad)
        self.no_truncation = max_length is None or max_length <= 0

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        prompt = create_training_prompt(sample, self.tokenizer, self.max_length)
        
        if self.no_truncation:
            # Full example: no truncation, no padding (collator pads to longest in batch)
            encoding = self.tokenizer(
                prompt,
                truncation=False,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].flatten().clone()
            attention_mask = encoding["attention_mask"].flatten().clone()
        else:
            # Truncate from the left so the answer block at the end is always kept.
            # truncation_side is set on the tokenizer (not all transformers versions accept it as a kwarg).
            old_side = getattr(self.tokenizer, "truncation_side", "right")
            self.tokenizer.truncation_side = "left"
            try:
                encoding = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
            finally:
                self.tokenizer.truncation_side = old_side
            input_ids = encoding["input_ids"].flatten().clone()
            attention_mask = encoding["attention_mask"].flatten().clone()
        # Decision-token-only loss: only the SPEAK/SILENT token position gets a non-ignore label.
        # Pinpoint the *answer* block (last "<reasoning>...</reasoning> <decision>SPEAK|SILENT</decision>")
        # so we never use the format line's SPEAK/SILENT (which would make every sample one class).
        speak_ids = self.tokenizer.encode("SPEAK", add_special_tokens=False)
        silent_ids = self.tokenizer.encode("SILENT", add_special_tokens=False)
        speak_id = int(speak_ids[0]) if speak_ids else -1
        silent_id = int(silent_ids[0]) if silent_ids else -1
        open_reasoning_ids = self.tokenizer.encode("<reasoning>", add_special_tokens=False)
        open_reasoning_first_id = int(open_reasoning_ids[0]) if open_reasoning_ids else -1
        end_reasoning_ids = self.tokenizer.encode("</reasoning>", add_special_tokens=False)
        end_reasoning_last_id = int(end_reasoning_ids[-1]) if end_reasoning_ids else -1
        labels = torch.full_like(input_ids, -100, dtype=torch.long)
        # Answer block = after the last "<reasoning>" (format has one, answer has one; we want the answer)
        answer_start = -1
        for j in range(input_ids.shape[0] - 1, -1, -1):
            if input_ids[j].item() == open_reasoning_first_id:
                answer_start = j
                break
        start_after = 0
        if answer_start >= 0:
            for j in range(answer_start, input_ids.shape[0]):
                if input_ids[j].item() == end_reasoning_last_id:
                    start_after = j + 1
                    break
        for j in range(start_after, input_ids.shape[0]):
            tok = input_ids[j].item()
            if tok == speak_id or tok == silent_id:
                labels[j] = tok
                break
        else:
            # Fallback: answer truncated; use last occurrence (may be format line)
            for j in range(input_ids.shape[0] - 1, -1, -1):
                tok = input_ids[j].item()
                if tok == speak_id or tok == silent_id:
                    labels[j] = tok
                    break

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _write_sample_inputs_outputs(
    tokenizer,
    train_dataset: "TurnTakingDataset",
    val_dataset: "TurnTakingDataset",
    train_samples: List[Dict],
    val_samples: List[Dict],
    max_length: int,
    filepath: Path,
    n_per_split: int = 4,
) -> None:
    """Write full model input and target (label) for a few train/val samples to a file.
    Includes truncation info so you can see how max_length affects training."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    no_truncation = max_length is None or max_length <= 0
    with open(filepath, "w", encoding="utf-8") as f:
        if no_truncation:
            f.write("max_length = None (no truncation; full examples)\n")
        else:
            f.write(f"max_length = {max_length}\n")
        f.write("Each sample shows: full tokenized input (what the model sees), target (the single token we train on: SPEAK or SILENT), and whether the prompt was truncated.\n")
        f.write("=" * 80 + "\n\n")
        for name, dataset, samples in [
            ("TRAIN", train_dataset, train_samples),
            ("VAL", val_dataset, val_samples),
        ]:
            n = min(n_per_split, len(samples))
            for i in range(n):
                sample = samples[i]
                prompt_text = create_training_prompt(sample, tokenizer, max_length)
                full_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
                n_tokens_before = full_enc["input_ids"].shape[1]
                truncated = False if no_truncation else (n_tokens_before > max_length)
                item = dataset[i]
                input_ids = item["input_ids"]
                labels = item["labels"]
                full_text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
                label_positions = (labels != -100).nonzero(as_tuple=True)[0]
                if label_positions.numel() > 0:
                    pos = label_positions[0].item()
                    tok_id = int(labels[pos].item())
                    tok_str = tokenizer.decode([tok_id])
                    label_info = f"Position {pos}: {tok_str!r} (id={tok_id})"
                else:
                    label_info = "No label position (all -100); answer block may be truncated"
                decision = sample.get("decision", "?")
                f.write(f"\n{'='*80}\n")
                f.write(f"{name} sample {i}  |  decision={decision}  |  tokens_used={len(input_ids)}  |  tokens_before_truncation={n_tokens_before}  |  truncated={truncated}\n")
                f.write(f"{'='*80}\n")
                f.write(f"TARGET (the single token we train on): {label_info}\n")
                f.write(f"\nFULL MODEL INPUT (decoded, {len(input_ids)} tokens):\n")
                f.write("-" * 80 + "\n")
                f.write(full_text)
                f.write("\n" + "-" * 80 + "\n")
        f.write("\n" + "=" * 80 + "\n")
    print(f"[debug] Full model input/output for {n_per_split} train + {n_per_split} val samples written to: {filepath}")


import random
from torch.utils.data import Sampler

from utils.data_utils import load_samples

ALL_DATASETS = ["ami", "friends", "spgi"]


class BalancedBatchSampler(Sampler):
    """
    Yields batches with a configurable SPEAK/SILENT ratio so each batch has
    representation from both decisions. silent_ratio_in_batch=0.5 is 50/50;
    use >0.5 to oversample SILENT and counter SPEAK bias.
    """
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
    """
    Distributed version of BalancedBatchSampler: each rank gets a partition of
    SPEAK/SILENT indices and builds batches with the given silent_ratio from
    that partition so no sample is seen by two ranks.
    """
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
        # Partition indices across ranks: this rank gets indices at position rank, rank+world_size, ...
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
    train_dataset = TurnTakingDataset(train_samples, tokenizer, max_length, debug=debug)
    val_dataset = TurnTakingDataset(val_samples, tokenizer, max_length, debug=debug)
    print(f"  Train dataset: {len(train_dataset):,} samples")
    print(f"  Val dataset: {len(val_dataset):,} samples")

    # Write full model input and target for a few train/val samples to a file (only on rank 0)
    if debug and (rank is None or rank == 0) and debug_sample_io_path is not None:
        _write_sample_inputs_outputs(
            tokenizer, train_dataset, val_dataset, train_samples, val_samples,
            max_length, Path(debug_sample_io_path), n_per_split=4,
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
    """Returns a collator that stacks or pads batches and preserves our decision-token-only labels.
    Use max_length_cap when using full examples (no truncation) to avoid OOM on very long batches."""

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

        # Variable length: pad to longest in batch, cap to avoid OOM
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
