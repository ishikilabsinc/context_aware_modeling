#!/usr/bin/env python3
"""
Data Loader for Fine-Tuning

Loads and preprocesses training data for LoRA fine-tuning.
Formats samples according to the training format from ami/stage5_format_training.py
"""

import json
import re
from pathlib import Path
from typing import List, Dict
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import BASE_MODEL, TRAIN_FILE, VAL_FILE


SYSTEM_PROMPT = """You are a turn-taking decision model for a voice AI agent. Your job is to decide whether the AI agent should START TALKING or STAY SILENT after a detected pause in conversation.

You will receive:
1. An instruction telling you which speaker role the AI agent plays (e.g., "Speaker C" or "Speaker X" or "Nova")
2. The previous conversation context with speaker-labeled transcript
3. The current line: the most recent utterance before the pause

RULES FOR DECIDING:

STAY SILENT when:
- The current speaker is talking to someone else, not the AI agent
- The AI agent's name/role has not been referenced or addressed
- The speaker is mid-thought, brainstorming, or thinking aloud and not seeking input
- The sentence is clearly incomplete and the speaker is still formulating their thought
- The conversation is between other participants and does not involve the AI agent
- Someone mentions the AI agent in passing but is not requesting a response (e.g., "I was telling Speaker X about this earlier")
- The speaker is making a rhetorical statement or exclamation, not asking a question

START TALKING when:
- The speaker directly addresses the AI agent by name/role with a question or request, possibly with ASR errors
- The speaker asked the AI agent something and this is a clear follow-up to that exchange (even without re-stating the name)
- The context makes it unambiguous that the speaker is waiting for the AI agent's response
- The speaker redirects the conversation to the AI agent (e.g., "What do you think?" in a context where AI was part of the prior exchange)

IMPORTANT NUANCES:
- Once someone initiates a dialogue with the AI agent, follow-up turns from the same speaker are likely still directed at the AI agent until context clearly shifts away
- In multi-party conversations, default to SILENT unless there is clear evidence the AI agent is being addressed
- ASR (speech recognition) errors are common -- account for misspellings, homophones, and garbled names
- When uncertain, prefer SILENT -- false interruptions are far worse than missed turns
- An incomplete sentence after a long pause should remain SILENT if context suggests the speaker is still thinking, but should START TALKING if the incomplete sentence is clearly directed at the AI agent as a trailing question

Output your decision in this exact format:
<decision>SILENT or SPEAK</decision>
<confidence>high, medium, or low</confidence>
<reason>one line explanation</reason>"""



def format_context_turns(context_turns: List[Dict], max_turns: int = None) -> str:
    if not context_turns:
        return "(No previous context)"
    
    if max_turns is not None and len(context_turns) > max_turns:
        context_turns = context_turns[-max_turns:]
    
    lines = []
    for turn in context_turns:
        lines.append(f"Speaker {turn['speaker']}: {turn['text']}")
    
    return "\n".join(lines)


def format_current_turn(current_turn: Dict) -> str:
    return f"Speaker {current_turn['speaker']}: {current_turn['text']}"


def estimate_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False, max_length=10000, truncation=True))


def create_training_prompt(sample: Dict, tokenizer, max_length: int) -> str:
    target_speaker = sample.get('target_speaker', '?')
    instruction = f"You are playing the role of Speaker {target_speaker}. Decide if you should SPEAK or stay SILENT after the current utterance."
    current_str = format_current_turn(sample.get('current_turn', {}))
    decision = sample.get('decision', 'UNKNOWN')
    confidence = sample.get('confidence', 'medium')
    reason = sample.get('reason', '')
    
    system_tokens = estimate_tokens(SYSTEM_PROMPT, tokenizer)
    instruction_tokens = estimate_tokens(instruction, tokenizer)
    current_tokens = estimate_tokens(current_str, tokenizer)
    output_tokens = estimate_tokens(f"<decision>{decision}</decision><confidence>{confidence}</confidence><reason>{reason}</reason>", tokenizer)
    
    reserved_tokens = system_tokens + instruction_tokens + current_tokens + output_tokens + 100
    available_tokens = max_length - reserved_tokens
    
    context_turns = sample.get('context_turns', [])
    if not context_turns or available_tokens <= 0:
        context_str = format_context_turns(context_turns)
    else:
        context_str = _select_context_turns(context_turns, tokenizer, available_tokens)
    
    prompt = f"""<|system|>{SYSTEM_PROMPT}<|/system|>
<|instruction|>{instruction}<|/instruction|>
<|context|>{context_str}<|/context|>
<|current|>{current_str}<|/current|>
<decision>{decision}</decision>
<confidence>{confidence}</confidence>
<reason>{reason}</reason>"""
    
    return prompt


def _select_context_turns(context_turns: List[Dict], tokenizer, max_tokens: int) -> str:
    if not context_turns:
        return "(No previous context)"
    
    selected_turns = []
    current_tokens = 0
    
    for turn in reversed(context_turns):
        turn_str = f"Speaker {turn['speaker']}: {turn['text']}\n"
        turn_tokens = estimate_tokens(turn_str, tokenizer)
        
        if current_tokens + turn_tokens <= max_tokens:
            selected_turns.insert(0, turn)
            current_tokens += turn_tokens
        else:
            break
    
    if not selected_turns:
        return "(No previous context)"
    
    lines = []
    for turn in selected_turns:
        lines.append(f"Speaker {turn['speaker']}: {turn['text']}")
    
    return "\n".join(lines)



class TurnTakingDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer: AutoTokenizer, max_length: int = 2048):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
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
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten(),
        }



import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data_utils import load_samples


def prepare_datasets(tokenizer: AutoTokenizer, max_length: int = 2048):
    print("Loading training data...")
    train_samples = load_samples(TRAIN_FILE)
    print(f"  Loaded {len(train_samples):,} training samples")
    
    print("Loading validation data...")
    val_samples = load_samples(VAL_FILE)
    print(f"  Loaded {len(val_samples):,} validation samples")
    
    print("Creating datasets...")
    train_dataset = TurnTakingDataset(train_samples, tokenizer, max_length)
    val_dataset = TurnTakingDataset(val_samples, tokenizer, max_length)
    
    print(f"  Train dataset: {len(train_dataset):,} samples")
    print(f"  Val dataset: {len(val_dataset):,} samples")
    
    return train_dataset, val_dataset



def data_collator(features: List[Dict]):
    import torch
    
    batch = {}
    batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
    batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
    batch["labels"] = torch.stack([f["labels"] for f in features])
    
    return batch
