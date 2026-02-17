#!/usr/bin/env python3
"""
Inspect training/val data format and tokenization. Set DATASET via --dataset before loading config.
"""

import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ami", choices=["ami", "friends", "spgi"])
args = parser.parse_args()
os.environ["DATASET"] = args.dataset

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRAIN_FILE, VAL_FILE, BASE_MODEL
from data_loader import create_training_prompt, TurnTakingDataset
from transformers import AutoTokenizer
from utils.data_utils import load_samples


def analyze_samples(file_path, tokenizer, max_samples=5):
    print(f"\n{'='*70}")
    print(f"ANALYZING: {file_path}")
    print(f"{'='*70}")
    
    samples = load_samples(file_path)
    print(f"Total samples: {len(samples)}")
    
    if not samples:
        print("No samples found!")
        return
    
    # Analyze first few samples
    for i, sample in enumerate(samples[:max_samples]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Decision: {sample.get('decision')}")
        print(f"Target speaker: {sample.get('target_speaker')}")
        print(f"Confidence: {sample.get('confidence')}")
        print(f"Category: {sample.get('category', 'N/A')}")
        print(f"Context turns: {len(sample.get('context_turns', []))}")
        
        # Create prompt
        prompt = create_training_prompt(sample, tokenizer, max_length=384)
        
        # Tokenize
        encoding = tokenizer(
            prompt,
            truncation=True,
            max_length=384,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        
        print(f"\nTokenization:")
        print(f"  Input IDs shape: {input_ids.shape}")
        print(f"  Attention mask sum: {attention_mask.sum().item()} (non-padding tokens)")
        print(f"  Padding tokens: {(attention_mask == 0).sum().item()}")
        
        # Find decision token position
        decision_text = f"<decision>{sample.get('decision')}</decision>"
        decision_tokens = tokenizer.encode(decision_text, add_special_tokens=False)
        
        # Decode to verify
        decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
        
        # Find where decision appears
        decision_pos = decoded.find("<decision>")
        if decision_pos >= 0:
            decision_end = decoded.find("</decision>", decision_pos)
            if decision_end > decision_pos:
                decision_in_text = decoded[decision_pos:decision_end+11]
                print(f"\nDecision in decoded text:")
                print(f"  {decision_in_text}")
        
        # Show token IDs
        print(f"\nToken IDs (first 50): {input_ids[:50].tolist()}")
        print(f"Token IDs (last 50): {input_ids[-50:].tolist()}")
        
        # Check if labels match input_ids (they should for causal LM)
        labels = input_ids.clone()
        print(f"\nLabels check:")
        print(f"  Labels == Input IDs: {(labels == input_ids).all().item()}")
        print(f"  Labels shape: {labels.shape}")
        
        # Show prompt structure
        print(f"\nPrompt structure:")
        print(f"  Length: {len(prompt)} chars")
        sections = ['<|system|>', '<|instruction|>', '<|context|>', '<|current|>', '<decision>']
        for section in sections:
            if section in prompt:
                pos = prompt.find(section)
                print(f"  {section} at position {pos}")

def main():
    print("="*70)
    print("TRAINING DATA DEBUGGER")
    print("="*70)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    # Analyze training data
    if TRAIN_FILE.exists():
        analyze_samples(TRAIN_FILE, tokenizer, max_samples=3)
    else:
        print(f"\nTraining file not found: {TRAIN_FILE}")
    
    # Analyze validation data
    if VAL_FILE.exists():
        analyze_samples(VAL_FILE, tokenizer, max_samples=2)
    else:
        print(f"\nValidation file not found: {VAL_FILE}")
    
    print(f"\n{'='*70}")
    print("DEBUG COMPLETE")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
