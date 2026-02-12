#!/usr/bin/env python3
"""
Stage 5: Format for Training

Goal:
    Transform categorized samples into final JSONL format for training.
    
    Create two output files:
    1. Intermediate JSONL: Human-readable with all metadata
    2. Training JSONL: Formatted with system prompt structure

Training Format:
    <|system|>{System prompt}<|/system|>
    <|instruction|>{Role instruction}<|/instruction|>
    <|context|>{Previous conversation}<|/context|>
    <|current|>{Current utterance}<|/current|>
    <decision>SPEAK/SILENT</decision>
    <confidence>high/medium/low</confidence>
    <reason>...</reason>

Input:
    - stage4_categorized_samples.json from Stage 4

Output:
    - training_data_intermediate.jsonl - human-readable with metadata
    - training_data_formatted.jsonl - final training format
"""

import json
import os
from pathlib import Path
from typing import Dict

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = 'data_final/stage45_filtered_samples.jsonl'  # Use filtered JSONL from Stage 4.5
# To use unfiltered data: INPUT_FILE = 'stage4_categorized_samples.json'

# Directory for all intermediate JSON dumps across stages
JSON_DUMPS_DIR = Path(__file__).parent / 'json_dumps'
JSON_DUMPS_DIR.mkdir(exist_ok=True)

INTERMEDIATE_OUTPUT = JSON_DUMPS_DIR / 'training_data_intermediate.jsonl'
TRAINING_OUTPUT = JSON_DUMPS_DIR / 'training_data_formatted.jsonl'

# System prompt from the PDF
SYSTEM_PROMPT = """You are a turn-taking decision model for a voice AI agent. Your job is to decide whether the AI agent should START TALKING or STAY SILENT after a detected pause in conversation.

You will receive:
1. An instruction telling you which speaker role the AI agent plays
2. The previous conversation context with speaker-labeled transcript
3. The current line: the most recent utterance before the pause

RULES FOR DECIDING:

STAY SILENT when:
- The current speaker is talking to someone else, not the AI agent
- The AI agent's name/role has not been referenced or addressed
- The speaker is mid-thought, brainstorming, or thinking aloud
- The sentence is clearly incomplete
- The conversation is between other participants
- Someone mentions the AI agent in passing but not requesting response

START TALKING when:
- The speaker directly addresses the AI agent by name/role
- The speaker asked the AI agent something and this is a follow-up
- The context makes it unambiguous the speaker is waiting for AI response
- The speaker redirects the conversation to the AI agent

IMPORTANT: When uncertain, prefer SILENT

Output your decision in this exact format:
<decision>SILENT or SPEAK</decision>
<confidence>high, medium, or low</confidence>
<reason>one line explanation</reason>"""

# ============================================================================
# FORMATTING FUNCTIONS
# ============================================================================

def format_context_turns(context_turns: list) -> str:
    """Format context turns into string"""
    if not context_turns:
        return "(No previous context)"
    
    lines = []
    for turn in context_turns:
        lines.append(f"Speaker {turn['speaker']}: {turn['text']}")
    
    return "\n".join(lines)


def format_current_turn(current_turn: dict) -> str:
    """Format current turn into string"""
    return f"Speaker {current_turn['speaker']}: {current_turn['text']}"


def create_intermediate_sample(sample: Dict, sample_id: int) -> Dict:
    """
    Create intermediate JSONL sample (human-readable)
    
    Includes all metadata for inspection and debugging
    """
    context_str = format_context_turns(sample['context_turns'])
    current_str = format_current_turn(sample['current_turn'])
    
    return {
        'sample_id': sample_id,
        'decision_point_id': sample['decision_point_id'],
        'meeting_id': sample['meeting_id'],
        'sequence_id': sample['sequence_id'],
        'target_speaker': sample['target_speaker'],
        'all_speakers': sample['all_speakers'],
        'context': context_str,
        'current_turn': current_str,
        'decision': sample['decision'],
        'confidence': sample['confidence'],
        'reason': sample['reason'],
        'category': sample['category']
    }


def create_training_sample(sample: Dict) -> str:
    """
    Create training format sample (with system prompt structure)
    
    This is the final format that will be used for model training
    """
    context_str = format_context_turns(sample['context_turns'])
    current_str = format_current_turn(sample['current_turn'])
    target_speaker = sample['target_speaker']
    
    instruction = f"You are playing the role of Speaker {target_speaker}. Decide if you should SPEAK or stay SILENT after the current utterance."
    
    training_text = f"""<|system|>{SYSTEM_PROMPT}<|/system|>
<|instruction|>{instruction}<|/instruction|>
<|context|>{context_str}<|/context|>
<|current|>{current_str}<|/current|>
<decision>{sample['decision']}</decision>
<confidence>{sample['confidence']}</confidence>
<reason>{sample['reason']}</reason>"""
    
    return training_text


def print_examples(categorized_samples: list, training_samples: list):
    """Print example formatted samples"""
    print("\n" + "="*70)
    print("EXAMPLE FORMATTED SAMPLES")
    print("="*70 + "\n")
    
    # Show one SPEAK and one SILENT example
    speak_idx = next(i for i, s in enumerate(categorized_samples) if s['decision'] == 'SPEAK')
    silent_idx = next(i for i, s in enumerate(categorized_samples) if s['decision'] == 'SILENT')
    
    print("SPEAK Example:")
    print("-" * 70)
    print(training_samples[speak_idx])
    print("\n" + "="*70 + "\n")
    
    print("SILENT Example:")
    print("-" * 70)
    print(training_samples[silent_idx])
    print("\n" + "="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    # Load filtered samples from Stage 4.5
    print(f"Loading samples from {INPUT_FILE}...")
    
    categorized_samples = []
    
    # Check if input is JSONL or JSON
    if INPUT_FILE.endswith('.jsonl'):
        # Read JSONL format (one JSON object per line)
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    categorized_samples.append(json.loads(line))
    else:
        # Read JSON format (single array)
        with open(INPUT_FILE, 'r') as f:
            categorized_samples = json.load(f)
    
    print(f"✓ Loaded {len(categorized_samples):,} categorized samples")
    print("Generating intermediate JSONL...")
    print("="*70)
    
    # ========================================================================
    # Generate intermediate JSONL
    # ========================================================================
    intermediate_samples = []
    
    for i, sample in enumerate(categorized_samples, start=1):
        intermediate_sample = create_intermediate_sample(sample, i)
        intermediate_samples.append(intermediate_sample)
        
        if i % 10000 == 0:
            print(f"Formatted {i:,}/{len(categorized_samples):,} intermediate samples...")
    
    print("="*70)
    print(f"Generated {len(intermediate_samples):,} intermediate samples")
    
    # Save intermediate JSONL
    with open(INTERMEDIATE_OUTPUT, 'w') as f:
        for sample in intermediate_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"✓ Saved {len(intermediate_samples):,} samples to {INTERMEDIATE_OUTPUT}")
    
    # ========================================================================
    # Generate training JSONL
    # ========================================================================
    print("\nGenerating training JSONL...")
    print("="*70)
    
    training_samples = []
    
    for i, sample in enumerate(categorized_samples, start=1):
        training_text = create_training_sample(sample)
        training_samples.append(training_text)
        
        if i % 10000 == 0:
            print(f"Formatted {i:,}/{len(categorized_samples):,} training samples...")
    
    print("="*70)
    print(f"Generated {len(training_samples):,} training samples")
    
    # Save training JSONL
    with open(TRAINING_OUTPUT, 'w') as f:
        for sample_text in training_samples:
            # Each line is a JSON object with 'text' field
            f.write(json.dumps({'text': sample_text}) + '\n')
    
    print(f"✓ Saved {len(training_samples):,} samples to {TRAINING_OUTPUT}")
    
    # ========================================================================
    # Print examples and file stats
    # ========================================================================
    print_examples(categorized_samples, training_samples)
    
    # File size statistics
    print("\n" + "="*70)
    print("FILE STATISTICS")
    print("="*70)
    
    intermediate_size = os.path.getsize(INTERMEDIATE_OUTPUT)
    training_size = os.path.getsize(TRAINING_OUTPUT)
    
    print(f"\nIntermediate JSONL: {INTERMEDIATE_OUTPUT}")
    print(f"  Size: {intermediate_size / 1024 / 1024:.2f} MB")
    print(f"  Samples: {len(intermediate_samples):,}")
    
    print(f"\nTraining JSONL: {TRAINING_OUTPUT}")
    print(f"  Size: {training_size / 1024 / 1024:.2f} MB")
    print(f"  Samples: {len(training_samples):,}")
    
    avg_sample_size = training_size / len(training_samples)
    print(f"\nAverage training sample size: {avg_sample_size:.0f} bytes")
    
    print(f"\n" + "="*70)
    print(f"\nReady for Stage 6: Statistics & Validation")


if __name__ == '__main__':
    main()
