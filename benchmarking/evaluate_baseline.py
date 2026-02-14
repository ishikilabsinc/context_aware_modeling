#!/usr/bin/env python3
"""
Baseline Evaluation for QWEN 8B

Evaluates the base QWEN 8B model (before fine-tuning) on the turn-taking task.
Measures accuracy, latency, and per-category performance.
"""

import json
import re
import time
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.constants import SPEAK_CATEGORIES, SILENT_CATEGORIES
from utils.data_utils import load_samples

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
USE_VLLM = True
FALLBACK_TO_TRANSFORMERS = True


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
TEST_FILE = DATA_DIR / 'test' / 'test_samples.jsonl'
VAL_FILE = DATA_DIR / 'val' / 'val_samples.jsonl'

# Use validation set for baseline (test set reserved for final evaluation)
EVAL_FILE = VAL_FILE

# Output file
RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / 'baseline_results.json'

# Inference settings
BATCH_SIZE = 32 if USE_VLLM else 1  # vLLM supports batching, transformers doesn't
MAX_NEW_TOKENS = 50  # Limit generation for fast inference
TEMPERATURE = 0.0  # Deterministic outputs
STOP_AFTER_DECISION = True  # Truncate after </decision> tag for latency

class QwenModelLoader:
    """Load and configure QWEN model for inference."""
    def __init__(self, use_vllm: bool = USE_VLLM):
        self.use_vllm = use_vllm
    
    def load(self) -> Dict:
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not hf_token and not os.path.exists(MODEL_NAME):
            print("\nWarning: No HuggingFace token found and model is not local.")
            print("   If the model is gated, you may need to:")
            print("   1. Run: huggingface-cli login")
            print("   2. Or set: export HF_TOKEN='your_token'\n")
        
        if self.use_vllm:
            try:
                return self._load_vllm()
            except Exception as e:
                error_msg = str(e)
                print(f"Warning: Failed to load with vLLM: {error_msg}")
                if "401" in error_msg or "Unauthorized" in error_msg:
                    print("\nAuthentication error detected!")
                    print("   Please authenticate with HuggingFace first.\n")
                if FALLBACK_TO_TRANSFORMERS:
                    print("Falling back to transformers...")
                    return self._load_transformers()
                else:
                    raise
        else:
            return self._load_transformers()
    
    def _load_vllm(self) -> Dict:
        from vllm import LLM
        from transformers import AutoTokenizer
        
        print("Loading QWEN model with vLLM...")
        print(f"Model: {MODEL_NAME}")
        
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        llm = LLM(
            model=MODEL_NAME,
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=16384,
            gpu_memory_utilization=0.85,
            max_num_seqs=128,
            trust_remote_code=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            token=hf_token,
        )
        
        print("Model loaded successfully with vLLM")
        return {"model": llm, "tokenizer": tokenizer, "use_vllm": True}
    
    def _load_transformers(self) -> Dict:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("Loading QWEN model with transformers...")
        print(f"Model: {MODEL_NAME}")
        
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            token=hf_token,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Model loaded successfully with transformers")
        return {"model": model, "tokenizer": tokenizer, "use_vllm": False}


def format_sample_for_inference(sample: Dict) -> str:
    system_prompt = """You are a turn-taking decision model for a voice AI agent. Your job is to decide whether the AI agent should START TALKING or STAY SILENT after a detected pause in conversation.

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
    
    context_turns = sample.get('context_turns', [])
    if context_turns:
        context_str = '\n'.join([
            f"Speaker {turn['speaker']}: {turn['text']}"
            for turn in context_turns
        ])
    else:
        context_str = "(No previous context)"
    
    current_turn = sample.get('current_turn', {})
    current_str = f"Speaker {current_turn.get('speaker', '?')}: {current_turn.get('text', '')}"
    
    target_speaker = sample.get('target_speaker', '?')
    instruction = f"You are playing the role of Speaker {target_speaker}. Decide if you should SPEAK or stay SILENT after the current utterance."
    
    prompt = f"""<|system|>{system_prompt}<|/system|>
<|instruction|>{instruction}<|/instruction|>
<|context|>{context_str}<|/context|>
<|current|>{current_str}<|/current|>
<decision>"""
    
    return prompt



def extract_decision_from_output(text: str) -> Optional[str]:
    match = re.search(r'<decision>(.*?)</decision>', text, re.DOTALL | re.IGNORECASE)
    if match:
        decision = match.group(1).strip().upper()
        if decision in ['SPEAK', 'SILENT']:
            return decision
        if 'SPEAK' in decision or 'TALK' in decision:
            return 'SPEAK'
        if 'SILENT' in decision:
            return 'SILENT'
    return None


def infer_with_vllm(model, tokenizer, prompts: List[str]) -> List[Tuple[str, float]]:
    """Run inference using vLLM with batching."""
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
        stop=None,
    )
    
    start_time = time.time()
    outputs = model.generate(prompts, sampling_params)
    total_time = time.time() - start_time
    
    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        latency = total_time / len(prompts)
        
        if STOP_AFTER_DECISION:
            decision_match = re.search(r'<decision>.*?</decision>', generated_text, re.DOTALL | re.IGNORECASE)
            if decision_match:
                generated_text = generated_text[:decision_match.end()]
        
        results.append((generated_text, latency))
    
    return results


def infer_with_transformers(model, tokenizer, prompts: List[str]) -> List[Tuple[str, float]]:
    """Run inference using transformers."""
    import torch
    
    results = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                do_sample=TEMPERATURE > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
        inference_time = time.time() - start_time
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if STOP_AFTER_DECISION:
            decision_match = re.search(r'<decision>.*?</decision>', generated_text, re.DOTALL | re.IGNORECASE)
            if decision_match:
                generated_text = generated_text[:decision_match.end()]
        
        results.append((generated_text, inference_time))
    
    return results



def evaluate_samples(
    samples: List[Dict],
    model,
    tokenizer,
    use_vllm: bool,
    batch_size: int = BATCH_SIZE
) -> Dict:
    print(f"\nEvaluating {len(samples)} samples...")
    print(f"Batch size: {batch_size}")
    print(f"Using vLLM: {use_vllm}")
    
    prompts = [format_sample_for_inference(sample) for sample in samples]
    
    all_predictions = []
    all_latencies = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_samples = samples[i:i+batch_size]
        
        print(f"  Processing batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1} "
              f"({len(batch_prompts)} samples)...", end=' ', flush=True)
        
        if use_vllm:
            batch_results = infer_with_vllm(model, tokenizer, batch_prompts)
        else:
            batch_results = infer_with_transformers(model, tokenizer, batch_prompts)
        
        for (output_text, latency), sample in zip(batch_results, batch_samples):
            prediction = extract_decision_from_output(output_text)
            ground_truth = sample.get('decision', 'UNKNOWN')
            
            all_predictions.append({
                'sample_id': sample.get('decision_point_id', f'sample_{i}'),
                'ground_truth': ground_truth,
                'prediction': prediction,
                'category': sample.get('category', 'UNKNOWN'),
                'output_text': output_text[:200],
                'latency': latency
            })
            all_latencies.append(latency)
        
        print(f"(avg latency: {np.mean([r[1] for r in batch_results]):.3f}s)")
    
    correct = sum(1 for p in all_predictions if p['prediction'] == p['ground_truth'])
    accuracy = correct / len(all_predictions) if all_predictions else 0
    
    category_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    for pred in all_predictions:
        cat = pred['category']
        category_metrics[cat]['total'] += 1
        if pred['prediction'] == pred['ground_truth']:
            category_metrics[cat]['correct'] += 1
    
    confusion = defaultdict(int)
    for pred in all_predictions:
        key = f"{pred['ground_truth']}_->_{pred['prediction']}"
        confusion[key] += 1
    
    latencies = np.array(all_latencies)
    latency_stats = {
        'mean': float(np.mean(latencies)),
        'median': float(np.median(latencies)),
        'p50': float(np.percentile(latencies, 50)),
        'p95': float(np.percentile(latencies, 95)),
        'p99': float(np.percentile(latencies, 99)),
        'min': float(np.min(latencies)),
        'max': float(np.max(latencies)),
    }
    
    false_positives = sum(1 for p in all_predictions 
                          if p['ground_truth'] == 'SILENT' and p['prediction'] == 'SPEAK')
    false_negatives = sum(1 for p in all_predictions 
                          if p['ground_truth'] == 'SPEAK' and p['prediction'] == 'SILENT')
    
    total_silent = sum(1 for p in all_predictions if p['ground_truth'] == 'SILENT')
    total_speak = sum(1 for p in all_predictions if p['ground_truth'] == 'SPEAK')
    
    fpr = false_positives / total_silent if total_silent > 0 else 0
    fnr = false_negatives / total_speak if total_speak > 0 else 0
    
    results = {
        'total_samples': len(all_predictions),
        'accuracy': accuracy,
        'correct': correct,
        'incorrect': len(all_predictions) - correct,
        'category_accuracy': {
            cat: {
                'accuracy': metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0,
                'correct': metrics['correct'],
                'total': metrics['total']
            }
            for cat, metrics in category_metrics.items()
        },
        'confusion_matrix': dict(confusion),
        'latency_stats': latency_stats,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'predictions': all_predictions[:100],
    }
    
    return results



def main():
    print("="*70)
    print("BASELINE EVALUATION: QWEN 8B")
    print("="*70)
    
    print("\nLoading model...")
    loader = QwenModelLoader(use_vllm=USE_VLLM)
    model_result = loader.load()
    model = model_result['model']
    tokenizer = model_result['tokenizer']
    use_vllm = model_result['use_vllm']
    
    print(f"\nLoading samples from {EVAL_FILE}...")
    samples = load_samples(EVAL_FILE)
    print(f"Loaded {len(samples)} samples")
    
    results = evaluate_samples(samples, model, tokenizer, use_vllm, BATCH_SIZE)
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nTotal samples: {results['total_samples']:,}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Correct: {results['correct']:,}")
    print(f"Incorrect: {results['incorrect']:,}")
    
    print(f"\nFalse Positive Rate (SILENT -> SPEAK): {results['false_positive_rate']:.2%}")
    print(f"False Negative Rate (SPEAK -> SILENT): {results['false_negative_rate']:.2%}")
    
    print(f"\nLatency Statistics:")
    for stat, value in results['latency_stats'].items():
        print(f"  {stat}: {value:.4f}s")
    
    print(f"\nPer-Category Accuracy:")
    for cat in sorted(results['category_accuracy'].keys()):
        metrics = results['category_accuracy'][cat]
        print(f"  {cat}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    
    print(f"\nSaving results to {RESULTS_FILE}...")
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved")
    
    print("\n" + "="*70)
    print("BASELINE EVALUATION COMPLETE")
    print("="*70)
    
    return results


if __name__ == '__main__':
    main()
