#!/usr/bin/env python3
"""
Baseline evaluation for instruct models on the turn-taking task.
Writes prediction-only JSON; metrics are computed via benchmarking.metrics
Model and dataset via CLI or env (MODEL, DATASET).
"""

import json
import re
import time
import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data_utils import load_samples, filter_samples_with_context
from fine_tuning.config import MODEL_OPTIONS, MODEL as DEFAULT_MODEL, BASE_MODEL as DEFAULT_BASE_MODEL
from benchmarking.metrics import compute_metrics, generate_detail_report

# Model / inference configuration
USE_VLLM = True
FALLBACK_TO_TRANSFORMERS = True

BASE_DIR = Path(__file__).resolve().parents[1]

# Inference settings
BATCH_SIZE = 32 if USE_VLLM else 1  # vLLM supports batching, transformers doesn't
MAX_NEW_TOKENS = 256  # Same for all models (fair comparison)
TEMPERATURE = 0.0 # No temperature for deterministic decisions
STOP_AFTER_DECISION = True # Stop after the decision is made to save time

# In saved results, show at most this many context turns per example (latest w.r.t current turn)
MAX_CONTEXT_TURNS_IN_RESULTS = 20

# System prompt for the model
SYSTEM_PROMPT = """
    You are a turn-taking decision model in a multi-party conversation where multiple people are talking.
    You are roleplaying the role of the target speaker you are given.
    Your job is to decide whether the target speaker should START TALKING or STAY SILENT after a detected pause in conversation.

    You will receive:
        1. An instruction telling you the target speaker role (e.g., "Speaker C" or "Speaker X" or "Nova")
        2. The previous conversation context with speaker-labeled transcript
        3. Most recent utterance: the most recent utterance said in the conversation, after which you have to make a decision

    First, determine the target speaker's ROLE in the current exchange:
        - ACTIVE PARTICIPANT: The target speaker has been speaking, was addressed, or is part of an ongoing back-and-forth in the current topic.
        - BYSTANDER: The target speaker has not been involved in the current exchange and is passively listening.

    RULES FOR DECIDING:

        Output SILENT when:
        - The target speaker is a BYSTANDER and the recent utterance is directed at someone else
        - The target speaker has not been referenced, addressed, or involved, and the context does not suggest they are expected to contribute
        - The recent utterance is clearly incomplete — the speaker is visibly mid-sentence or mid-clause and still formulating their thought
        - Someone mentions the target speaker in third person without expecting a response (e.g., "I was telling Speaker X about this earlier")

        Output SPEAK when:
        - The recent utterance directly addresses the target speaker by name/role with a question or request, possibly with ASR errors
        - The recent utterance asked the target speaker something and this is a clear follow-up to that exchange (even without re-stating the name)
        - The context makes it unambiguous that the speaker is waiting for the target speaker to respond
        - The speaker redirects the conversation to the target speaker (e.g., "What do you think?" in a context where the target speaker was part of the prior exchange)
        - The recent utterance is a general or group-directed question (e.g., "What do we think?", "Anyone disagree?", "Right?", "You know?") and the target speaker is part of the group
        - The target speaker is an ACTIVE PARTICIPANT and the recent utterance completes a thought, story, opinion, or statement on the topic they have been engaging with — a response (agreement, reaction, follow-up, acknowledgment) is socially expected to maintain natural conversational flow
        - The target speaker previously asked a question or made a request, and the recent utterance is the answer or response to it — the target speaker is expected to acknowledge or follow up
        - The target speaker is an ACTIVE PARTICIPANT and staying silent would unnaturally drop them from the conversation or create an awkward pause
        - The conversation has reached a natural transition point where a brief backchannel or reactive response (e.g., agreement, laughter, "wow", "yeah") from the target speaker would be natural given the content

    IMPORTANT NUANCES:
        - The key distinction is ACTIVE PARTICIPANT vs BYSTANDER. Active participants should SPEAK at natural turn boundaries. Bystanders should default to SILENT unless directly addressed.
        - When uncertain AND the target speaker is a bystander → prefer SILENT
        - When uncertain AND the target speaker has been actively participating in this exchange → consider whether the most recent utterance is clearly directed at someone else or is a self-contained statement. If so, SILENT is still correct even for active participants.
        - False interruptions of OTHER PEOPLE'S conversations are bad, but failing to respond when you are part of the conversation is equally bad — it kills the interaction

    Output your response in this EXACT format:

        <reasoning>Determine if target speaker is an ACTIVE PARTICIPANT or BYSTANDER, then explain in 1 sentence who is being addressed and whether the target speaker should respond.</reasoning>
        <decision>SPEAK</decision> or <decision>SILENT</decision>
        <confidence>high, medium, or low</confidence>

    CRITICAL: The <decision> tag must contain ONLY the single word SPEAK or SILENT. Nothing else.
    Do not summarize or repeat the conversation. Output only: one sentence in <reasoning>, then <decision>SPEAK or SILENT</decision>, then <confidence>.

    EXAMPLES:

    EXAMPLE 1:
    Target speaker: Alex
    Speakers: Alex, Sam
    Recent utterance (Sam): "Wait, you actually told her?"
    <reasoning>Alex is an ACTIVE PARTICIPANT — Sam is directly asking Alex a question. A response is expected.</reasoning>
    <decision>SPEAK</decision>
    <confidence>high</confidence>

    EXAMPLE 2:
    Target speaker: Jordan
    Speakers: Alex, Jordan, Sam
    Recent utterance (Alex): "Yeah, it was rough. I didn't sleep at all last night."
    <reasoning>Jordan is a BYSTANDER — Alex is narrating a personal experience to the group. Jordan hasn't been addressed or involved. No response expected.</reasoning>
    <decision>SILENT</decision>
    <confidence>high</confidence>
    """

# Use system prompt once (1) or twice (2).
SYSTEM_PROMPT_REPEAT = 2


def _get_system_prompt_content() -> str:
    """Return SYSTEM_PROMPT repeated SYSTEM_PROMPT_REPEAT times."""
    base = SYSTEM_PROMPT.strip()
    return "\n\n".join([base] * SYSTEM_PROMPT_REPEAT)


class InstructModelLoader:
    """Load and configure instruct model for inference."""
    def __init__(self, model_id: str, use_vllm: bool = USE_VLLM):
        self.model_id = model_id
        self.use_vllm = use_vllm

    def load(self) -> Dict:
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not hf_token and not os.path.exists(self.model_id):
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

        print("Loading model with vLLM...")
        print(f"Model: {self.model_id}")

        gpu_mem = 0.85 if self.model_id in ("Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct") else 0.75
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        llm = LLM(
            model=self.model_id,
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=18400,
            gpu_memory_utilization=gpu_mem,
            max_num_seqs=128,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            token=hf_token,
        )

        print("Model loaded successfully with vLLM")
        return {"model": llm, "tokenizer": tokenizer, "use_vllm": True}

    def _load_transformers(self) -> Dict:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading model with transformers...")
        print(f"Model: {self.model_id}")

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            token=hf_token,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
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
    system_prompt = _get_system_prompt_content()

    context_turns = sample.get('context_turns', [])
    current_turn = sample.get('current_turn', {})

    all_turns = context_turns + ([current_turn] if current_turn else [])
    if all_turns:
        context_lines = [
            f"Speaker {turn['speaker']}: {turn['text']}"
            for turn in all_turns[:-1]
        ]
        if current_turn:
            context_lines.append(f"Speaker {current_turn.get('speaker', '?')}: {current_turn.get('text', '')}  [MOST RECENT - after this there was a pause]")
        context_str = '\n'.join(context_lines)
    else:
        context_str = "(No previous context)"

    if current_turn:
        current_str = f"Speaker {current_turn.get('speaker', '?')}: {current_turn.get('text', '')}"
    else:
        current_str = "(No current utterance)"

    target_speaker = sample.get('target_speaker', '?')
    instruction = f"You are playing the role of Speaker {target_speaker}. The conversation history above shows all utterances including the most recent one (marked as [MOST RECENT]). After that most recent utterance, there was a pause. Decide if you (Speaker {target_speaker}) should START TALKING or STAY SILENT now."

    prompt = f"""<|system|>{system_prompt}<|/system|>
<|instruction|>{instruction}<|/instruction|>
<|context|>{context_str}<|/context|>
<|current|>MOST RECENT UTTERANCE (the previous utterance that just occurred): {current_str}<|/current|>
Reply with your decision in this exact format: <reasoning>One sentence: ACTIVE PARTICIPANT or BYSTANDER, and who is addressed.</reasoning> <decision>SPEAK</decision> or <decision>SILENT</decision> <confidence>high</confidence> or <confidence>medium</confidence> or <confidence>low</confidence>
<reasoning>"""

    return prompt


def is_qwen3(model_id: str) -> bool:
    """True if model uses chat template with enable_thinking=False."""
    return model_id and "qwen3" in model_id.lower()


def get_max_new_tokens(model_id: Optional[str] = None) -> int:
    return MAX_NEW_TOKENS


def _build_system_and_user_content(sample: Dict) -> Tuple[str, str]:
    """
    Build the canonical system and user message content used for all models.
    Same content for every model family to keep evaluation fair (no model favored).
    """
    system_content = _get_system_prompt_content()
    context_turns = sample.get('context_turns', [])
    current_turn = sample.get('current_turn', {})

    all_turns = context_turns + ([current_turn] if current_turn else [])
    if all_turns:
        context_lines = [
            f"Speaker {turn['speaker']}: {turn['text']}"
            for turn in all_turns[:-1]
        ]
        if current_turn:
            context_lines.append(f"Speaker {current_turn.get('speaker', '?')}: {current_turn.get('text', '')}  [MOST RECENT - after this there was a pause]")
        context_str = '\n'.join(context_lines)
    else:
        context_str = "(No previous context)"

    if current_turn:
        current_str = f"Speaker {current_turn.get('speaker', '?')}: {current_turn.get('text', '')}"
    else:
        current_str = "(No current utterance)"

    target_speaker = sample.get('target_speaker', '?')
    instruction = f"You are playing the role of Speaker {target_speaker}. The conversation history above shows all utterances including the most recent one (marked as [MOST RECENT]). After that most recent utterance, there was a pause. Decide if you (Speaker {target_speaker}) should START TALKING or STAY SILENT now."

    user_content = f"""{instruction}

CONVERSATION CONTEXT:
{context_str}

MOST RECENT UTTERANCE (the previous utterance that just occurred): {current_str}

Reply with your decision in this exact format: <reasoning>One sentence: ACTIVE PARTICIPANT or BYSTANDER, and who is addressed.</reasoning> <decision>SPEAK</decision> or <decision>SILENT</decision> <confidence>high</confidence> or <confidence>medium</confidence> or <confidence>low</confidence>"""
    return system_content, user_content.strip()


def format_sample_for_qwen3(sample: Dict, tokenizer) -> str:
    """Build prompt using Qwen3 chat template with enable_thinking=False."""
    system_content, user_content = _build_system_and_user_content(sample)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + "\n<reasoning>"


# Model IDs (HuggingFace) that use the generic chat template with the same system/user content.
# Ensures fair comparison: identical prompt content, only tokenization differs per model.
CHAT_TEMPLATE_MODEL_IDS = frozenset({
    "meta-llama/Llama-3.1-8B-Instruct",
    "openai/gpt-oss-20b",
    "mistralai/Mistral-7B-Instruct-v0.3",
})


def _uses_generic_chat_template(model_id: Optional[str]) -> bool:
    """True if model should use generic chat template (same content as Qwen3, different tokenization)."""
    return bool(model_id and model_id in CHAT_TEMPLATE_MODEL_IDS)


def format_sample_for_chat_template(sample: Dict, tokenizer) -> str:
    """
    Build prompt using the tokenizer's chat template with canonical system/user content.
    Same content as format_sample_for_qwen3 for fair comparison; no model-specific prompt changes.
    """
    system_content, user_content = _build_system_and_user_content(sample)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + "\n<reasoning>"


def extract_decision_from_output(text: str) -> Optional[str]:
    # Prefer last <decision>...</decision> when multiple (model may output role then actual decision)
    tag_matches = re.findall(r'<decision>(.*?)</decision>', text, re.DOTALL | re.IGNORECASE)
    if tag_matches:
        decision = tag_matches[-1].strip().upper()
    else:
        match = re.search(r'^(.*?)</decision>', text, re.DOTALL | re.IGNORECASE)
        if match:
            decision = match.group(1).strip().upper()
        else:
            # Output may end with <decision>SPEAK or <decision>SILENT without </decision> (e.g. vLLM excluded stop string)
            partial = re.search(r'<decision>\s*(\S+)', text, re.IGNORECASE)
            if partial:
                raw = partial.group(1).strip().upper()
                if raw.startswith('SPEAK'):
                    return 'SPEAK'
                if raw.startswith('SILE'):
                    return 'SILENT'
            underscore_match = re.search(r'_([A-Z]+)_', text, re.IGNORECASE)
            if underscore_match:
                decision = underscore_match.group(1).strip().upper()
            else:
                decision_match = re.search(r'^(STAY\s+)?(SILENT|SPEAK|TALK|START\s+TALK)', text, re.IGNORECASE)
                if decision_match and decision_match.group(2):
                    decision = decision_match.group(2).upper()
                else:
                    return None

    if decision == 'SPEAK' or decision == 'SILENT':
        return decision
    if 'SPEAK' in decision:
        return 'SPEAK'
    if 'SILENT' in decision:
        return 'SILENT'
    if ('SPEAK' in decision or 'TALK' in decision or 'START' in decision) and 'PARTICIPANT' not in decision:
        return 'SPEAK'
    if ('SILENT' in decision or 'STAY' in decision) and 'PARTICIPANT' not in decision:
        return 'SILENT'
    # Fallback: model put role in <decision> tag (ACTIVE PARTICIPANT -> SPEAK, BYSTANDER -> SILENT)
    if 'ACTIVE PARTICIPANT' in decision or decision == 'ACTIVE PARTICIPANT':
        return 'SPEAK'
    if 'BYSTANDER' in decision:
        return 'SILENT'
    return None


def infer_with_vllm(model, tokenizer, prompts: List[str], model_id: Optional[str] = None) -> List[Tuple[str, float]]:
    from vllm import SamplingParams

    max_tokens = get_max_new_tokens(model_id)
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
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


def infer_with_transformers(
    model, tokenizer, prompts: List[str], model_id: Optional[str] = None
) -> List[Tuple[str, float]]:
    import torch

    max_new_tokens = get_max_new_tokens(model_id)
    results = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
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
    batch_size: int = BATCH_SIZE,
    debug_prompts: bool = False,
    model_id: Optional[str] = None,
) -> Dict:
    print(f"\nEvaluating {len(samples)} samples...")
    print(f"Batch size: {batch_size}")
    print(f"Using vLLM: {use_vllm}")

    if model_id and is_qwen3(model_id):
        prompts = [format_sample_for_qwen3(sample, tokenizer) for sample in samples]
        print("Using chat template (enable_thinking=False)")
    elif model_id and _uses_generic_chat_template(model_id):
        prompts = [format_sample_for_chat_template(sample, tokenizer) for sample in samples]
        print("Using generic chat template (same prompt content as other models)")
    else:
        prompts = [format_sample_for_inference(sample) for sample in samples]
    
    if debug_prompts:
        num_debug_samples = min(4, len(samples))
        print(f"\n{'='*70}")
        print(f"[DEBUG] Full Prompts for First {num_debug_samples} Samples")
        print(f"{'='*70}")
        for idx in range(num_debug_samples):
            sample = samples[idx]
            prompt = prompts[idx]
            
            system_start = prompt.find('<|system|>')
            system_end = prompt.find('<|/system|>')
            if system_start != -1 and system_end != -1:
                system_prompt = prompt[system_start + len('<|system|>'):system_end].strip()
            else:
                system_prompt = "(Not found)"
            
            instruction_start = prompt.find('<|instruction|>')
            instruction_end = prompt.find('<|/instruction|>')
            if instruction_start != -1 and instruction_end != -1:
                instruction = prompt[instruction_start + len('<|instruction|>'):instruction_end].strip()
            else:
                instruction = "(Not found)"
            
            context_start = prompt.find('<|context|>')
            context_end = prompt.find('<|/context|>')
            if context_start != -1 and context_end != -1:
                context = prompt[context_start + len('<|context|>'):context_end].strip()
            else:
                context = "(Not found)"
            
            current_start = prompt.find('<|current|>')
            current_end = prompt.find('<|/current|>')
            if current_start != -1 and current_end != -1:
                current = prompt[current_start + len('<|current|>'):current_end].strip()
            else:
                current = "(Not found)"
            
            print(f"\n{'='*70}")
            print(f"[SAMPLE {idx + 1}] {sample.get('decision_point_id', 'N/A')}")
            print(f"{'='*70}")
            print(f"\n--- System Prompt ---")
            print(system_prompt)
            print(f"\n--- Instruction Prompt ---")
            print(instruction)
            print(f"\n--- Context ---")
            print(context)
            print(f"\n--- Current Turn ---")
            print(current)
            print(f"\n--- Full Input to Model ---")
            print(prompt)
            print(f"{'='*70}\n")
    
    all_predictions = []
    all_latencies = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_samples = samples[i:i+batch_size]
        
        print(f"  Processing batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1} "
              f"({len(batch_prompts)} samples)...", end=' ', flush=True)
        
        if use_vllm:
            batch_results = infer_with_vllm(model, tokenizer, batch_prompts, model_id=model_id)
        else:
            batch_results = infer_with_transformers(model, tokenizer, batch_prompts, model_id=model_id)
        
        for batch_idx, ((output_text, latency), sample) in enumerate(zip(batch_results, batch_samples)):
            prediction = extract_decision_from_output(output_text)
            ground_truth = sample.get('decision', 'UNKNOWN')
            
            if i == 0 and batch_idx < 3:
                print(f"\n{'='*70}")
                print(f"[DEBUG] Sample {batch_idx + 1} - Full Details")
                print(f"{'='*70}")
                print(f"Sample ID: {sample.get('decision_point_id', 'N/A')}")
                print(f"Category: {sample.get('category', 'N/A')}")
                print(f"Confidence (ground truth): {sample.get('confidence', 'N/A')}")
                print(f"\n--- AI Agent Role ---")
                print(f"Target Speaker: {sample.get('target_speaker', 'N/A')}")
                print(f"All Speakers: {sample.get('all_speakers', [])}")
                
                print(f"\n--- Context Turns ({len(sample.get('context_turns', []))} turns) ---")
                context_turns = sample.get('context_turns', [])
                if context_turns:
                    for idx, turn in enumerate(context_turns[-5:], 1):
                        print(f"  {idx}. Speaker {turn.get('speaker', '?')}: {turn.get('text', '')[:100]}{'...' if len(turn.get('text', '')) > 100 else ''}")
                    if len(context_turns) > 5:
                        print(f"  ... ({len(context_turns) - 5} more turns before)")
                else:
                    print(f"  (No previous context)")
                
                print(f"\n--- Current Turn ---")
                current_turn = sample.get('current_turn', {})
                print(f"Speaker: {current_turn.get('speaker', '?')}")
                print(f"Text: {current_turn.get('text', '')}")
                
                print(f"\n--- Decision ---")
                print(f"Ground Truth: {ground_truth}")
                print(f"Extracted Prediction: {prediction}")
                print(f"Match: {'CORRECT' if prediction == ground_truth else 'WRONG'}")
                
                print(f"\n--- Model Output ---")
                print(f"Full output: {output_text}")
                if '</decision>' in output_text:
                    decision_end = output_text.find('</decision>')
                    decision_content = output_text[:decision_end].strip()
                    print(f"Extracted from: '{decision_content}' → {prediction}")
                print(f"{'='*70}")
            
            context_turns_list = sample.get('context_turns', [])
            if len(context_turns_list) > MAX_CONTEXT_TURNS_IN_RESULTS:
                context_turns_display = context_turns_list[-MAX_CONTEXT_TURNS_IN_RESULTS:]
                context_turns_total = len(context_turns_list)
            else:
                context_turns_display = context_turns_list
                context_turns_total = len(context_turns_list)

            all_predictions.append({
                'sample_id': sample.get('decision_point_id', f'sample_{i+batch_idx}'),
                'ground_truth': ground_truth,
                'prediction': prediction,
                'category': sample.get('category', 'UNKNOWN'),
                'output_text': output_text,
                'latency': latency,
                'target_speaker': sample.get('target_speaker', 'N/A'),
                'all_speakers': sample.get('all_speakers', []),
                'context_turns': context_turns_display,
                'context_turns_total': context_turns_total,
                'current_turn': sample.get('current_turn', {}),
                'confidence': sample.get('confidence', 'N/A'),
            })
            all_latencies.append(latency)
        
        print(f"(avg latency: {np.mean([r[1] for r in batch_results]):.3f}s)")
    
    print(f"\n[DEBUG] Prediction analysis:")
    prediction_counts = defaultdict(int)
    ground_truth_counts = defaultdict(int)
    none_predictions = 0
    
    for p in all_predictions:
        pred = p['prediction']
        gt = p['ground_truth']
        prediction_counts[pred if pred else 'None'] += 1
        ground_truth_counts[gt] += 1
        if pred is None:
            none_predictions += 1
    
    print(f"  Total samples: {len(all_predictions)}")
    if len(all_predictions) == 0:
        print("  No samples to evaluate. Please check if data files exist.")
        return []

    print(f"  Predictions distribution:")
    for pred, count in sorted(prediction_counts.items()):
        print(f"    {pred}: {count} ({count/len(all_predictions)*100:.1f}%)")
    print(f"  Ground truth distribution:")
    for gt, count in sorted(ground_truth_counts.items()):
        print(f"    {gt}: {count} ({count/len(all_predictions)*100:.1f}%)")
    print(f"  None predictions (failed extraction): {none_predictions} ({none_predictions/len(all_predictions)*100:.1f}%)")
    
    mismatches = [p for p in all_predictions if p['prediction'] != p['ground_truth']]
    if mismatches:
        print(f"\n[DEBUG] First 10 mismatches (with full details):")
        for idx, p in enumerate(mismatches[:10]):
            print(f"\n{'='*70}")
            print(f"[MISMATCH {idx+1}] Sample ID: {p['sample_id']}")
            print(f"{'='*70}")
            print(f"Category: {p['category']}")
            print(f"Confidence (ground truth): {p['confidence']}")
            print(f"\n--- AI Agent Role ---")
            print(f"Target Speaker: {p['target_speaker']}")
            print(f"All Speakers: {p['all_speakers']}")
            
            total_ctx = p.get('context_turns_total', len(p['context_turns']))
            suffix = f", {total_ctx} total" if total_ctx > MAX_CONTEXT_TURNS_IN_RESULTS else ""
            print(f"\n--- Context Turns ({len(p['context_turns'])} turns{suffix}) ---")
            context_turns = p['context_turns']
            if context_turns:
                for turn_idx, turn in enumerate(context_turns[-5:], 1):
                    print(f"  {turn_idx}. Speaker {turn.get('speaker', '?')}: {turn.get('text', '')[:100]}{'...' if len(turn.get('text', '')) > 100 else ''}")
                if len(context_turns) > 5:
                    print(f"  ... ({len(context_turns) - 5} more turns before)")
            else:
                print(f"  (No previous context)")
            
            print(f"\n--- Current Turn ---")
            current_turn = p['current_turn']
            print(f"Speaker: {current_turn.get('speaker', '?')}")
            print(f"Text: {current_turn.get('text', '')}")
            
            print(f"\n--- Decision ---")
            print(f"Ground Truth: {p['ground_truth']}")
            print(f"Extracted Prediction: {p['prediction']}")
            print(f"Match: WRONG")
            
            print(f"\n--- Model Output ---")
            print(f"Full output: {p['output_text']}")
            if '</decision>' in p['output_text']:
                decision_end = p['output_text'].find('</decision>')
                decision_content = p['output_text'][:decision_end].strip()
                print(f"Extracted from: '{decision_content}' → {p['prediction']}")
            print(f"{'='*70}")

    return all_predictions



def main(
    dataset: str = 'ami',
    debug_prompts: bool = False,
    model: Optional[str] = None,
    system_prompt_repeat: Optional[int] = None,
    filter_no_context: bool = True,
):
    global SYSTEM_PROMPT_REPEAT
    if system_prompt_repeat is not None:
        SYSTEM_PROMPT_REPEAT = system_prompt_repeat

    DATA_DIR = BASE_DIR / 'data' / dataset
    TEST_FILE = DATA_DIR / 'test' / 'test_samples.jsonl'
    VAL_FILE = DATA_DIR / 'val' / 'val_samples.jsonl'
    EVAL_FILE = TEST_FILE

    model_key = model if model is not None else DEFAULT_MODEL
    model_id = MODEL_OPTIONS.get(model_key) or DEFAULT_BASE_MODEL
    if model_key not in MODEL_OPTIONS:
        model_key = next((k for k, v in MODEL_OPTIONS.items() if v == model_id), model_key)

    RESULTS_DIR = Path(__file__).parent / 'results'
    REPORTS_DIR = RESULTS_DIR / 'reports'
    RESULTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    sp_val = system_prompt_repeat if system_prompt_repeat is not None else SYSTEM_PROMPT_REPEAT
    sp_suffix = f"_sp{sp_val}"
    PREDICTIONS_FILE = RESULTS_DIR / f"baseline_predictions_{dataset}_{model_key}{sp_suffix}.json"

    print("="*70)
    print(f"BASELINE EVALUATION: {model_key} ({model_id})")
    print(f"Dataset: {dataset.upper()}")
    print("="*70)

    print("\nLoading model...")
    loader = InstructModelLoader(model_id=model_id, use_vllm=USE_VLLM)
    model_result = loader.load()
    model = model_result['model']
    tokenizer = model_result['tokenizer']
    use_vllm = model_result['use_vllm']

    print(f"\nLoading samples from {EVAL_FILE}...")
    samples = load_samples(EVAL_FILE)
    print(f"Loaded {len(samples)} samples")
    if filter_no_context:
        n_before = len(samples)
        samples = filter_samples_with_context(samples)
        n_removed = n_before - len(samples)
        print(f"Filtered to samples with context_turns: {len(samples)} (removed {n_removed} with no context)")
    if not samples:
        print("No samples left after filtering. Exiting.")
        return None

    all_predictions = evaluate_samples(
        samples, model, tokenizer, use_vllm, BATCH_SIZE,
        debug_prompts=debug_prompts, model_id=model_id,
    )

    payload = {
        "dataset": dataset,
        "model_key": model_key,
        "model_id": model_id,
        "system_prompt_repeat": sp_val,
        "predictions": all_predictions,
    }
    print(f"\nSaving predictions to {PREDICTIONS_FILE}...")
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print("Predictions saved.")

    metrics = compute_metrics(all_predictions)
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nTotal samples: {metrics['total_samples']:,}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Macro accuracy: {metrics.get('macro_accuracy', 0):.2%}")
    print(f"Correct: {metrics['correct']:,}")
    print(f"Incorrect: {metrics['incorrect']:,}")
    print(f"\nSpeak  P/R/F1: {metrics.get('precision_speak', 0):.2%} / {metrics.get('recall_speak', 0):.2%} / {metrics.get('f1_speak', 0):.2%}")
    print(f"Silent P/R/F1: {metrics.get('precision_silent', 0):.2%} / {metrics.get('recall_silent', 0):.2%} / {metrics.get('f1_silent', 0):.2%}")
    print(f"Macro F1: {metrics.get('macro_f1', 0):.2%}")
    print(f"\nFalse Positive Rate (SILENT -> SPEAK): {metrics['false_positive_rate']:.2%}")
    print(f"False Negative Rate (SPEAK -> SILENT): {metrics['false_negative_rate']:.2%}")
    print(f"\nLatency Statistics:")
    for stat, value in metrics['latency_stats'].items():
        print(f"  {stat}: {value:.4f}s")
    print(f"\nPer-Category Accuracy:")
    for cat in sorted(metrics.get('category_accuracy', {}).keys()):
        m = metrics['category_accuracy'][cat]
        print(f"  {cat}: {m['accuracy']:.2%} ({m['correct']}/{m['total']})")

    report = generate_detail_report({**metrics, "predictions": all_predictions})
    report_path = REPORTS_DIR / f"baseline_analysis_{dataset}_{model_key}_sp{sp_val}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    print("\n" + "="*70)
    print("BASELINE EVALUATION COMPLETE")
    print("="*70)

    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate baseline model')
    parser.add_argument('--dataset', type=str, default='ami',
                        choices=['ami', 'friends', 'spgi'],
                        help='Dataset name (default: ami)')
    parser.add_argument('--model', type=str, default=None,
                        choices=list(MODEL_OPTIONS.keys()),
                        help=f'Model key (default: from MODEL env or {DEFAULT_MODEL!r})')
    parser.add_argument('--debug-prompts', action='store_true',
                        help='Print full system prompt, instruction, and input for first 3-4 samples')
    parser.add_argument('--system-prompt-repeat', type=int, default=None, choices=[1, 2],
                        help='Repeat system prompt 1 or 2 times (default: use module default)')
    parser.add_argument('--filter-no-context', action='store_true', default=True,
                        help='Exclude samples with no context_turns from evaluation (default: True)')
    parser.add_argument('--no-filter-no-context', action='store_false', dest='filter_no_context',
                        help='Do not filter; include samples with no context_turns')
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        debug_prompts=args.debug_prompts,
        model=args.model,
        system_prompt_repeat=args.system_prompt_repeat,
        filter_no_context=args.filter_no_context,
    )
