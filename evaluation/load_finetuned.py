#!/usr/bin/env python3
"""
Load fine-tuned LoRA model for evaluation. Uses config for base model and adapter path.
Supports loading a specific checkpoint (e.g. checkpoint-2000) via adapter_path.
"""

import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from fine_tuning.config import BASE_MODEL, FINAL_MODEL_DIR


def load_finetuned_model_vllm(adapter_path: Optional[Path] = None):
    from vllm import LLM
    from transformers import AutoTokenizer
    import os
    
    adapter_dir = adapter_path if adapter_path is not None else FINAL_MODEL_DIR
    print("\nLoading fine-tuned model with vLLM...")
    print(f"Base model: {BASE_MODEL}")
    print(f"LoRA adapters: {adapter_dir}")
    
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found at {adapter_dir}. "
            "Please run fine_tuning/train_lora.py first or use --checkpoint <name> for a saved checkpoint (e.g. checkpoint-2000)."
        )
    
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    print("\nLoading model with vLLM (native LoRA support)...")
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=16384,  # Match the training config
        gpu_memory_utilization=0.75,  # Reduced from 0.85 to save memory
        max_num_seqs=64,  # Reduced from 128 to avoid OOM during warmup
        trust_remote_code=True,
        enable_lora=True,  # Enable LoRA support
        max_lora_rank=16,  # Maximum LoRA rank
        max_loras=1,  # Number of LoRA adapters
    )
    
    adapter_name = "finetuned_lora"
    lora_loaded = False
    
    try:
        if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'add_lora'):
            llm.llm_engine.add_lora(adapter_name, str(adapter_dir))
            lora_loaded = True
            print(f"LoRA adapter '{adapter_name}' loaded successfully via llm_engine.add_lora")
        elif hasattr(llm, 'add_lora'):
            llm.add_lora(adapter_name, str(adapter_dir))
            lora_loaded = True
            print(f"LoRA adapter '{adapter_name}' loaded successfully via add_lora")
        else:
            print("Warning: Could not find LoRA loading method.")
            print(f"Available methods with 'lora': {[m for m in dir(llm) if 'lora' in m.lower()]}")
            if hasattr(llm, 'llm_engine'):
                print(f"Engine methods with 'lora': {[m for m in dir(llm.llm_engine) if 'lora' in m.lower()]}")
            adapter_name = None
    except Exception as e:
        print(f"Warning: Could not load LoRA adapter: {e}")
        import traceback
        traceback.print_exc()
        print("Will proceed with base model (LoRA not applied, but vLLM is still faster)")
        adapter_name = None
        lora_loaded = False
    
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=hf_token,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    llm._lora_adapter_name = adapter_name
    llm._lora_adapter_path = str(adapter_dir)
    
    print("\nModel loaded successfully with vLLM")
    if adapter_name:
        print(f"  LoRA adapter: {adapter_name}")
    return llm, tokenizer


def load_finetuned_model(
    use_vllm: bool = True,
    use_merged: bool = False,
    merged_model_path: Optional[Path] = None,
    adapter_path: Optional[Path] = None,
):
    """
    Load fine-tuned LoRA model. adapter_path overrides the default (FINAL_MODEL_DIR),
    e.g. to evaluate a specific checkpoint: adapter_path=Path(".../checkpoint-2000").
    """
    print("="*70)
    print("LOADING FINE-TUNED MODEL")
    print("="*70)
    
    if use_vllm:
        try:
            return load_finetuned_model_vllm(adapter_path=adapter_path)
        except Exception as e:
            print(f"\nWarning: Failed to load with vLLM: {e}")
            print("Falling back to transformers...")
    
    adapter_dir = adapter_path if adapter_path is not None else FINAL_MODEL_DIR
    
    if use_merged and merged_model_path:
        print(f"\nLoading merged model from: {merged_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            str(merged_model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(merged_model_path),
            trust_remote_code=True,
        )
    else:
        print(f"\nLoading base model: {BASE_MODEL}")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found at {adapter_dir}. "
                "Please run fine_tuning/train_lora.py first or use --checkpoint <name> for a saved checkpoint (e.g. checkpoint-2000)."
            )
        
        print(f"Loading LoRA adapters from: {adapter_dir}")
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer.padding_side = "left"
    print("\nModel loaded successfully")
    return model, tokenizer


def main():
    try:
        model, tokenizer = load_finetuned_model()
        print("\n" + "="*70)
        print("MODEL LOADING TEST COMPLETE")
        print("="*70)
        return model, tokenizer
    except Exception as e:
        print(f"\nError loading model: {e}")
        raise


if __name__ == '__main__':
    main()
