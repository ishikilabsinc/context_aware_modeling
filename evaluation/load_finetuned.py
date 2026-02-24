#!/usr/bin/env python3
"""Load fine-tuned LoRA model for evaluation."""

import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from fine_tuning.config import BASE_MODEL, FINAL_MODEL_DIR


def load_finetuned_model_vllm(
    adapter_path: Optional[Path] = None,
    lora_rank: Optional[int] = None,
):
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
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"LoRA adapter config not found at {adapter_dir} (missing adapter_config.json). "
            "If you trained with FSDP and final_model was not saved, run train_lora.py with "
            "--resume-from-checkpoint <path-to-checkpoint> (same --dataset, --lora-rank, --fsdp as training). "
            "Then re-run evaluation."
        )

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    max_lora_rank = max(64, (lora_rank or 0) * 2) if lora_rank else 64

    print("\nLoading model with vLLM (native LoRA support)...")
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=16384,
        gpu_memory_utilization=0.75,
        max_num_seqs=64,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        max_loras=1,
    )

    adapter_name = "finetuned_lora"
    llm._lora_adapter_name = adapter_name
    llm._lora_adapter_path = str(adapter_dir)
    print(f"LoRA adapter will be applied at request time from: {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=hf_token,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("\nModel loaded successfully with vLLM")
    print(f"  LoRA adapter: {adapter_name} ({adapter_dir})")
    return llm, tokenizer


def load_finetuned_model(
    use_vllm: bool = True,
    use_merged: bool = False,
    merged_model_path: Optional[Path] = None,
    adapter_path: Optional[Path] = None,
    lora_rank: Optional[int] = None,
):
    print("="*70)
    print("LOADING FINE-TUNED MODEL")
    print("="*70)

    if use_vllm:
        try:
            return load_finetuned_model_vllm(
                adapter_path=adapter_path,
                lora_rank=lora_rank,
            )
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
        adapter_config = adapter_dir / "adapter_config.json"
        if not adapter_config.is_file():
            raise FileNotFoundError(
                f"LoRA adapter config not found at {adapter_dir} (missing adapter_config.json). "
                "If you trained with FSDP, re-run training with the updated train_lora.py so the final model is saved with FULL_STATE_DICT. "
                "Alternatively evaluate with vLLM: python evaluation/evaluate_finetuned.py (uses vLLM by default)."
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
