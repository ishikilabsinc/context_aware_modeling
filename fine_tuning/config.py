#!/usr/bin/env python3
"""
Fine-Tuning Configuration

Configuration for LoRA fine-tuning of QWEN model for context-aware turn-taking.
"""

from pathlib import Path


# Base model (should match the one used in benchmarking)
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# LoRA hyperparameters
LORA_CONFIG = {
    "r": 8,  # Reduced from 16 for faster training (less parameters, but less capacity)
    "lora_alpha": 16,  # Reduced proportionally (alpha = 2*r is common)
    "lora_dropout": 0.1,
    "target_modules": [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

TRAINING_CONFIG = {
    "output_dir": "checkpoints",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 12,  # Optimized for 384 token sequences
    "per_device_eval_batch_size": 12,
    "gradient_accumulation_steps": 2,  # Effective batch size: 12*2=24 (slightly larger for faster convergence)
    "learning_rate": 3e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 100,
    "logging_steps": 100,
    "save_steps": 2000,
    "eval_steps": 2000,
    "save_total_limit": 3,
    "eval_strategy": "steps",
    "save_strategy": "steps",
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "greater_is_better": True,
    "fp16": False,
    "bf16": True,
    "gradient_checkpointing": False,
    "optim": "adamw_torch",
    "report_to": "none",
    "seed": 42,
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
    "remove_unused_columns": False,
}


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'

TRAIN_FILE = DATA_DIR / 'train' / 'train_samples.jsonl'
VAL_FILE = DATA_DIR / 'val' / 'val_samples.jsonl'


OUTPUT_DIR = Path(__file__).parent / 'checkpoints'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Final model output
FINAL_MODEL_DIR = OUTPUT_DIR / 'final_model'
