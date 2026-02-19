#!/usr/bin/env python3
"""
LoRA fine-tuning config. MODEL and DATASET are read from the environment at import time.
If MODEL is set, it must be one of MODEL_OPTIONS keys; otherwise ImportError/ValueError can occur
when any code imports this module (e.g. run_benchmark, train_lora, evaluate_finetuned).
"""

import os
from pathlib import Path

MODEL_OPTIONS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3-4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "llama3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
}

_MODEL_KEY = os.environ.get("MODEL", "qwen2.5-7b").strip().lower()
if _MODEL_KEY not in MODEL_OPTIONS:
    raise ValueError(
        f"Unknown MODEL='{_MODEL_KEY}'. Choose one of: {list(MODEL_OPTIONS.keys())}"
    )
MODEL = _MODEL_KEY
BASE_MODEL = MODEL_OPTIONS[MODEL]

DATASET = os.environ.get("DATASET", "ami")

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
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
    "per_device_train_batch_size": 12,
    "per_device_eval_batch_size": 12,
    "gradient_accumulation_steps": 2,
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
DATA_DIR = BASE_DIR / "data" / DATASET
TRAIN_FILE = DATA_DIR / "train" / "train_samples.jsonl"
VAL_FILE = DATA_DIR / "val" / "val_samples.jsonl"

OUTPUT_DIR = Path(__file__).parent / "checkpoints" / MODEL
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_DIR = OUTPUT_DIR / "final_model"
