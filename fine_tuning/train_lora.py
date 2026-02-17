#!/usr/bin/env python3
"""
LoRA fine-tuning for context-aware turn-taking. Uses config for model, dataset, paths.
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import torch


def setup_model_and_tokenizer():
    print("="*70)
    print("LOADING BASE MODEL")
    print("="*70)
    print(f"Model: {BASE_MODEL}")
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    
    print("Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    model.train()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    if trainable_params == 0:
        raise RuntimeError("No trainable parameters found! LoRA adapters may not be properly configured.")
    
    print("\nModel setup complete")
    return model, tokenizer



def compute_metrics(eval_pred):
    import numpy as np
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    mask = labels != -100
    matches = (predictions == labels) & mask
    accuracy = np.sum(matches) / np.sum(mask) if np.sum(mask) > 0 else 0.0
    return {"accuracy": accuracy}



def main(dataset: str = "ami"):
    os.environ["DATASET"] = dataset
    if "config" in sys.modules:
        importlib.reload(sys.modules["config"])
    from config import (
        BASE_MODEL,
        LORA_CONFIG,
        TRAINING_CONFIG,
        OUTPUT_DIR,
        FINAL_MODEL_DIR,
        DATASET,
        TRAIN_FILE,
        VAL_FILE
    )
    from data_loader import prepare_datasets, data_collator
    
    print("="*70)
    print(f"LORA FINE-TUNING FOR CONTEXT-AWARE TURN-TAKING")
    print(f"Dataset: {DATASET}")
    print("="*70)
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer()
    
    # Prepare datasets
    print("\n" + "="*70)
    print("PREPARING DATASETS")
    print("="*70)
    print(f"Train file: {TRAIN_FILE}")
    print(f"Val file: {VAL_FILE}")
    
    DEBUG = os.environ.get("DEBUG_TRAINING", "false").lower() == "true"
    train_dataset, val_dataset = prepare_datasets(tokenizer, max_length=384, debug=DEBUG)
    
    # Training arguments
    print("\n" + "="*70)
    print("SETTING UP TRAINING")
    print("="*70)
    
    training_config = {k: v for k, v in TRAINING_CONFIG.items() if k != 'output_dir'}
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        **training_config
    )
    
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Training epochs: {TRAINING_CONFIG['num_train_epochs']}")
    print(f"Batch size: {TRAINING_CONFIG['per_device_train_batch_size']}")
    print(f"Gradient accumulation: {TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"Effective batch size: {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"Learning rate: {TRAINING_CONFIG['learning_rate']}")
    
    data_collator_obj = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator_obj,
        compute_metrics=compute_metrics,
    )
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    trainer.train()
    
    print("\n" + "="*70)
    print("SAVING FINAL MODEL")
    print("="*70)
    
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))
    
    print(f"Model saved to {FINAL_MODEL_DIR}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune model with LoRA")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ami",
        choices=["ami", "friends", "spgi"],
        help="Dataset name (default: ami)",
    )
    args = parser.parse_args()
    main(dataset=args.dataset)
