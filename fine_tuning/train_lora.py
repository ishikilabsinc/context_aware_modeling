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

from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import torch


def setup_model_and_tokenizer(lora_rank=None, use_fsdp=False):
    """lora_rank: if set, overrides LORA_CONFIG['r'] for this run (e.g. 32 or 64).
    use_fsdp: if True, load model on CPU so FSDP can shard across GPUs (use with accelerate launch).
    """
    from config import LORA_CONFIG, BASE_MODEL
    r = lora_rank if lora_rank is not None else LORA_CONFIG["r"]
    print("="*70)
    print("LOADING BASE MODEL")
    print("="*70)
    print(f"Model: {BASE_MODEL}")
    if use_fsdp:
        print("FSDP: enabled (model loaded on CPU for sharding)")
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Loading model...")
    try:
        _attn_impl = "flash_attention_2"
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            attn_implementation=_attn_impl,
            device_map="cpu" if use_fsdp else "auto",
            low_cpu_mem_usage=True if use_fsdp else False,
            trust_remote_code=True,
        )
        print(f"  Using attention: {_attn_impl}")
    except (ValueError, ImportError, AssertionError) as e:
        if "flash_attention" in str(e).lower() or "flash_attn" in str(e).lower():
            print("  Flash Attention 2 not available, trying SDPA.")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    device_map="cpu" if use_fsdp else "auto",
                    low_cpu_mem_usage=True if use_fsdp else False,
                    trust_remote_code=True,
                )
                print("  Using attention: sdpa")
            except (ValueError, ImportError, AssertionError):
                print("  SDPA not available, using default attention.")
                model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu" if use_fsdp else "auto",
                    low_cpu_mem_usage=True if use_fsdp else False,
                    trust_remote_code=True,
                )
        else:
            raise
    
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    
    print("Configuring LoRA...")
    print(f"  LoRA rank: {r}")
    lora_config = LoraConfig(
        r=r,
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


class EvalSummaryCallback(TrainerCallback):
    """Prints one line per eval: step, train loss, eval loss, eval accuracy."""

    def __init__(self):
        self.last_train_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_train_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        step = state.global_step
        train_loss = self.last_train_loss if self.last_train_loss is not None else "-"
        eval_loss = metrics.get("eval_loss", "-")
        eval_acc = metrics.get("eval_accuracy", metrics.get("accuracy", "-"))
        print(f"  [check] Step {step} | train_loss={train_loss} | eval_loss={eval_loss} | eval_acc={eval_acc}")


class TrainerWithBalancedBatches(Trainer):
    """Trainer that uses a custom train dataloader with balanced SPEAK/SILENT batches when provided."""

    def __init__(self, train_batch_sampler=None, **kwargs):
        super().__init__(**kwargs)
        self.train_batch_sampler = train_batch_sampler

    def get_train_dataloader(self):
        if self.train_batch_sampler is not None:
            return DataLoader(
                self.train_dataset,
                batch_sampler=self.train_batch_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        return super().get_train_dataloader()


def main(dataset: str = "ami", lora_rank=None, use_fsdp=False, max_length: int = 256):
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
        VAL_FILE,
        MODEL,
    )
    from data_loader import prepare_datasets, data_collator

    # When using a custom LoRA rank, save to a separate dir so runs don't overwrite (e.g. checkpoints/qwen2.5-7b_r32)
    if lora_rank is not None:
        run_output_dir = Path(__file__).resolve().parent / "checkpoints" / f"{MODEL}_r{lora_rank}"
        run_final_model_dir = run_output_dir / "final_model"
        run_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_output_dir = OUTPUT_DIR
        run_final_model_dir = FINAL_MODEL_DIR

    print("="*70)
    print(f"LORA FINE-TUNING FOR CONTEXT-AWARE TURN-TAKING")
    print(f"Dataset: {DATASET}")
    if lora_rank is not None:
        print(f"LoRA rank: {lora_rank} (output: {run_output_dir})")
    if use_fsdp:
        print("FSDP: enabled (run with 'accelerate launch' for multi-GPU sharding)")
    print(f"Max sequence length: {max_length} (use --max-length 512 or 1024 for fair benchmark comparison)")
    print("="*70)

    # Setup model (use config rank if lora_rank not set; load on CPU if FSDP)
    model, tokenizer = setup_model_and_tokenizer(lora_rank=lora_rank, use_fsdp=use_fsdp)

    # Prepare datasets
    print("\n" + "="*70)
    print("PREPARING DATASETS")
    print("="*70)
    if DATASET == "all":
        print("Train/val: combined from ami, friends, spgi")
    else:
        print(f"Train file: {TRAIN_FILE}")
        print(f"Val file: {VAL_FILE}")

    batch_size = TRAINING_CONFIG["per_device_train_batch_size"]
    DEBUG = os.environ.get("DEBUG_TRAINING", "false").lower() == "true"
    # Balanced SPEAK/SILENT batches: single-GPU uses BalancedBatchSampler; FSDP uses DistributedBalancedBatchSampler
    rank = None
    world_size = None
    if use_fsdp:
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            # accelerate/torchrun set these before the script runs; use them so we partition data across ranks
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
    train_dataset, val_dataset, train_batch_sampler = prepare_datasets(
        tokenizer,
        max_length=max_length,
        debug=DEBUG,
        batch_size=batch_size,
        use_balanced_batches=True,
        rank=rank,
        world_size=world_size,
    )

    # Training arguments
    print("\n" + "="*70)
    print("SETTING UP TRAINING")
    print("="*70)

    training_config = {k: v for k, v in TRAINING_CONFIG.items() if k != "output_dir"}
    if use_fsdp:
        # FSDP shards model/gradients/optimizer across GPUs to reduce OOM
        training_config["fsdp"] = ["full_shard"]
        # Trainer's gradient_checkpointing breaks autograd with FSDP; use fsdp_activation_checkpointing in accelerate config instead
        training_config["gradient_checkpointing"] = False
    # Fused optimizer requires CUDA; fall back to avoid errors on CPU / unsupported envs
    if training_config.get("optim") == "adamw_torch_fused" and not torch.cuda.is_available():
        training_config["optim"] = "adamw_torch"

    training_args = TrainingArguments(
        output_dir=str(run_output_dir),
        **training_config
    )
    
    print(f"Output directory: {run_output_dir}")
    print(f"Training epochs: {TRAINING_CONFIG['num_train_epochs']}")
    print(f"Batch size: {TRAINING_CONFIG['per_device_train_batch_size']}")
    print(f"Gradient accumulation: {TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"Effective batch size: {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"Learning rate: {TRAINING_CONFIG['learning_rate']}")
    
    data_collator_obj = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = TrainerWithBalancedBatches(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator_obj,
        compute_metrics=compute_metrics,
        train_batch_sampler=train_batch_sampler,
        callbacks=[EvalSummaryCallback()],
    )
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    trainer.train()

    # Plot training curve (loss / eval loss / lr) for stability inspection
    try:
        from plot_training_curve import plot_training_curve
        plot_training_curve(
            trainer.state.log_history,
            run_output_dir / "training_curve.png",
            title=f"Training curve - {DATASET}" + (f" (r={lora_rank})" if lora_rank else "") + (f" max_len={max_length}" if max_length != 256 else ""),
        )
    except Exception as e:
        print(f"Could not plot training curve: {e}")

    print("\n" + "="*70)
    print("SAVING FINAL MODEL")
    print("="*70)

    run_final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(run_final_model_dir))
    tokenizer.save_pretrained(str(run_final_model_dir))

    print(f"Model saved to {run_final_model_dir}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune model with LoRA")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ami",
        choices=["ami", "friends", "spgi", "all"],
        help="Dataset: ami, friends, spgi, or 'all' to combine train/val from all three (default: ami)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        metavar="N",
        help="LoRA rank (default: from config, e.g. 8). Use 32 or 64 for more capacity; output saved to checkpoints/<model>_r<N>/",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="Enable FSDP (Fully Sharded Data Parallel) to reduce GPU memory. Run with: accelerate launch fine_tuning/train_lora.py --fsdp ...",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        metavar="N",
        help="Max sequence length (default: 256 for fast runs). Use 512 or 1024 for fair benchmark-aligned training (may require smaller batch size).",
    )
    args = parser.parse_args()
    main(dataset=args.dataset, lora_rank=args.lora_rank, use_fsdp=args.fsdp, max_length=args.max_length)
