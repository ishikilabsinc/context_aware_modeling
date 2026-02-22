#!/usr/bin/env python3
"""
LoRA fine-tuning for context-aware turn-taking.

- Normal run: trains and saves the PEFT adapter to checkpoints/<model>_r<N>/final_model/
  (with FSDP, adapter is saved at the end after gathering full state).
- If training was interrupted or final_model is missing: use --resume-from-checkpoint
  pointing at a checkpoint dir (e.g. .../checkpoint-1008). The script loads that checkpoint,
  does zero new steps, and saves the adapter to final_model. Same command as training,
  with --resume-from-checkpoint added.
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Avoid tokenizer fork warning when DataLoader spawns workers (set before any tokenizer use)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
try:
    from transformers import EarlyStoppingCallback
except ImportError:
    EarlyStoppingCallback = None
from peft import LoraConfig, get_peft_model, TaskType
import torch


def setup_model_and_tokenizer(lora_rank=None, use_fsdp=False, local_rank=-1):
    """lora_rank: if set, overrides LORA_CONFIG['r'] for this run (e.g. 32 or 64).
    use_fsdp: if True, load model on CPU so FSDP can shard across GPUs (use with accelerate launch).
    local_rank: when >= 0 (DDP), load model onto cuda:local_rank so Accelerator can train; avoids device_map='auto' in distributed.
    """
    from config import LORA_CONFIG, BASE_MODEL
    r = lora_rank if lora_rank is not None else LORA_CONFIG["r"]
    # DDP: one GPU per process; FSDP: CPU then shard; single-GPU: auto
    if use_fsdp:
        device_map = "cpu"
        low_cpu = True
    elif local_rank >= 0:
        device_map = {"": f"cuda:{local_rank}"}
        low_cpu = False
    else:
        device_map = "auto"
        low_cpu = False

    print("="*70)
    print("LOADING BASE MODEL")
    print("="*70)
    print(f"Model: {BASE_MODEL}")
    if use_fsdp:
        print("FSDP: enabled (model loaded on CPU for sharding)")
    elif local_rank >= 0:
        print(f"DDP: loading model onto cuda:{local_rank}")

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
            dtype=torch.bfloat16,
            attn_implementation=_attn_impl,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu,
            trust_remote_code=True,
        )
        print(f"  Using attention: {_attn_impl}")
    except (ValueError, ImportError, AssertionError) as e:
        if "flash_attention" in str(e).lower() or "flash_attn" in str(e).lower():
            print("  Flash Attention 2 not available, trying SDPA.")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL,
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    device_map=device_map,
                    low_cpu_mem_usage=low_cpu,
                    trust_remote_code=True,
                )
                print("  Using attention: sdpa")
            except (ValueError, ImportError, AssertionError):
                print("  SDPA not available, using default attention.")
                model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL,
                    dtype=torch.bfloat16,
                    device_map=device_map,
                    low_cpu_mem_usage=low_cpu,
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



def _make_compute_metrics(tokenizer):
    """Build compute_metrics that uses tokenizer to find decision token and compute accuracy, macro F1, balanced accuracy.
    Accepts predictions as full logits [N,L,V], pred_tokens [N,L], or 1D class indices [N] (0=SILENT, 1=SPEAK) from our custom prediction_step."""
    import numpy as np
    speak_ids = tokenizer.encode("SPEAK", add_special_tokens=False)
    silent_ids = tokenizer.encode("SILENT", add_special_tokens=False)
    speak_id = int(speak_ids[0]) if speak_ids else -1
    silent_id = int(silent_ids[0]) if silent_ids else -1

    def compute_metrics(eval_pred):
        if hasattr(eval_pred, "predictions"):
            predictions, labels = eval_pred.predictions, eval_pred.label_ids
        else:
            predictions, labels = eval_pred[0], eval_pred[1]
        labels = np.asarray(labels)
        pred_arr = np.asarray(predictions)
        n = labels.shape[0]
        # 1D predictions and 1D labels: from our custom prediction_step (0/1 per sample); DDP gathers these correctly
        if pred_arr.ndim == 1 and labels.ndim == 1 and len(labels) == len(pred_arr):
            y_pred = pred_arr.ravel()
            y_true = labels.ravel().astype(np.int64)
        elif pred_arr.ndim == 1:
            # Legacy: 2D labels, extract y_true from decision position
            y_pred = pred_arr.ravel()
            seq_len = labels.shape[1]
            y_true = []
            for i in range(n):
                pos = None
                for j in range(seq_len - 1, -1, -1):
                    if labels[i, j] == speak_id or labels[i, j] == silent_id:
                        pos = j
                        break
                if pos is None:
                    continue
                y_true.append(1 if labels[i, pos] == speak_id else 0)
            y_true = np.array(y_true) if y_true else np.array([])
            if len(y_true) != len(y_pred):
                y_pred = y_pred[: len(y_true)]
        else:
            if hasattr(predictions, "ndim") and predictions.ndim == 3:
                pred_tokens = np.argmax(predictions, axis=-1)
            else:
                pred_tokens = np.asarray(predictions)
            seq_len = labels.shape[1]
            y_true, y_pred = [], []
            for i in range(n):
                pos = None
                for j in range(seq_len - 1, -1, -1):
                    if labels[i, j] == speak_id or labels[i, j] == silent_id:
                        pos = j
                        break
                if pos is None:
                    continue
                y_true.append(1 if labels[i, pos] == speak_id else 0)
                y_pred.append(1 if pred_tokens[i, pos] == speak_id else 0)
            y_true = np.array(y_true) if y_true else np.array([])
            y_pred = np.array(y_pred) if y_pred else np.array([])
        if len(y_true) == 0:
            return {"accuracy": 0.0, "macro_f1": 0.0, "balanced_accuracy": 0.0}
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        # Safeguard: if only one class in gathered eval set, metrics are misleading (val set balance or label bug)
        n_classes = len(np.unique(y_true))
        if n_classes == 1 and os.environ.get("LOCAL_RANK", "0") == "0":
            import logging
            logging.warning(
                "Eval set has only one class (n=%d). Check val set balance and that labels use the answer position, not the format line.",
                len(y_true),
            )
        accuracy = np.mean(y_true == y_pred)
        # Macro F1: 1 = SPEAK, 0 = SILENT
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        prec_speak = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_speak = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_speak = 2 * prec_speak * rec_speak / (prec_speak + rec_speak) if (prec_speak + rec_speak) > 0 else 0.0
        prec_silent = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        rec_silent = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_silent = 2 * prec_silent * rec_silent / (prec_silent + rec_silent) if (prec_silent + rec_silent) > 0 else 0.0
        macro_f1 = (f1_speak + f1_silent) / 2.0
        balanced_accuracy = (rec_speak + rec_silent) / 2.0
        return {"accuracy": float(accuracy), "macro_f1": float(macro_f1), "balanced_accuracy": float(balanced_accuracy)}

    return compute_metrics


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
        # Print only on rank 0 to avoid duplicate lines under DDP
        if os.environ.get("LOCAL_RANK", "0") != "0":
            return
        step = state.global_step
        train_loss = self.last_train_loss if self.last_train_loss is not None else "-"
        eval_loss = metrics.get("eval_loss", "-")
        eval_acc = metrics.get("eval_accuracy", metrics.get("accuracy", "-"))
        eval_macro_f1 = metrics.get("eval_macro_f1", metrics.get("macro_f1", "-"))
        eval_bal_acc = metrics.get("eval_balanced_accuracy", metrics.get("balanced_accuracy", "-"))
        print(f"  [check] Step {step} | train_loss={train_loss} | eval_loss={eval_loss} | eval_acc={eval_acc} | macro_f1={eval_macro_f1} | balanced_acc={eval_bal_acc}")


class TrainerWithBalancedBatches(Trainer):
    """Trainer that uses a custom train dataloader with balanced SPEAK/SILENT batches when provided.
    Overrides prediction_step to return only decision-token class (0/1) per sample instead of full logits,
    so evaluation does not OOM when concatenating predictions over 20k+ val samples."""

    def __init__(self, train_batch_sampler=None, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.train_batch_sampler = train_batch_sampler
        self._tokenizer = tokenizer
        self._speak_id = None
        self._silent_id = None
        if tokenizer is not None:
            speak_ids = tokenizer.encode("SPEAK", add_special_tokens=False)
            silent_ids = tokenizer.encode("SILENT", add_special_tokens=False)
            self._speak_id = int(speak_ids[0]) if speak_ids else -1
            self._silent_id = int(silent_ids[0]) if silent_ids else -1

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Run forward, then return (loss, pred_classes, labels) with pred_classes shape [B] to avoid OOM from storing full logits."""
        if self._tokenizer is None or self._speak_id is None or self._silent_id is None:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
        labels = inputs.get("labels")
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss if outputs.loss is not None else None
            logits = outputs.logits  # [B, L, V]
        if prediction_loss_only or logits is None or labels is None:
            return (loss, None, None)
        # Move to same device for indexing
        labels = labels.to(logits.device)
        B, L, V = logits.shape
        speak_id = self._speak_id
        silent_id = self._silent_id
        pred_classes = []
        true_classes = []
        for i in range(B):
            pos = None
            for j in range(L - 1, -1, -1):
                if labels[i, j].item() == speak_id or labels[i, j].item() == silent_id:
                    pos = j
                    break
            if pos is None:
                pred_classes.append(0)
                true_classes.append(0)
                continue
            true_classes.append(1 if labels[i, pos].item() == speak_id else 0)
            # 1 = SPEAK if logits[i,pos,speak_id] >= logits[i,pos,silent_id] else 0 = SILENT
            if speak_id >= 0 and silent_id >= 0 and speak_id < V and silent_id < V:
                pred_classes.append(1 if logits[i, pos, speak_id].item() >= logits[i, pos, silent_id].item() else 0)
            else:
                pred_classes.append(0)
        # Return 1D tensors so DDP gathers them correctly; 2D labels were causing single-class eval bug
        out = torch.tensor(pred_classes, dtype=torch.long, device=logits.device)
        labels_out = torch.tensor(true_classes, dtype=torch.long, device=logits.device)
        return (loss, out, labels_out)

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


def main(
    dataset: str = "ami",
    lora_rank=None,
    use_fsdp=False,
    max_length: int = 0,
    train_fraction: float = 1.0,
    val_fraction: float = 1.0,
    resume_from_checkpoint: Optional[str] = None,
    debug: bool = False,
):
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
    from data_loader import prepare_datasets, make_data_collator

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
    if max_length is None or max_length <= 0:
        print("Max sequence length: none (no truncation; full examples)")
    else:
        print(f"Max sequence length: {max_length} (optional cap for memory; use --max-length 0 for no truncation)")
    if train_fraction < 1.0 or val_fraction < 1.0:
        print(f"Data subset: train_fraction={train_fraction}, val_fraction={val_fraction}")
    print("="*70)

    # Setup model (use config rank if lora_rank not set; load on CPU if FSDP, else cuda:LOCAL_RANK for DDP)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if not use_fsdp and local_rank >= 0 and max_length is not None and max_length > 1536:
        if local_rank == 0:
            print(f"[DDP] Capping max_length {max_length} -> 1536 to fit 22GB GPUs (use single-GPU or FSDP for longer contexts).")
        max_length = 1536
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    model, tokenizer = setup_model_and_tokenizer(lora_rank=lora_rank, use_fsdp=use_fsdp, local_rank=local_rank)

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
    if max_length is None or max_length <= 0:
        batch_size = 1
    elif not use_fsdp and local_rank >= 0 and max_length >= 512:
        batch_size = 1
    debug = debug or (os.environ.get("DEBUG_TRAINING", "false").lower() == "true")
    rank = None
    world_size = None
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            rank = int(os.environ.get("RANK", 0))
    train_dataset, val_dataset, train_batch_sampler = prepare_datasets(
        tokenizer,
        max_length=max_length,
        debug=debug,
        batch_size=batch_size,
        use_balanced_batches=True,
        rank=rank,
        world_size=world_size,
        silent_ratio_in_batch=TRAINING_CONFIG.get("silent_ratio_in_batch", 0.5),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        debug_sample_io_path=str(run_output_dir / "debug_sample_io.txt") if debug else None,
    )

    # Training arguments
    print("\n" + "="*70)
    print("SETTING UP TRAINING")
    print("="*70)

    training_config = {k: v for k, v in TRAINING_CONFIG.items() if k not in ("output_dir", "silent_ratio_in_batch")}
    if use_fsdp:
        training_config["fsdp"] = ["full_shard"]
        training_config["gradient_checkpointing"] = False
    elif local_rank >= 0:
        training_config["gradient_checkpointing"] = False
        if max_length is None or max_length <= 0 or max_length >= 512:
            training_config["per_device_train_batch_size"] = 1
            training_config["gradient_accumulation_steps"] = max(training_config["gradient_accumulation_steps"], 40)
    # Fused optimizer requires CUDA; fall back to avoid errors on CPU / unsupported envs
    if training_config.get("optim") == "adamw_torch_fused" and not torch.cuda.is_available():
        training_config["optim"] = "adamw_torch"

    if resume_from_checkpoint:
        state_path = Path(resume_from_checkpoint) / "trainer_state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"Resume checkpoint missing trainer_state.json: {state_path}")
        with open(state_path) as f:
            trainer_state = json.load(f)
        global_step = trainer_state.get("global_step", 0)
        training_config["max_steps"] = global_step
        training_config["load_best_model_at_end"] = False  # keep resumed checkpoint, don't switch to "best"
        print(f"Resume from {resume_from_checkpoint} (global_step={global_step}); will load then save adapter (no new steps).")

    training_args = TrainingArguments(
        output_dir=str(run_output_dir),
        **training_config
    )
    
    print(f"Output directory: {run_output_dir}")
    print(f"Training epochs: {training_config['num_train_epochs']}")
    print(f"Batch size: {training_config['per_device_train_batch_size']}")
    print(f"Gradient accumulation: {training_config['gradient_accumulation_steps']}")
    print(f"Effective batch size: {training_config['per_device_train_batch_size'] * training_config['gradient_accumulation_steps']}")
    print(f"Learning rate: {training_config['learning_rate']}")
    
    # Collator: when max_length is 0 we do not truncate; pad to longest in batch only.
    if max_length is None or max_length <= 0:
        collator_cap = 2**20
    else:
        collator_cap = getattr(tokenizer, "model_max_length", 8192)
    data_collator_obj = make_data_collator(tokenizer, max_length_cap=collator_cap)

    compute_metrics_fn = _make_compute_metrics(tokenizer)
    callbacks = [EvalSummaryCallback()]
    if (
        training_config.get("eval_strategy") != "no"
        and not resume_from_checkpoint
        and EarlyStoppingCallback is not None
    ):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0))

    trainer = TrainerWithBalancedBatches(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator_obj,
        compute_metrics=compute_metrics_fn,
        train_batch_sampler=train_batch_sampler,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Plot training curve (loss / eval loss / lr) for stability inspection
    try:
        from plot_training_curve import plot_training_curve
        plot_training_curve(
            trainer.state.log_history,
            run_output_dir / "training_curve.png",
            title=f"Training curve - {DATASET}" + (f" (r={lora_rank})" if lora_rank else "") + (f" max_len={max_length}" if max_length and max_length != 256 else ""),
        )
    except Exception as e:
        print(f"Could not plot training curve: {e}")

    print("\n" + "="*70)
    print("SAVING FINAL MODEL")
    print("="*70)

    run_final_model_dir.mkdir(parents=True, exist_ok=True)
    # With FSDP, Trainer only saves properly after gathering full state (so PEFT adapter is saved)
    if getattr(trainer, "is_fsdp_enabled", False) and getattr(
        trainer.accelerator.state, "fsdp_plugin", None
    ):
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
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
        default=0,
        metavar="N",
        help="Max sequence length (default: 0 = no truncation). Use 512 or 1024 only for memory-limited runs.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        metavar="F",
        help="Use only this fraction of training data (default: 1.0). Use e.g. 0.1 to debug pipeline quickly.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=1.0,
        metavar="F",
        help="Use only this fraction of validation data (default: 1.0). Use e.g. 0.1 to debug pipeline quickly.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume from this checkpoint (e.g. .../checkpoint-1008), then save PEFT adapter to final_model. No new training steps.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug (print full sample I/O for a few train/val examples). Also enabled by DEBUG_TRAINING=true.",
    )
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        lora_rank=args.lora_rank,
        use_fsdp=args.fsdp,
        max_length=args.max_length,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        resume_from_checkpoint=args.resume_from_checkpoint,
        debug=args.debug,
    )
