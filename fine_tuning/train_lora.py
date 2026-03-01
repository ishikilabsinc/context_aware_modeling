#!/usr/bin/env python3
"""LoRA fine-tuning for context-aware turn-taking. Saves adapter to checkpoints/<model>_r<N>/final_model/. Use --resume-from-checkpoint to resume or save adapter from a checkpoint."""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

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

# Expected decision for 4-way categories (for sampler sanity check)
_CATEGORY_TO_EXPECTED_DECISION = {
    "SPEAK_explicit": "SPEAK",
    "SPEAK_implicit": "SPEAK",
    "SILENT_ref": "SILENT",
    "SILENT_no_ref": "SILENT",
}


def _run_sampler_sanity_check(train_batch_sampler, train_samples, output_path: Path, rank=None):
    """Check that indices from the batch sampler point to samples with matching category/decision. Write results to output_path."""
    if train_batch_sampler is None or train_samples is None:
        return
    try:
        first_batch = next(iter(train_batch_sampler))
    except Exception as e:
        with open(output_path, "w") as f:
            f.write(f"Failed to get first batch from sampler: {e}\n")
        return
    n_show = min(10, len(first_batch))
    lines = [
        "Sampler sanity check: first batch indices vs actual sample category and decision.",
        "If 'OK' is False, the sampler index does not match the sample (label mismatch bug).",
        "",
        f"Batch size: {len(first_batch)}  Showing first {n_show}.",
        "",
        "index | category        | decision | expected (from category) | OK",
        "-" * 70,
    ]
    n_ok = 0
    for i in range(n_show):
        idx = first_batch[i]
        if idx < 0 or idx >= len(train_samples):
            lines.append(f"  {idx}  | OUT OF RANGE (len={len(train_samples)}) | FAIL")
            continue
        s = train_samples[idx]
        category = s.get("category") or "(missing)"
        decision = s.get("decision") or "(missing)"
        expected = _CATEGORY_TO_EXPECTED_DECISION.get(category)
        if expected is None:
            ok = decision in ("SPEAK", "SILENT")  # can't verify category
            status = "OK (category not in 4-way)" if ok else "FAIL (bad decision)"
        else:
            ok = decision == expected
            status = "OK" if ok else "FAIL"
        if ok:
            n_ok += 1
        lines.append(f"  {idx:<5} | {str(category):<15} | {str(decision):<8} | {str(expected):<24} | {status}")
    lines.extend(["", f"Summary: {n_ok}/{n_show} passed.", ""])
    out_text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(out_text)
    print(f"  Sampler sanity check written to {output_path} ({n_ok}/{n_show} passed)")


def setup_model_and_tokenizer(lora_rank=None, use_fsdp=False, local_rank=-1):
    from config import LORA_CONFIG, BASE_MODEL
    r = lora_rank if lora_rank is not None else LORA_CONFIG["r"]
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



def _make_compute_metrics(tokenizer, eval_predictions_path=None, model_key=None):
    import numpy as np
    use_logit_pos_minus_one = model_key == "gpt-oss-20b"
    speak_ids = tokenizer.encode("SPEAK", add_special_tokens=False)
    silent_ids = tokenizer.encode("SILENT", add_special_tokens=False)
    speak_id = int(speak_ids[1]) if len(speak_ids) >= 2 else (int(speak_ids[-1]) if speak_ids else -1)
    silent_id = int(silent_ids[1]) if len(silent_ids) >= 2 else (int(silent_ids[-1]) if silent_ids else -1)
    _eval_run_counter = [0]  # mutable so we can increment in closure

    def compute_metrics(eval_pred):
        if hasattr(eval_pred, "predictions"):
            predictions, labels = eval_pred.predictions, eval_pred.label_ids
        else:
            predictions, labels = eval_pred[0], eval_pred[1]
        labels = np.asarray(labels)
        pred_arr = np.asarray(predictions)
        n = labels.shape[0]
        if pred_arr.ndim == 1 and labels.ndim == 1 and len(labels) == len(pred_arr):
            y_pred = pred_arr.ravel()
            y_true = labels.ravel().astype(np.int64)
        elif pred_arr.ndim == 1:
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
                # gpt-oss-20b: logits at t predict token at t+1; use pred at pos-1 for label at pos
                pred_pos = (pos - 1) if (pos >= 1 and use_logit_pos_minus_one) else pos
                y_pred.append(1 if pred_tokens[i, pred_pos] == speak_id else 0)
            y_true = np.array(y_true) if y_true else np.array([])
            y_pred = np.array(y_pred) if y_pred else np.array([])
        if len(y_true) == 0:
            return {"accuracy": 0.0, "macro_f1": 0.0, "balanced_accuracy": 0.0}
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Save predictions to file (rank 0 only) for inspection
        if eval_predictions_path and os.environ.get("LOCAL_RANK", "0") == "0":
            try:
                _eval_run_counter[0] += 1
                run_num = _eval_run_counter[0]
                path = Path(eval_predictions_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a") as f:
                    f.write(f"\n=== Eval run {run_num} (n={len(y_true)} samples) ===\n")
                    f.write("idx   true   pred   true_label  pred_label  correct\n")
                    for i in range(len(y_true)):
                        t, p = int(y_true[i]), int(y_pred[i])
                        tl = "SPEAK" if t == 1 else "SILENT"
                        pl = "SPEAK" if p == 1 else "SILENT"
                        ok = "yes" if t == p else "no"
                        f.write(f"{i:<5} {t}      {p}      {tl:<11} {pl:<11} {ok}\n")
                    f.write("\n")
            except Exception as e:
                import traceback
                print(f"  [eval_predictions] Failed to write {eval_predictions_path}: {e}\n{traceback.format_exc()}")

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
    def __init__(self):
        self.last_train_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_train_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        if os.environ.get("LOCAL_RANK", "0") != "0":
            return
        step = state.global_step
        train_loss = self.last_train_loss if self.last_train_loss is not None else "-"
        eval_loss = metrics.get("eval_loss", "-")
        eval_acc = metrics.get("eval_accuracy", metrics.get("accuracy", "-"))
        eval_macro_f1 = metrics.get("eval_macro_f1", metrics.get("macro_f1", "-"))
        eval_bal_acc = metrics.get("eval_balanced_accuracy", metrics.get("balanced_accuracy", "-"))
        print(f"  [check] Step {step} | train_loss={train_loss} | eval_loss={eval_loss} | eval_acc={eval_acc} | macro_f1={eval_macro_f1} | balanced_acc={eval_bal_acc}")


class LogTargetVsPredictedCallback(TrainerCallback):
    def __init__(self, n_samples=4, model_key=None):
        self.n_samples = n_samples
        self._model_key = model_key

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        trainer = getattr(self, "trainer", None)
        if trainer is None or metrics is None:
            return
        tokenizer = getattr(trainer, "_tokenizer", None)
        speak_id = getattr(trainer, "_speak_id", None)
        silent_id = getattr(trainer, "_silent_id", None)
        if tokenizer is None or speak_id is None or silent_id is None:
            return

        local_rank = os.environ.get("LOCAL_RANK", "0")
        is_rank0 = local_rank == "0"
        out_path = Path(trainer.args.output_dir) / "target_vs_predicted.txt"
        err_path = Path(trainer.args.output_dir) / "target_vs_predicted_err.txt"
        step = state.global_step

        try:
            device = getattr(getattr(trainer, "accelerator", None), "device", None)
            if device is None:
                try:
                    device = next(trainer.model.parameters()).device
                except (StopIteration, Exception):
                    device = torch.device("cuda", int(local_rank)) if torch.cuda.is_available() else torch.device("cpu")
        except Exception:
            device = torch.device("cuda", int(local_rank)) if local_rank.isdigit() and torch.cuda.is_available() else torch.device("cpu")

        try:
            lines = [f"\n{'='*60}", f"Step {step}", f"{'='*60}"]
            for split_name, dataset in [("train", trainer.train_dataset), ("val", trainer.eval_dataset)]:
                if dataset is None or len(dataset) == 0:
                    continue
                n = min(self.n_samples, len(dataset))
                lines.append(f"\n--- {split_name.upper()} (first {n} samples) ---")
                items = [dataset[i] for i in range(n)]
                batch = trainer.data_collator(items)
                batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
                with torch.no_grad():
                    outputs = trainer.model(**batch)
                logits = outputs.logits
                labels = batch["labels"]
                B, L, V = logits.shape
                for i in range(B):
                    pos = None
                    for j in range(L - 1, -1, -1):
                        if labels[i, j].item() in (speak_id, silent_id):
                            pos = j
                            break
                    if pos is None:
                        target_str = "?"
                        pred_str = "?"
                    else:
                        target_str = "SPEAK" if labels[i, pos].item() == speak_id else "SILENT"
                        if speak_id < V and silent_id < V:
                            # gpt-oss-20b only: logits at t predict token at t+1
                            logit_pos = (pos - 1) if (pos >= 1 and getattr(self, "_model_key", None) == "gpt-oss-20b") else pos
                            pred_str = "SPEAK" if logits[i, logit_pos, speak_id].item() >= logits[i, logit_pos, silent_id].item() else "SILENT"
                        else:
                            pred_str = "?"
                    ok = "ok" if target_str == pred_str else "wrong"
                    lines.append(f"  sample {i}: target={target_str}  predicted={pred_str}  {ok}")
            lines.append("")

            if is_rank0:
                write_header = not out_path.exists()
                with open(out_path, "a", encoding="utf-8") as f:
                    if write_header:
                        f.write("Target vs predicted for first few train/val samples (appended after each eval).\n")
                    f.write("\n".join(lines))
                if step == args.eval_steps:
                    print(f"  [target_vs_predicted] Appended step {step} to {out_path}")
        except Exception as e:
            import traceback
            err_msg = f"Step {step}: {e}\n{traceback.format_exc()}"
            if is_rank0:
                try:
                    with open(err_path, "a", encoding="utf-8") as f:
                        f.write(err_msg)
                    if not out_path.exists():
                        with open(out_path, "w", encoding="utf-8") as f:
                            f.write(f"Error (see {err_path.name}): {e}\n")
                except Exception:
                    print(f"  [target_vs_predicted] Error: {e}")
                else:
                    print(f"  [target_vs_predicted] Error (see {err_path}): {e}")


class TrainerWithBalancedBatches(Trainer):
    def __init__(self, train_batch_sampler=None, tokenizer=None, model_key=None, **kwargs):
        super().__init__(**kwargs)
        self.train_batch_sampler = train_batch_sampler
        self._tokenizer = tokenizer
        self._use_logit_pos_minus_one = model_key == "gpt-oss-20b"
        self._speak_id = None
        self._silent_id = None
        if tokenizer is not None:
            speak_ids = tokenizer.encode("SPEAK", add_special_tokens=False)
            silent_ids = tokenizer.encode("SILENT", add_special_tokens=False)
            self._speak_id = int(speak_ids[1]) if len(speak_ids) >= 2 else (int(speak_ids[-1]) if speak_ids else -1)
            self._silent_id = int(silent_ids[1]) if len(silent_ids) >= 2 else (int(silent_ids[-1]) if silent_ids else -1)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if self._tokenizer is None or self._speak_id is None or self._silent_id is None:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
        labels = inputs.get("labels")
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss if outputs.loss is not None else None
            logits = outputs.logits
        if prediction_loss_only or logits is None or labels is None:
            return (loss, None, None)
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
            if speak_id >= 0 and silent_id >= 0 and speak_id < V and silent_id < V:
                # gpt-oss-20b only: logits at t predict token at t+1; use logits at pos-1 for label at pos
                logit_pos = (pos - 1) if (pos >= 1 and self._use_logit_pos_minus_one) else pos
                pred_classes.append(1 if logits[i, logit_pos, speak_id].item() >= logits[i, logit_pos, silent_id].item() else 0)
            else:
                pred_classes.append(0)
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
    training_mode: str = "decision_only",
    equal_sampling: bool = False,
    filter_no_context: bool = True,
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
    from data_loader import prepare_datasets, make_data_collator, TRAINING_MODE_DECISION_ONLY, TRAINING_MODE_COT, TRAINING_MODES

    if training_mode not in TRAINING_MODES:
        raise ValueError(f"training_mode must be one of {TRAINING_MODES}, got {training_mode!r}")

    base_run_name = f"{MODEL}_r{lora_rank}" if lora_rank is not None else MODEL
    if training_mode == TRAINING_MODE_COT:
        run_name = f"{base_run_name}_cot"
    else:
        run_name = base_run_name

    run_output_dir = Path(__file__).resolve().parent / "checkpoints" / run_name
    run_final_model_dir = run_output_dir / "final_model"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print(f"LORA FINE-TUNING FOR CONTEXT-AWARE TURN-TAKING")
    print(f"Dataset: {DATASET}")
    print(f"Training mode: {training_mode}")
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
    if filter_no_context:
        print("Filter: excluding samples with no context_turns")
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
        if equal_sampling:
            print("Equal sampling: enabled (SPGI subsampled to 11K stratified)")
    else:
        print(f"Train file: {TRAIN_FILE}")
        print(f"Val file: {VAL_FILE}")

    batch_size = TRAINING_CONFIG["per_device_train_batch_size"]
    if max_length is None or max_length <= 0:
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
    train_dataset, val_dataset, train_batch_sampler, train_samples = prepare_datasets(
        tokenizer,
        max_length=max_length,
        debug=debug,
        batch_size=batch_size,
        use_balanced_batches=True,
        four_way_balanced=TRAINING_CONFIG.get("four_way_balanced", True),
        rank=rank,
        world_size=world_size,
        silent_ratio_in_batch=TRAINING_CONFIG.get("silent_ratio_in_batch", 0.5),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        debug_sample_io_path=str(run_output_dir / "debug_sample_io.txt") if debug else None,
        training_mode=training_mode,
        equal_sampling=equal_sampling,
        filter_no_context=filter_no_context,
    )

    print("\n" + "="*70)
    print("SETTING UP TRAINING")
    print("="*70)

    training_config = {k: v for k, v in TRAINING_CONFIG.items() if k not in ("output_dir", "silent_ratio_in_batch", "four_way_balanced")}
    if use_fsdp:
        training_config["fsdp"] = ["full_shard"]
        training_config["gradient_checkpointing"] = False
        if max_length is None or max_length <= 0:
            training_config["per_device_train_batch_size"] = 1
            training_config["gradient_accumulation_steps"] = max(training_config["gradient_accumulation_steps"], 40)
    elif local_rank >= 0:
        training_config["gradient_checkpointing"] = False
        if max_length is None or max_length <= 0:
            training_config["per_device_train_batch_size"] = 1
            training_config["gradient_accumulation_steps"] = max(training_config["gradient_accumulation_steps"], 40)
    if training_config.get("optim") == "adamw_torch_fused" and not torch.cuda.is_available():
        training_config["optim"] = "adamw_torch"

    if resume_from_checkpoint:
        resume_path = Path(resume_from_checkpoint)
        if not resume_path.is_absolute():
            resume_path = run_output_dir / resume_path
        state_path = resume_path / "trainer_state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"Resume checkpoint missing trainer_state.json: {state_path}")
        resume_from_checkpoint = str(resume_path)
        with open(state_path) as f:
            trainer_state = json.load(f)
        global_step = trainer_state.get("global_step", 0)
        training_config["max_steps"] = global_step
        training_config["load_best_model_at_end"] = False
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
    
    if max_length is None or max_length <= 0:
        collator_cap = 2**20
    else:
        collator_cap = getattr(tokenizer, "model_max_length", 8192)
    data_collator_obj = make_data_collator(tokenizer, max_length_cap=collator_cap)

    compute_metrics_fn = _make_compute_metrics(
        tokenizer,
        eval_predictions_path=run_output_dir / "eval_predictions.txt",
        model_key=MODEL,
    )
    callbacks = [EvalSummaryCallback(), LogTargetVsPredictedCallback(n_samples=4, model_key=MODEL)]
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
        model_key=MODEL,
        callbacks=callbacks,
    )
    if hasattr(trainer.model, "gradient_checkpointing_disable"):
        trainer.model.gradient_checkpointing_disable()

    # Sanity check: verify sampler indices match sample category/decision (saved to file for inspection)
    if rank is None or rank == 0:
        _run_sampler_sanity_check(
            train_batch_sampler,
            train_samples,
            run_output_dir / "sampler_sanity_check.txt",
            rank=rank,
        )

    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

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
    # With FSDP, ensure full state dict so final save is loadable. Mid checkpoints are full only if
    # accelerate_fsdp_config.yaml uses fsdp_state_dict_type: FULL_STATE_DICT (recommended for eval).
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
    parser.add_argument(
        "--training-mode",
        type=str,
        default="decision_only",
        choices=["decision_only", "cot"],
        help="Training mode: decision_only (response = <decision>SPEAK|SILENT</decision> only) or cot (chain-of-thought: <reasoning> + <decision> + <confidence>; uses train_samples_with_reasoning.jsonl). Checkpoints are saved to mode-specific dirs (e.g. ..._cot) so modes do not overwrite each other.",
    )
    parser.add_argument(
        "--equal-sampling",
        action="store_true",
        help="When dataset=all: subsample SPGI to 11K (stratified 50/50 SPEAK/SILENT, category-proportional). AMI and Friends kept as-is. Total ~32.5K, ~1 epoch = ~1,015 steps.",
    )
    parser.add_argument(
        "--filter-no-context",
        action="store_true",
        default=True,
        help="Exclude samples with no context_turns from train and val (default: True).",
    )
    parser.add_argument(
        "--no-filter-no-context",
        action="store_false",
        dest="filter_no_context",
        help="Do not filter; include samples with no context_turns.",
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
        training_mode=args.training_mode,
        equal_sampling=args.equal_sampling,
        filter_no_context=args.filter_no_context,
    )
