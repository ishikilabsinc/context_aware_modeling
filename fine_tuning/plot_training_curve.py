#!/usr/bin/env python3
"""
Plot training loss (and eval loss / learning rate) from Trainer log_history.
Use after training to inspect the training curve for instability.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Optional matplotlib; fail gracefully if not installed
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_training_curve(
    log_history: List[Dict],
    save_path: Path,
    title: str = "Training curve",
) -> None:
    """
    Plot train loss, eval loss, and learning rate from Hugging Face Trainer log_history.
    save_path: where to save the figure (e.g. training_curve.png).
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    lr_steps, lr_values = [], []

    for entry in log_history:
        if "loss" in entry and "epoch" in entry:
            step = entry.get("step", len(steps))
            steps.append(step)
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", 0))
            eval_losses.append(entry["eval_loss"])
        if "learning_rate" in entry:
            lr_steps.append(entry.get("step", 0))
            lr_values.append(entry["learning_rate"])

    if not steps and not eval_steps:
        print("No loss data found in log_history.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Step")

    if steps and train_losses:
        ax1.plot(steps, train_losses, "b-", alpha=0.8, label="Train loss")
    if eval_steps and eval_losses:
        ax1.plot(eval_steps, eval_losses, "g-", alpha=0.8, label="Eval loss")

    ax1.set_ylabel("Loss", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    if lr_steps and lr_values:
        ax2 = ax1.twinx()
        ax2.plot(lr_steps, lr_values, "gray", alpha=0.6, linestyle="--", label="Learning rate")
        ax2.set_ylabel("Learning rate", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 0.85))

    plt.title(title)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curve saved to {save_path}")


def main(model: Optional[str] = None, output: Optional[str] = None) -> None:
    import os
    model_key = (model or os.environ.get("MODEL", "qwen2.5-7b")).strip().lower()
    fine_tuning_dir = Path(__file__).resolve().parent
    checkpoints_dir = fine_tuning_dir / "checkpoints" / model_key
    state_file = checkpoints_dir / "trainer_state.json"

    if not state_file.exists():
        print(f"trainer_state.json not found at {state_file}. Run training first.")
        return

    with open(state_file, "r") as f:
        state = json.load(f)
    log_history = state.get("log_history", [])

    save_path = Path(output) if output else checkpoints_dir / "training_curve.png"
    plot_training_curve(log_history, save_path, title=f"Training curve - {model_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training loss curve from trainer_state.json")
    parser.add_argument("--model", type=str, default=None, help="Model key (default: from MODEL env)")
    parser.add_argument("--output", type=str, default=None, help="Output path for figure (default: checkpoints/<model>/training_curve.png)")
    args = parser.parse_args()
    main(model=args.model, output=args.output)
