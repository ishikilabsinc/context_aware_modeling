# Fine-Tuning Directory

This directory contains scripts for LoRA fine-tuning of QWEN 8B for context-aware turn-taking.

## Purpose

- Configure LoRA hyperparameters
- Load and preprocess training data
- Train QWEN 8B with LoRA adapters
- Monitor training progress
- Export fine-tuned models

## Training Format

Input follows the format from `ami/stage5_format_training.py`:
- `<|system|>{System Prompt}<|/system|>`
- `<|instruction|>{Role instruction}<|/instruction|>`
- `<|context|>{Previous conversation}<|/context|>`
- `<|current|>{Current utterance}<|/current|>`

Output format:
- `<decision>SPEAK or SILENT</decision>`
- `<confidence>high|medium|low</confidence>`
- `<reason>{Brief one-line reasoning}</reason>`

## Files

- `config.py` - Training and LoRA configuration
- `data_loader.py` - Data loading and preprocessing
- `train_lora.py` - Main training script
- `monitor_training.py` - Training monitoring utilities
- `export_model.py` - Export fine-tuned model for deployment

## Checkpoints

Model checkpoints are saved in `checkpoints/` (gitignored)
