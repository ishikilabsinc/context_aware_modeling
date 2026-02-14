# Evaluation Directory

This directory contains scripts for comprehensive evaluation of the fine-tuned model.

## Purpose

- Load fine-tuned LoRA adapters
- Evaluate fine-tuned model performance
- Compare against baseline results
- Generate category-specific analysis
- Test latency optimization
- Generate final reports

## Evaluation Metrics

- Overall accuracy improvement vs baseline
- Per-category performance (I1-I3, S1-S5)
- Precision, recall, F1-score by category
- Latency comparison
- Error analysis and examples

## Files

- `load_finetuned.py` - Load fine-tuned model and adapters
- `evaluate_finetuned.py` - Run comprehensive evaluation
- `category_analysis.py` - Detailed category-wise analysis
- `latency_test.py` - Latency benchmarking
- `generate_report.py` - Generate final evaluation report

## Results

Results are stored in `results/`
