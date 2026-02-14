# Benchmarking Directory

This directory contains scripts and tools for evaluating the baseline QWEN 8B model performance on the turn-taking task before fine-tuning.

## Purpose

- Download and validate training data from S3
- Set up and load the base QWEN 8B model
- Evaluate baseline performance metrics
- Generate baseline results and analysis

## Key Metrics

- Overall accuracy (SPEAK vs SILENT classification)
- Per-category accuracy (I1-I3, S1-S5)
- Confusion matrix
- Latency statistics (mean, p50, p95, p99)
- False positive/negative rates

## Files

- `download_data.py` - Download prepared samples from S3
- `validate_data.py` - Validate data format and generate statistics
- `setup_model.py` - Load and configure QWEN 8B base model
- `evaluate_baseline.py` - Run baseline evaluation
- `analyze_baseline.py` - Generate detailed analysis reports

## Results

Results are stored in `results/baseline_results.json`
