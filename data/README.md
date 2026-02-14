# Data Directory

This directory stores raw and processed training data.

## Structure

- `raw_data/` - Raw JSON files from AMI dataset (gitignored)
- `train/` - Training set samples (processed, gitignored)
- `val/` - Validation set samples (processed, gitignored)
- `test/` - Test set samples (processed, gitignored)

## Source

Raw data is loaded from: `data/raw_data/` (local JSON files)

Alternative source (if USE_S3=True in download_data.py): `s3://ishiki-ml-datasets/AMI_samples/json_run/`

## Usage

Run `benchmarking/download_data.py` to:
1. Load raw JSON files from `raw_data/`
2. Split into train/val/test sets
3. Save processed splits to `train/`, `val/`, `test/`

## Note

This directory is gitignored. Data files should not be committed to the repository.
