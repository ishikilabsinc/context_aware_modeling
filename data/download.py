#!/usr/bin/env python3
"""
Download the multi-party dialogue dataset from Hugging Face.

Dataset: https://huggingface.co/datasets/ishiki-labs/multi-party-dialogue

Usage:
    python data/download.py
    python data/download.py --output_dir data/
    python data/download.py --dataset ami --output_dir data/ami/
"""

import argparse
import os
from pathlib import Path


def download_dataset(dataset_name: str, output_dir: Path, hf_token: str = None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: `huggingface_hub` package not installed.")
        print("Install with: pip install huggingface_hub")
        raise SystemExit(1)

    HF_REPO = "ishiki-labs/multi-party-dialogue"
    AVAILABLE_DATASETS = ["ami", "friends", "spgi"]

    domains_to_download = AVAILABLE_DATASETS if dataset_name == "all" else [dataset_name]

    # Build glob patterns so only the requested domain folders are downloaded.
    # Files are mirrored as-is from the repo with no processing.
    allow_patterns = [f"{d}/**" for d in domains_to_download]

    print(f"\n{'='*60}")
    print(f"Downloading: {', '.join(d.upper() for d in domains_to_download)}")
    print(f"From: {HF_REPO}")
    print(f"To:   {output_dir.resolve()}")
    print(f"{'='*60}\n")

    try:
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            token=hf_token,
            local_dir=str(output_dir),
            allow_patterns=allow_patterns,
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        print(f"Make sure you have access to {HF_REPO}")
        print("If the dataset is private, set HF_TOKEN env var or pass --hf_token.")
        return

    print(f"\n{'='*60}")
    print("Download complete.")
    print(f"Data saved to: {output_dir.resolve()}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Download the multi-party dialogue dataset from Hugging Face."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "ami", "friends", "spgi"],
        help="Which dataset split to download (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to save the downloaded data (default: data/)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    download_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        hf_token=hf_token,
    )


if __name__ == "__main__":
    main()
