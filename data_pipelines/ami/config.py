"""
Configuration for AMI pipeline: paths, API keys, and hyperparameters.

API keys are loaded from environment variables. Set them before running:
    export GEMINI_API_KEY="your-gemini-key"
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

# Raw AMI corpus (XML files). Place the ami_public_manual_1.6.2 folder here
# or update this path to point to your local AMI corpus.
AMI_CORPUS_DIR = BASE_DIR / 'ami_corpus'

# Pipeline intermediates (stage JSON dumps)
JSON_DUMPS_DIR = BASE_DIR / 'json_dumps'

# Final filtered output (stage4_filtered_samples.jsonl + filtering_summary.json)
OUTPUT_DIR = BASE_DIR / 'data_final'

# Per-sample JSON files written by stage 4
JSON_RUN_DIR = BASE_DIR / 'json_run'

# ============================================================================
# API KEYS (from environment variables)
# ============================================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# ============================================================================
# STAGE 1: EXTRACT DIALOGUES
# ============================================================================

# Maximum number of previous turns to include as context when extracting sequences
STAGE1_MAX_CONTEXT_TURNS = 10

# Maximum number of meetings to process (None = process all)
STAGE1_MAX_MEETINGS = None

# ============================================================================
# STAGE 1B: INFER ADDRESSEES
# ============================================================================

# Gemini model name to use for inference
STAGE1B_MODEL_NAME = 'gemini-flash-latest'

# Confidence threshold (0-10 scale). Only use predictions with confidence >= this value.
# Lower = more data but lower quality. Higher = less data but higher quality.
# Recommended: 5-7 for balanced results
STAGE1B_CONFIDENCE_THRESHOLD = 6

# Number of previous turns to include in Gemini prompt
STAGE1B_MAX_CONTEXT_TURNS = 10

# Number of turns to process before API call delay (for rate limiting)
STAGE1B_BATCH_SIZE = 10

# Seconds to wait between batches (rate limiting)
STAGE1B_API_DELAY = 1.0

# Stride sampling: take every Nth high-quality dialogue act per meeting
# 1 = use all, 2 = use every other, etc.
STAGE1B_STRIDE = 5

# Optional cap on number of high-quality dialogue acts per meeting passed to Gemini
# None = no limit
STAGE1B_MAX_SAMPLES_PER_MEETING = None

# ============================================================================
# STAGE 2: GENERATE DECISION POINTS
# ============================================================================

# Currently no configurable hyperparameters in Stage 2
# (All context is preserved from sequences)

# ============================================================================
# STAGE 4: FILTER AND BALANCE HIGH-QUALITY SAMPLES
# ============================================================================

# Target ratio for SPEAK/SILENT
STAGE4_SPEAK_RATIO = 0.5  # 50% SPEAK, 50% SILENT

# Minimum text length (characters) after cleaning
STAGE4_MIN_TEXT_LENGTH = 3

# Maximum samples to keep (None = keep all that pass filters)
STAGE4_MAX_SAMPLES = None

# Subcategory balance threshold (max ratio any category can be)
STAGE4_MAX_CATEGORY_RATIO = 0.35

# Enable/disable deduplication step
STAGE4_ENABLE_DEDUPLICATION = True
# N-gram dedup: treat samples as duplicate when (current + context) n-gram similarity >= threshold
STAGE4_DEDUP_NGRAM_N = 3
STAGE4_DEDUP_SIMILARITY_THRESHOLD = 0.6  # Jaccard >= this → keep only one (disjoint set)
STAGE4_DEDUP_SAME_MEETING_ONLY = True
STAGE4_DEDUP_CONTEXT_WINDOW = 5  # last 5 context turns + current turn only

# Remove samples with no context turns
STAGE4_REMOVE_NO_CONTEXT = False

# Remove samples where current turn is very short (query-like)
STAGE4_REMOVE_SHORT_QUERY = True

# Threshold for "short query" in words (used when STAGE4_REMOVE_SHORT_QUERY is True)
STAGE4_MIN_QUERY_WORDS = 3

# Additional Gemini-based confidence filtering for AI-inferred samples.
# If None, disabled. If set to 0-10, removes samples with
#   source == 'gemini_inferred' and inference_confidence < this threshold.
STAGE4_GEMINI_CONFIDENCE_FILTER = None
