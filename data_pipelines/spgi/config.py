"""
Configuration for SPGI pipeline: paths and hyperparameters (heuristic only, no API).
"""

from pathlib import Path

# ============================================================================
# SPGI DATA
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
# Raw SPGI word-level segment data. Update this path to point to your local SPGI dataset.
SPGI_DATASET_DIR = BASE_DIR / 'spgi_dataset'
# Pipeline intermediates and output
JSON_DUMPS_DIR = BASE_DIR / 'json_dumps'
OUTPUT_DIR = BASE_DIR / 'data_final'
JSON_RUN_DIR = BASE_DIR / 'json_run'

# ============================================================================
# STAGE 1: EXTRACT TURNS
# ============================================================================

# Max context turns when building sequences (Stage 1b)
STAGE1_MAX_CONTEXT_TURNS = 10

# Optional cap on number of calls to process (None = all)
STAGE1_MAX_CALLS = None

# Gap in seconds to optionally split turns (None = do not split by gap)
STAGE1_TURN_GAP_SECONDS = None

# ============================================================================
# STAGE 1B: HEURISTIC SEQUENCES (legacy — kept for backward compat)
# ============================================================================

STAGE1B_MAX_CONTINUATION_TURNS = 10
STAGE1B_HEURISTIC_CONFIDENCE = 5

# ============================================================================
# STAGE 2: DECISION POINT GENERATION (flat turn-stream)
# ============================================================================

# Min speakers in a call to generate decision points (2-party calls are trivial)
STAGE2_MIN_SPEAKERS_PER_CALL = 3

# Min context turns before a turn to create a decision point
STAGE2_MIN_CONTEXT_TURNS = 3

# How many recent turns to look back when detecting "recently active" speakers
STAGE2_RECENT_SPEAKER_WINDOW = 3

# ============================================================================
# STAGE 4: FILTER AND BALANCE HIGH-QUALITY SAMPLES
# ============================================================================

STAGE4_SPEAK_RATIO = 0.5
STAGE4_MIN_TEXT_LENGTH = 7

# ── SILENT quality filters ──
# Only keep SILENT samples where the target had some reason to speak.
# A SILENT sample is "high quality" if the target:
#   - was the addressee (previous speaker), OR
#   - spoke at least once in the last SILENT_RECENCY_WINDOW turns, OR
#   - spoke at least SILENT_MIN_SPEAKER_TURNS times total in the call context
# Setting STAGE4_REQUIRE_ENGAGED_SILENT = True drops trivially-silent targets
# (e.g., participants who haven't spoken in many turns and have no contextual cue).
STAGE4_REQUIRE_ENGAGED_SILENT = True
STAGE4_SILENT_RECENCY_WINDOW = 5     # target must have spoken in last N context turns…
STAGE4_SILENT_MIN_SPEAKER_TURNS = 2  # …OR have spoken at least this many times in full context
STAGE4_MAX_SAMPLES = None
STAGE4_MAX_SAMPLES_PER_MEETING = 2000  # cap samples per meeting_id; None = no cap
STAGE4_MAX_CATEGORY_RATIO = 0.35
STAGE4_ENABLE_DEDUPLICATION = True
# N-gram dedup: treat samples as duplicate when (current + context) n-gram similarity >= threshold
STAGE4_DEDUP_NGRAM_N = 3
STAGE4_DEDUP_SIMILARITY_THRESHOLD = 0.2  # Jaccard >= this → keep only one (disjoint set)
# Only compare within same meeting; use last N context turns + current for similarity (not all sentences)
STAGE4_DEDUP_SAME_MEETING_ONLY = True
STAGE4_DEDUP_CONTEXT_WINDOW = 5  # last 5 context turns + current turn
STAGE4_REMOVE_NO_CONTEXT = True
STAGE4_REMOVE_SHORT_QUERY = True
STAGE4_MIN_QUERY_WORDS = 7
STAGE4_GEMINI_CONFIDENCE_FILTER = None
# Exclude very long samples (reduces memory; set to None to disable)
STAGE4_MAX_CURRENT_TURN_CHARS = 5000    # drop if current turn text length > this
STAGE4_MAX_CONTEXT_TURNS = 100          # drop if sample has more than this many context turns
STAGE4_MAX_LINE_BYTES = 100_000         # skip raw input lines larger than this (avoid parsing huge JSON)
# Process in chunks to avoid loading full JSONL into memory (None = load all)
STAGE4_CHUNK_SIZE = 100_000  # lines per chunk; set to None for in-memory (original) behavior
