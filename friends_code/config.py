"""
Configuration for Friends-MMC pipeline: paths and hyperparameters.
"""

from pathlib import Path

# ============================================================================
# FRIENDS-MMC DATA
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
FRIENDS_MMC_DIR = BASE_DIR / 'friends_mmc'
JSON_DUMPS_DIR = BASE_DIR / 'json_dumps'
OUTPUT_DIR = BASE_DIR / 'data_final'
JSON_RUN_DIR = BASE_DIR / 'json_run'

# Which metadata files to use: (num_turns, split) -> filename suffix
# e.g. 5_turns/train-metadata.json
FRIENDS_NUM_TURNS = [5, 8]
FRIENDS_SPLITS = ['train', 'test']

# Optional cap for quick runs (None = no cap)
STAGE1_MAX_SEQUENCES_PER_SPLIT = None

# ============================================================================
# STAGE 4: FILTER AND BALANCE HIGH-QUALITY SAMPLES
# ============================================================================

STAGE4_SPEAK_RATIO = 0.5
STAGE4_MIN_TEXT_LENGTH = 3
STAGE4_MAX_SAMPLES = None
STAGE4_MAX_CATEGORY_RATIO = 0.35
STAGE4_ENABLE_DEDUPLICATION = True
STAGE4_DEDUP_NGRAM_N = 3
STAGE4_DEDUP_SIMILARITY_THRESHOLD = 0.7  # Jaccard >= this → keep only one (disjoint set)
STAGE4_DEDUP_SAME_MEETING_ONLY = True
STAGE4_DEDUP_CONTEXT_WINDOW = 5  # last 5 context turns + current turn only
STAGE4_REMOVE_NO_CONTEXT = False
STAGE4_REMOVE_SHORT_QUERY = False
STAGE4_MIN_QUERY_WORDS = 3
STAGE4_GEMINI_CONFIDENCE_FILTER = None
