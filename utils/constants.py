#!/usr/bin/env python3
"""
Shared Constants

Category definitions and other constants used across the codebase.
"""

# Category definitions
SPEAK_CATEGORIES = ['I1', 'I2', 'I3']
SILENT_CATEGORIES = ['S1', 'S2', 'S3', 'S4', 'S5']
ALL_CATEGORIES = SPEAK_CATEGORIES + SILENT_CATEGORIES

CATEGORY_NAMES = {
    'I1': 'SPEAK - Direct address',
    'I2': 'SPEAK - Context follow-up',
    'I3': 'SPEAK - Implicit redirect',
    'S1': 'SILENT - No reference',
    'S2': 'SILENT - Mentioned but not addressed',
    'S3': 'SILENT - Context shifted away',
    'S4': 'SILENT - Incomplete sentence',
    'S5': 'SILENT - Explicit context switch'
}
