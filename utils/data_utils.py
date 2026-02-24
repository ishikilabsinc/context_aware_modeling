#!/usr/bin/env python3
"""Shared data loading and filtering utilities."""

import json
from pathlib import Path
from typing import List, Dict


def has_context_turns(sample: Dict) -> bool:
    ctx = sample.get("context_turns", [])
    return bool(ctx)


def filter_samples_with_context(samples: List[Dict]) -> List[Dict]:
    return [s for s in samples if has_context_turns(s)]


def load_samples(file_path: Path) -> List[Dict]:
    samples = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num} of {file_path}: {e}")
        
        return samples
    
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []
