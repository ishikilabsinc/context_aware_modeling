#!/usr/bin/env python3
"""
Data Utilities

Shared functions for loading and processing data files.
"""

import json
from pathlib import Path
from typing import List, Dict


def load_samples(file_path: Path) -> List[Dict]:
    """
    Load samples from JSONL file
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of sample dictionaries
    """
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
