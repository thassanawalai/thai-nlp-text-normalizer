import csv
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SLANG_DICT_PATH
from data.evasion_patterns import EVASION_DICT


def load_slang_csv(filepath: str = SLANG_DICT_PATH) -> dict:
    """Load slang->formal pairs from a CSV file with 'slang' and 'formal' columns."""
    result = {}
    path = Path(filepath)
    if not path.exists():
        print(f"Warning: slang dictionary not found at {path}")
        return result
    with path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slang = row.get('slang', '').strip()
            formal = row.get('formal', '').strip()
            if slang and formal:
                result[slang] = formal
    return result


def build_combined_word_dict() -> dict:
    """
    Merge slang_dict.csv and evasion_patterns.py into one lookup dict.
    EVASION_DICT takes precedence on key collision (more specific patterns).
    """
    combined = load_slang_csv()
    combined.update(EVASION_DICT)
    return combined
