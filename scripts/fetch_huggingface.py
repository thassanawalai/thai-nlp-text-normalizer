"""
Download the Wisesight Sentiment dataset from Hugging Face and save as CSV.
Run from project root: python scripts/fetch_huggingface.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datasets import load_dataset
from config import BASE_DIR

print("Connecting to Hugging Face...")
dataset = load_dataset("wisesight_sentiment")
df = pd.DataFrame(dataset['train'])

print(f"\nLoaded {len(df)} rows. First 10 samples:")
print("-" * 50)
for i, text in enumerate(df['texts'].head(10)):
    print(f"[{i+1}] {text}")
print("-" * 50)

out = os.path.join(BASE_DIR, "wisesight_raw.csv")
df[['texts']].to_csv(out, index=False, encoding='utf-8-sig')
print(f"Saved to '{out}'.")
