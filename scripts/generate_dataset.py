"""
Generate a synthetic Thai slang/noisy parallel dataset.
Run from project root: python scripts/generate_dataset.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import random
from datasets import load_dataset
from config import BASE_DIR


def load_slang_dictionary() -> dict:
    dataset = load_dataset("thassanawalai/thai-social-slang-dict", split="train")
    return dict(zip(dataset['formal'], dataset['slang']))


def load_base_sentences(filepath: str) -> list:
    if not os.path.exists(filepath):
        print(f"Error: '{filepath}' not found.")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def generate_synthetic_data(base_sentences: list, num_samples: int = 1000) -> pd.DataFrame:
    slang_dict = load_slang_dictionary()
    formal_words = list(slang_dict.keys())

    synthetic_pairs = []
    max_attempts = num_samples * 10
    attempts = 0

    while len(synthetic_pairs) < num_samples and attempts < max_attempts:
        attempts += 1
        sentence = random.choice(base_sentences)
        noisy = sentence

        for word in formal_words:
            if word in noisy and random.random() > 0.3:
                noisy = noisy.replace(word, slang_dict[word])

        if random.random() > 0.5 and len(noisy) > 5:
            idx = random.randint(0, len(noisy) - 1)
            ch = noisy[idx]
            if '฀' <= ch <= '๿':
                noisy = noisy[:idx] + ch * random.randint(2, 4) + noisy[idx + 1:]

        if noisy != sentence:
            synthetic_pairs.append({"noisy_text": noisy, "formal_text": sentence})
            attempts = 0

    return pd.DataFrame(synthetic_pairs)


if __name__ == "__main__":
    input_file = os.path.join(BASE_DIR, "formal_sentences.txt")
    output_file = os.path.join(BASE_DIR, "slang_dataset.csv")
    target_samples = 5000

    print(f"Loading base sentences from '{input_file}'...")
    sentences = load_base_sentences(input_file)

    if not sentences:
        print("Process terminated: missing input file.")
    else:
        print(f"Loaded {len(sentences)} sentences. Generating {target_samples} samples...")
        df = generate_synthetic_data(sentences, num_samples=target_samples)
        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"Saved {len(df)} samples to '{output_file}'.")
