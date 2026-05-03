"""
Generate a real-world Thai parallel corpus using Gemini LLM normalization.
Run from project root: python scripts/generate_real_social_dataset.py
Requires: GEMINI_API_KEY environment variable.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
from datasets import load_dataset
from google import genai
from tqdm import tqdm

from config import GEMINI_API_KEY, BASE_DIR

client = genai.Client(api_key=GEMINI_API_KEY)


def normalize_text_with_llm(social_text: str) -> str:
    prompt = (
        "You are a linguistic expert in the Thai language. "
        "Convert the following informal Thai social media text into a formal, "
        "grammatically correct Thai sentence. Preserve the core meaning. "
        "Output ONLY the formal Thai sentence.\n\n"
        f"Informal Text: {social_text}\nFormal Text:"
    )

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                print(f"\nRate limit hit. Waiting 60s (attempt {attempt + 1}/5)...")
                time.sleep(60.0)
            else:
                print(f"\nAPI error: {err}")
                return None
    return None


def generate_parallel_corpus(num_samples: int = 1000) -> pd.DataFrame:
    print("Loading 'wisesight_sentiment' from Hugging Face...")
    try:
        dataset = load_dataset("wisesight_sentiment", split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return pd.DataFrame()

    dataset = dataset.shuffle(seed=42).select(range(num_samples))
    rows = []

    for row in tqdm(dataset, desc="Normalizing"):
        text = row['texts'].strip()
        if len(text) < 5:
            continue
        formal = normalize_text_with_llm(text)
        if formal:
            rows.append({"noisy_text": text, "formal_text": formal})
        time.sleep(4.0)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_parallel_corpus(num_samples=2000)

    if not df.empty:
        out = os.path.join(BASE_DIR, "real_social_slang_dataset.csv")
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"\nGenerated {len(df)} pairs. Saved to '{out}'.")
    else:
        print("\nNo data generated.")
