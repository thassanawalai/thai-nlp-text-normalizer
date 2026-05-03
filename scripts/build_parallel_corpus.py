"""
Build a large Thai informal→formal parallel corpus for fine-tuning.

Sources:
  1. wisesight_sentiment  — 26K Thai social media texts
  2. wongnai_corpus       — Thai restaurant reviews (informal)
  3. thassanawalai/thai-slang-parallel-corpus — existing pairs (if available)

Strategy:
  - Load informal Thai texts from each source
  - Use Gemini API to produce formal Thai versions (LLM as teacher)
  - Deduplicate and save to data/thai_parallel_corpus.csv
  - Supports resume: skips texts already processed

Run from project root:
    python scripts/build_parallel_corpus.py --samples 5000
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm
from google import genai
from datasets import load_dataset

from config import GEMINI_API_KEY, BASE_DIR

OUTPUT_PATH = Path(BASE_DIR) / "data" / "thai_parallel_corpus.csv"
CHECKPOINT_PATH = Path(BASE_DIR) / "data" / ".corpus_checkpoint.json"

# -------------------------------------------------------------------------
# Gemini client
# -------------------------------------------------------------------------
def make_client(api_key: str = None):
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError("Set GEMINI_API_KEY env var or pass --api-key.")
    return genai.Client(api_key=key)


# -------------------------------------------------------------------------
# LLM normalizer (single text)
# -------------------------------------------------------------------------
_SYSTEM = (
    "คุณเป็นผู้เชี่ยวชาญด้านภาษาไทย แปลงข้อความภาษาไทยที่ไม่เป็นทางการ "
    "(มีคำสแลง คำย่อ คำวัยรุ่น หรือตัวอักษรซ้ำ) ให้เป็นภาษาไทยมาตรฐานที่ถูกต้อง "
    "รักษาความหมายเดิม ตอบเฉพาะข้อความที่แปลงแล้ว ห้ามอธิบาย"
)

def normalize_with_llm(client, text: str, model: str = "gemini-2.5-flash-lite") -> str | None:
    """Return formal Thai version of text, or None on failure."""
    prompt = f"{_SYSTEM}\n\nInput: {text}\nOutput:"
    for attempt in range(5):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            result = resp.text.strip()
            return result if result else None
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 60 * (attempt + 1)
                print(f"\nRate limit. Waiting {wait}s (attempt {attempt+1}/5)...")
                time.sleep(wait)
            else:
                print(f"\nAPI error: {err}")
                return None
    return None


# -------------------------------------------------------------------------
# Checkpoint helpers (resume support)
# -------------------------------------------------------------------------
import json

def load_checkpoint() -> set:
    if CHECKPOINT_PATH.exists():
        with CHECKPOINT_PATH.open(encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_checkpoint(done_hashes: set):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_PATH.open("w", encoding="utf-8") as f:
        json.dump(list(done_hashes), f)

def text_hash(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]


# -------------------------------------------------------------------------
# Dataset loaders
# -------------------------------------------------------------------------
def load_wisesight(max_samples: int) -> list[str]:
    print("Loading wisesight_sentiment...")
    ds = load_dataset("wisesight_sentiment", split="train+validation+test")
    texts = [r["texts"].strip() for r in ds if len(r["texts"].strip()) >= 10]
    return texts[:max_samples]


def load_wongnai(max_samples: int) -> list[str]:
    print("Loading wongnai_corpus...")
    try:
        ds = load_dataset("wongnai_corpus", split="train")
        texts = [r["review_body"].strip() for r in ds if len(r.get("review_body","").strip()) >= 10]
        return texts[:max_samples]
    except Exception as e:
        print(f"  wongnai_corpus not available: {e}")
        return []


def load_existing_parallel() -> list[dict]:
    """Load any already-existing parallel pairs from local files."""
    rows = []
    candidates = [
        Path(BASE_DIR) / "real_social_slang_dataset.csv",
        Path(BASE_DIR) / "data" / "thai_parallel_corpus.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if "noisy_text" in df.columns and "formal_text" in df.columns:
                rows.extend(df[["noisy_text", "formal_text"]].dropna().to_dict("records"))
                print(f"  Loaded {len(df)} existing pairs from {path.name}")
    return rows


# -------------------------------------------------------------------------
# Main build routine
# -------------------------------------------------------------------------
def build_corpus(
    total_samples: int = 5000,
    request_delay: float = 2.5,
    api_key: str = None,
    model: str = "gemini-2.5-flash-lite",
):
    client = make_client(api_key)
    done_hashes = load_checkpoint()

    # Seed with existing pairs
    existing = load_existing_parallel()
    pairs: list[dict] = list({p["noisy_text"]: p for p in existing}.values())
    print(f"Starting with {len(pairs)} existing pairs.")

    # Collect source texts
    per_source = max(100, (total_samples - len(pairs)) // 2)
    sources: list[str] = []
    sources.extend(load_wisesight(per_source * 2))
    sources.extend(load_wongnai(per_source))

    # Deduplicate sources
    seen = {text_hash(p["noisy_text"]) for p in pairs}
    unique_sources = []
    for t in sources:
        h = text_hash(t)
        if h not in seen and h not in done_hashes:
            seen.add(h)
            unique_sources.append(t)

    need = total_samples - len(pairs)
    unique_sources = unique_sources[:need * 2]  # buffer for failures
    print(f"Processing {len(unique_sources)} new texts to reach {total_samples} pairs...")

    # Generate pairs
    for text in tqdm(unique_sources, desc="Generating corpus"):
        if len(pairs) >= total_samples:
            break
        h = text_hash(text)
        formal = normalize_with_llm(client, text, model=model)
        if formal and formal != text:
            pairs.append({"noisy_text": text, "formal_text": formal})
        done_hashes.add(h)
        save_checkpoint(done_hashes)
        time.sleep(request_delay)

    # Save
    df = pd.DataFrame(pairs).drop_duplicates(subset=["noisy_text"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} pairs to '{OUTPUT_PATH}'.")
    return df


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Thai parallel corpus")
    parser.add_argument("--samples", type=int, default=5000, help="Target number of pairs")
    parser.add_argument("--delay", type=float, default=2.5, help="Seconds between API calls")
    parser.add_argument("--api-key", type=str, default=None, help="Gemini API key override")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-lite")
    args = parser.parse_args()

    df = build_corpus(
        total_samples=args.samples,
        request_delay=args.delay,
        api_key=args.api_key,
        model=args.model,
    )
    print(f"Done. {len(df)} total pairs.")
