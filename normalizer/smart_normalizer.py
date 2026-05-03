"""
SmartNormalizer — Unified Thai text normalization pipeline.

Architecture (single pass, no user-facing engine choice):
  1. ThaiTextNormalizer  : fast rule-based pre-processing (char-sub, evasion dict, regex)
  2. Gemini LLM          : context-aware normalization with few-shot prompting
  3. SHA256 JSON cache   : avoid re-calling API for identical texts

Usage:
    from normalizer.smart_normalizer import SmartNormalizer
    n = SmartNormalizer(api_key="...")            # or reads GEMINI_API_KEY env var
    n.normalize("ตะเองทำรายอยู่คับ วันนี้อากาศดีย์จุงเบยยย")
    # -> "ตัวเองทำอะไรอยู่ครับ วันนี้อากาศดีจังเลย"

    # Pandas compatible
    df["clean"] = df["text"].apply(n.normalize)
"""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from normalizer.thai_text_normalizer import ThaiTextNormalizer
from config import GEMINI_API_KEY

# --------------------------------------------------------------------------
# Few-shot examples embedded in every prompt.
# Cover: elongation, slang particles, word alteration, evasion patterns.
# --------------------------------------------------------------------------
_FEW_SHOT_EXAMPLES = [
    ("ตะเองทำรายอยู่คับ", "ตัวเองทำอะไรอยู่ครับ"),
    ("วันนี้อากาศดีย์จุงเบยยย", "วันนี้อากาศดีจังเลย"),
    ("อ้วนน หิวข้าวจางงงงเบยยย", "อ้วน หิวข้าวจังเลย"),
    ("ชิมิๆ ทำได้มุ้ยเตง", "ใช่ไหม ทำได้ไหมตัวเอง"),
    ("ไปกินข้าวกัbuงับ", "ไปกินข้าวกันครับ"),
    ("ขายของออนไลน์ดีย์มากเบย ลูกค้าเยอะมากก", "ขายของออนไลน์ดีมากเลย ลูกค้าเยอะมาก"),
    ("สล็อตแตกง่ายๆๆๆ รับเงินเร็ว", "สล็อตแตกง่าย รับเงินเร็ว"),
    ("งับ โอเคเลยค้าบ ขอบคุณมากๆๆ", "ครับ โอเคเลยครับ ขอบคุณมาก"),
]

_SYSTEM_PROMPT = (
    "คุณเป็นผู้เชี่ยวชาญด้านภาษาไทย ทำหน้าที่แปลงข้อความภาษาไทยที่ไม่เป็นทางการ "
    "(มีคำสแลง คำย่อ คำวัยรุ่น ตัวอักษรซ้ำ หรือการสะกดเพื่อเลี่ยงการกรองของโซเชียลมีเดีย) "
    "ให้เป็นภาษาไทยมาตรฐานที่ถูกต้องตามหลักภาษา\n\n"
    "กฎ:\n"
    "- รักษาความหมายและน้ำเสียงของประโยคต้นฉบับไว้\n"
    "- แปลงคำสแลง คำย่อ และตัวสะกดผิดให้ถูกต้อง\n"
    "- ตัดอักษรที่ซ้ำเกินความจำเป็นออก (เช่น 'ดีมากกก' → 'ดีมาก')\n"
    "- ตอบเฉพาะข้อความที่แปลงแล้ว ห้ามอธิบายหรือใส่เครื่องหมายคำพูด"
)


class SmartNormalizer:
    """
    Unified Thai text normalization pipeline combining rule-based pre-processing
    and Gemini LLM for context-aware normalization.

    Falls back to rule-based only if no API key is provided.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-2.5-flash",
        cache_path: str = None,
        request_delay: float = 1.0,
    ):
        """
        Args:
            api_key     : Gemini API key. Reads GEMINI_API_KEY env var if None.
            model       : Gemini model name.
            cache_path  : Path to JSON cache file. Defaults to data/normalizer_cache.json.
            request_delay: Seconds between successive API calls (rate-limit safety).
        """
        self._rules = ThaiTextNormalizer()
        self._model = model
        self._delay = request_delay
        self._client = None

        # Resolve API key
        _key = api_key or GEMINI_API_KEY
        if _key:
            try:
                from google import genai
                self._client = genai.Client(api_key=_key)
            except Exception as e:
                print(f"[SmartNormalizer] Gemini init failed: {e}. Using rule-based only.")

        # Persistent cache (SHA256 -> normalized text)
        if cache_path is None:
            cache_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "normalizer_cache.json",
            )
        self._cache_path = Path(cache_path)
        self._cache: dict = self._load_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, text: str) -> str:
        """
        Normalize a single Thai text string.
        Compatible with pandas .apply():
            df["clean"] = df["text"].apply(normalizer.normalize)
        """
        if not isinstance(text, str) or not text.strip():
            return text

        # Phase 1 — rule-based pre-processing
        preprocessed = self._rules.normalize(text)

        # Phase 2 — LLM normalization (if available)
        if self._client:
            return self._llm_normalize(preprocessed, original=text)

        return preprocessed

    def normalize_batch(self, texts: list, progress: bool = True) -> list:
        """Normalize a list of texts with optional tqdm progress bar."""
        try:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="Normalizing") if progress else texts
        except ImportError:
            iterator = texts

        results = []
        for t in iterator:
            results.append(self.normalize(t))
            if self._client:
                time.sleep(self._delay)
        return results

    @property
    def has_llm(self) -> bool:
        """True if Gemini API is available."""
        return self._client is not None

    def cache_size(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # Private — LLM
    # ------------------------------------------------------------------

    def _llm_normalize(self, preprocessed: str, original: str = None) -> str:
        """Call Gemini with few-shot prompting; serve from cache when possible."""
        key = hashlib.sha256(preprocessed.encode("utf-8")).hexdigest()
        if key in self._cache:
            return self._cache[key]

        prompt = self._build_prompt(preprocessed)
        for attempt in range(4):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                )
                result = response.text.strip()
                # Reject empty or clearly wrong responses
                if result and len(result) > 0:
                    self._cache[key] = result
                    self._save_cache()
                    return result
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    wait = 60 * (attempt + 1)
                    print(f"[SmartNormalizer] Rate limit. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[SmartNormalizer] API error: {err}")
                    break

        # Fallback: return pre-processed text
        return preprocessed

    def _build_prompt(self, text: str) -> str:
        """Construct few-shot prompt for Thai text normalization."""
        shots = "\n".join(
            f"Input: {inp}\nOutput: {out}" for inp, out in _FEW_SHOT_EXAMPLES
        )
        return (
            f"{_SYSTEM_PROMPT}\n\n"
            f"ตัวอย่าง:\n{shots}\n\n"
            f"Input: {text}\nOutput:"
        )

    # ------------------------------------------------------------------
    # Private — Cache
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict:
        if self._cache_path.exists():
            try:
                with self._cache_path.open(encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_cache(self):
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)
