"""
ThaiContentModerator — Thai filter-evasion detection and normalization.

Detects evasive Thai text (e.g., 'แตn' for 'แตก'), normalizes it,
and returns a structured moderation report with category and severity info.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvasionEntry:
    normalized: str
    category: str   # 'Profanity' | 'Gambling' | 'Loan/Finance' | 'NSFW/Illegal' | 'E-commerce' | 'Drugs'
    severity: str   # 'Low' | 'Medium' | 'High'


# ---------------------------------------------------------------------------
# Evasion dictionary
# Each key is the evasive/obfuscated form; value is the EvasionEntry.
# Longer keys are matched first (prevents partial-match shadowing).
# ---------------------------------------------------------------------------

EVASION_DICT: dict[str, EvasionEntry] = {

    # ── Profanity ────────────────────────────────────────────────────────────
    "แตn":      EvasionEntry("แตก",     "Profanity",    "Medium"),
    "แตเk":     EvasionEntry("แตก",     "Profanity",    "Medium"),
    "สัตv":     EvasionEntry("สัตว์",   "Profanity",    "Medium"),
    "หuำ":      EvasionEntry("หนำ",     "Profanity",    "Low"),
    "ควaย":     EvasionEntry("ควาย",    "Profanity",    "Low"),
    "ไอ้สัตv":  EvasionEntry("ไอ้สัตว์","Profanity",    "High"),

    # ── Gambling ─────────────────────────────────────────────────────────────
    "บอ ล":     EvasionEntry("บอล",     "Gambling",     "High"),
    "บo ล":     EvasionEntry("บอล",     "Gambling",     "High"),
    "พนัu":     EvasionEntry("พนัน",    "Gambling",     "High"),
    "พ นั น":   EvasionEntry("พนัน",    "Gambling",     "High"),
    "หวy":      EvasionEntry("หวย",     "Gambling",     "High"),
    "ห ว ย":    EvasionEntry("หวย",     "Gambling",     "High"),
    "คาสิโu":   EvasionEntry("คาสิโน", "Gambling",     "High"),
    "แทงบo ล":  EvasionEntry("แทงบอล", "Gambling",     "High"),

    # ── Loan/Finance ─────────────────────────────────────────────────────────
    "กู้เงิu":  EvasionEntry("กู้เงิน", "Loan/Finance", "Medium"),
    "กู้เงิ น":  EvasionEntry("กู้เงิน","Loan/Finance", "Medium"),
    "ดอกเบี้ y": EvasionEntry("ดอกเบี้ย","Loan/Finance","Medium"),
    "เงิuด่วu":  EvasionEntry("เงินด่วน","Loan/Finance","High"),
    "สินเชื่o":  EvasionEntry("สินเชื่อ","Loan/Finance","Medium"),
    "ปล่อยกู้":  EvasionEntry("ปล่อยกู้","Loan/Finance","High"),

    # ── NSFW / Illegal ───────────────────────────────────────────────────────
    "โป๊":      EvasionEntry("โป๊",     "NSFW/Illegal", "High"),
    "หนังxxx":  EvasionEntry("หนังโป๊", "NSFW/Illegal", "High"),
    "น้องใหม่": EvasionEntry("น้องใหม่","NSFW/Illegal", "High"),   # escort euphemism
    "บริกaร":   EvasionEntry("บริการ",  "NSFW/Illegal", "High"),
    "เปิดให้":  EvasionEntry("เปิดให้", "NSFW/Illegal", "Medium"),
    "ลับเฉพaะ": EvasionEntry("ลับเฉพาะ","NSFW/Illegal","Medium"),

    # ── E-commerce (counterfeit / scam) ──────────────────────────────────────
    "vายของ":   EvasionEntry("ขายของ",  "E-commerce",   "Low"),
    "ขaยของ":   EvasionEntry("ขายของ",  "E-commerce",   "Low"),
    "ของปลoม":  EvasionEntry("ของปลอม", "E-commerce",   "High"),
    "แบรuด์เก๊": EvasionEntry("แบรนด์เก๊","E-commerce", "High"),
    "ส่งฟรีทั้วไทy": EvasionEntry("ส่งฟรีทั่วไทย","E-commerce","Low"),
    "กดลิ้uค์":  EvasionEntry("กดลิ้งค์","E-commerce",  "Medium"),

    # ── Drugs ────────────────────────────────────────────────────────────────
    "ยาบ้a":    EvasionEntry("ยาบ้า",   "Drugs",        "High"),
    "ยา บ้ า":  EvasionEntry("ยาบ้า",   "Drugs",        "High"),
    "ไอซ์":     EvasionEntry("ไอซ์",    "Drugs",        "High"),
    "ไ อ ซ์":   EvasionEntry("ไอซ์",    "Drugs",        "High"),
    "กัญชa":    EvasionEntry("กัญชา",   "Drugs",        "Medium"),
    "โคเคu":    EvasionEntry("โคเคน",   "Drugs",        "High"),
}

_SEVERITY_RANK = {"None": 0, "Low": 1, "Medium": 2, "High": 3}


# ---------------------------------------------------------------------------
# Thai char-substitution helpers (same logic as normalizer/char_substitution.py)
# ---------------------------------------------------------------------------

_CHAR_MAP: dict[str, str] = {
    'e': 'ย', 'v': 'ข', 'w': 'พ',
    'a': 'า', 'u': 'น', 'o': 'อ',
    'n': 'ก', 'y': 'ย', 'k': 'ก',
}
_THAI_RE = re.compile(r'[฀-๿]')


def _char_sub(text: str) -> str:
    """Replace ASCII chars with Thai equivalents when adjacent to Thai script."""
    result = []
    for i, ch in enumerate(text):
        if ch in _CHAR_MAP:
            prev_thai = i > 0 and bool(_THAI_RE.match(text[i - 1]))
            next_thai = i < len(text) - 1 and bool(_THAI_RE.match(text[i + 1]))
            result.append(_CHAR_MAP[ch] if (prev_thai or next_thai) else ch)
        else:
            result.append(ch)
    return "".join(result)


def _remove_separators(text: str) -> str:
    """Collapse spaces/symbols inserted between Thai characters (e.g., 'ข า ย' → 'ขาย')."""
    pattern = re.compile(
        r'(?<![฀-๿])'
        r'[฀-๿](?![฀-๿])'
        r'(?:[^฀-๿][฀-๿](?![฀-๿]))*'
    )

    def _strip(m: re.Match) -> str:
        return re.sub(r'[^฀-๿]', '', m.group(0))

    return pattern.sub(_strip, text)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ThaiContentModerator:
    """
    Detects filter-evasion Thai text, normalizes it, and returns a
    structured moderation report.

    Usage:
        moderator = ThaiContentModerator()
        result = moderator.moderate_text("อยากซื้อยาบ้aมั้ย")
    """

    def __init__(self, extra_entries: Optional[dict[str, EvasionEntry]] = None):
        combined = dict(EVASION_DICT)
        if extra_entries:
            combined.update(extra_entries)
        # Sort longest-first to prevent shorter keys from shadowing longer ones
        self._dict: dict[str, EvasionEntry] = dict(
            sorted(combined.items(), key=lambda kv: len(kv[0]), reverse=True)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def moderate_text(self, raw: str) -> dict:
        """
        Normalize and moderate a raw Thai string.

        Returns:
            {
                'original_text'     : str,
                'normalized_text'   : str,
                'detected_categories': list[str],
                'max_severity'      : 'None' | 'Low' | 'Medium' | 'High',
                'is_safe'           : bool,
                'matches'           : list[dict],   # detailed hit log
            }
        """
        # Step 1: separator removal only — expose multi-char evasion patterns
        #         (char sub runs AFTER dict lookup so keys like 'ยาบ้a', 'ไอ้สัตv' still match)
        preprocessed = _remove_separators(raw)

        # Step 2: dictionary scan (longest-key first, before char sub)
        normalized = preprocessed
        matches: list[dict] = []
        seen_keys: set[str] = set()

        for evasive, entry in self._dict.items():
            if evasive in normalized and evasive not in seen_keys:
                normalized = normalized.replace(evasive, entry.normalized)
                seen_keys.add(evasive)
                matches.append({
                    "evasive_form":  evasive,
                    "normalized":    entry.normalized,
                    "category":      entry.category,
                    "severity":      entry.severity,
                })

        # Step 3: char substitution on whatever wasn't caught by the dict
        normalized = _char_sub(normalized)

        # Step 4: aggregate results
        categories = list(dict.fromkeys(m["category"] for m in matches))  # unique, ordered
        max_sev = max(
            (m["severity"] for m in matches),
            key=lambda s: _SEVERITY_RANK[s],
            default="None",
        )
        is_safe = _SEVERITY_RANK[max_sev] < _SEVERITY_RANK["Medium"]

        return {
            "original_text":      raw,
            "normalized_text":    normalized,
            "detected_categories": categories,
            "max_severity":       max_sev,
            "is_safe":            is_safe,
            "matches":            matches,
        }

    def add_entry(self, evasive: str, normalized: str, category: str, severity: str) -> None:
        """Register a new evasion pattern at runtime."""
        self._dict[evasive] = EvasionEntry(normalized, category, severity)
        # Re-sort
        self._dict = dict(
            sorted(self._dict.items(), key=lambda kv: len(kv[0]), reverse=True)
        )


# ---------------------------------------------------------------------------
# Demo / test script
# ---------------------------------------------------------------------------

def _print_result(result: dict) -> None:
    sep = "-" * 60
    print(sep)
    print(f"  Original  : {result['original_text']}")
    print(f"  Normalized: {result['normalized_text']}")
    print(f"  Categories: {result['detected_categories'] or ['(none)']}")
    print(f"  Severity  : {result['max_severity']}")
    safe_label = "Yes" if result['is_safe'] else "No (flagged for review)"
    print(f"  Safe?     : {safe_label}")
    if result["matches"]:
        print("  Matches   :")
        for m in result["matches"]:
            print(f"    [{m['severity']:6}] {m['evasive_form']!r:20} -> {m['normalized']!r}  ({m['category']})")
    print(sep)


if __name__ == "__main__":
    moderator = ThaiContentModerator()

    test_sentences = [
        # 1. Gambling + Profanity mix
        "แทงบo ล ได้เงินเยอะ ไอ้สัตv",

        # 2. Drug offer with char-substitution evasion
        "มีของดีนะ ยาบ้aราคาถูก โคเคuส่งถึงบ้าน",

        # 3. Loan scam + counterfeit goods
        "กู้เงิuด่วน ดอกเบี้ y ต่ำ แถมได้ของปลoมฟรี กดลิ้uค์เลย",
    ]

    print("\nThai Content Moderation Report")
    print("=" * 60)

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\nTest {i}:")
        result = moderator.moderate_text(sentence)
        _print_result(result)
