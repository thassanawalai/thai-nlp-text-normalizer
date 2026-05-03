"""
ThaiTextNormalizer — normalizes informal/evasion Thai social media text.

Pipeline (order is critical):
  1. _remove_separators  : 'ข า ย' -> 'ขาย'         (spaces/symbols between Thai chars)
  2. _apply_char_sub     : 'รับvายของ' -> 'รับขายของ' (Latin chars -> visual Thai equivalents)
  3. _apply_word_mapping : 'กี' -> 'หี', 'ขาe' -> handled after step 2
  4. _remove_elongation  : 'ค้าบบบบ' -> 'ค้าบ'       (collapse repeated chars)

Compatible with pandas .apply():
    df['clean'] = df['text'].apply(normalizer.normalize)
"""

import re
from normalizer.char_substitution import apply_char_substitution
from normalizer.data_loader import build_combined_word_dict


class ThaiTextNormalizer:
    """Normalize filter-evasion Thai text for ML preprocessing pipelines."""

    def __init__(self, word_dict: dict = None):
        """
        Args:
            word_dict: Optional custom mapping dict. If None, loads automatically
                       from data/slang_dict.csv + data/evasion_patterns.py.
        """
        self._word_dict = word_dict if word_dict is not None else build_combined_word_dict()

        # Pre-sort keys longest-first to prevent shorter keys from matching
        # inside longer tokens (e.g., 'ก' matching before 'กู้เงิน').
        self._sorted_keys = sorted(self._word_dict.keys(), key=len, reverse=True)

        # Matches: a sequence of "isolated" Thai chars (each NOT adjacent to another Thai
        # char directly) connected by single non-Thai separators.
        # e.g. 'ข า ย' matches (each char is isolated), but 'ขาย รถ' does NOT match
        # because 'ขาย' and 'รถ' are multi-char Thai words.
        # Replacement: strip all non-Thai chars from the matched span.
        self._sep_pattern = re.compile(
            r'(?<![฀-๿])'                   # not preceded by Thai (left word boundary)
            r'[฀-๿](?![฀-๿])'              # first isolated Thai char (not followed by Thai)
            r'(?:[^฀-๿][฀-๿](?![฀-๿]))*'  # (separator + isolated Thai)*
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, text: str) -> str:
        """
        Normalize a single string. Main entry point for pandas .apply().

        Example:
            normalizer = ThaiTextNormalizer()
            df['normalized'] = df['raw_text'].apply(normalizer.normalize)
        """
        if not isinstance(text, str) or not text.strip():
            return text

        text = self._remove_separators(text)
        text = self._apply_char_sub(text)
        text = self._apply_word_mapping(text)
        text = self._remove_elongation(text)
        return text

    # ------------------------------------------------------------------
    # Pipeline steps (private)
    # ------------------------------------------------------------------

    def _remove_separators(self, text: str) -> str:
        """
        Feature 3: Remove single separator chars inserted between isolated Thai
        characters to evade keyword filters.

        'ข า ย' -> 'ขาย'   (each char is isolated, match + strip separators)
        'ข-า-ย' -> 'ขาย'
        'ข า ย รถ' -> 'ขาย รถ'  (space before รถ preserved: รถ is not isolated)

        A single pass of re.sub is sufficient because the pattern matches the
        entire span (e.g. 'ข า ย') at once and replaces with 'ขาย'.
        """
        def _strip_seps(m: re.Match) -> str:
            return re.sub(r'[^฀-๿]', '', m.group(0))

        return self._sep_pattern.sub(_strip_seps, text)

    def _apply_char_sub(self, text: str) -> str:
        """Feature 1: Replace visually similar Latin chars with Thai equivalents."""
        return apply_char_substitution(text)

    def _apply_word_mapping(self, text: str) -> str:
        """
        Feature 2: Replace evasion tokens and slang words with formal forms.
        Longest keys are tried first to prevent partial substitutions.
        """
        for key in self._sorted_keys:
            if key in text:
                text = text.replace(key, self._word_dict[key])
        return text

    def _remove_elongation(self, text: str) -> str:
        """
        Collapse 3+ consecutive identical characters down to 1.
        'สวัสดีค้าบบบบ' -> 'สวัสดีค้าบ'
        """
        return re.sub(r'(.)\1{2,}', r'\1', text)
