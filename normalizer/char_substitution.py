import re

# Maps Latin characters that are visually similar to Thai characters.
# Used for cross-lingual filter evasion (e.g., 'ขาe' where 'e' looks like 'ย').
CHAR_MAP = {
    'e': 'ย',  # 'e' visually resembles Thai ย
    'v': 'ข',  # 'v' visually resembles Thai ข
    'w': 'พ',  # 'w' visually resembles Thai พ
    'a': 'า',  # 'a' visually resembles Thai สระ า
    'u': 'น',  # 'u' visually resembles Thai น
    'o': 'อ',  # 'o' visually resembles Thai อ
}

# Unicode range for Thai script (U+0E00 to U+0E7F)
_THAI_RE = re.compile(r'[฀-๿]')


def apply_char_substitution(text: str) -> str:
    """
    Replace visually similar Latin chars with Thai equivalents,
    but only when the Latin char is adjacent to at least one Thai character.

    This prevents mangling purely English words in mixed-language text.
    Example: 'รับvายของ' -> 'รับขายของ'  (v is between Thai chars)
             'vietnam' stays 'vietnam'     (no adjacent Thai chars)
    """
    result = []
    for i, ch in enumerate(text):
        if ch in CHAR_MAP:
            prev_is_thai = i > 0 and bool(_THAI_RE.match(text[i - 1]))
            next_is_thai = i < len(text) - 1 and bool(_THAI_RE.match(text[i + 1]))
            if prev_is_thai or next_is_thai:
                result.append(CHAR_MAP[ch])
            else:
                result.append(ch)
        else:
            result.append(ch)
    return ''.join(result)
