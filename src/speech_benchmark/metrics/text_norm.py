"""Language-aware text normalization applied before every text metric.

Kept deliberately simple and fully documented (docs/methodology.md):
  * Unicode NFC, lowercase, punctuation/symbols removed, whitespace collapsed.
  * Arabic: diacritics (tashkeel) and tatweel removed; alef variants unified;
    this is the standard light normalization for Arabic WER.
  * Chinese: scoring is character-based (space-insensitive) — see tokens().
The same normalizer is applied to reference and hypothesis, so all systems
are compared on equal footing.
"""

from __future__ import annotations

import re
import unicodedata

_ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")
_TATWEEL = "ـ"


def normalize_text(text: str, language: str = "en") -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    if language == "ar":
        text = _ARABIC_DIACRITICS.sub("", text)
        text = text.replace(_TATWEEL, "")
        text = re.sub("[آأإ]", "ا", text)  # alef variants -> ا
        text = text.replace("ة", "ه")  # ta marbuta -> ha
        text = text.replace("ى", "ي")  # alef maqsura -> ya
    # Strip punctuation & symbols (unicode categories P*, S*)
    text = "".join(
        " " if unicodedata.category(ch)[0] in ("P", "S") else ch for ch in text
    )
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str, language: str = "en") -> list[str]:
    """Scoring units: characters for Chinese, whitespace words otherwise."""
    norm = normalize_text(text, language)
    if language == "zh":
        return [ch for ch in norm if not ch.isspace()]
    return norm.split()
