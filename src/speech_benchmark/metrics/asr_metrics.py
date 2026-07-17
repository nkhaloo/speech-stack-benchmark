"""ASR quality metrics: WER / CER with language-aware tokenization.

For Chinese the primary metric is character error rate (WER over characters);
for the other languages WER over normalized words, with CER as a secondary
diagnostic. jiwer provides the alignment; token error *counts* are exposed so
cpWER can reuse them.
"""

from __future__ import annotations

from dataclasses import dataclass

import jiwer

from .text_norm import normalize_text, tokens


@dataclass
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    hits: int

    @property
    def ref_len(self) -> int:
        return self.substitutions + self.deletions + self.hits

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float | None:
        return self.errors / self.ref_len if self.ref_len else None


def token_error_counts(ref_tokens: list[str], hyp_tokens: list[str]) -> ErrorCounts:
    if not ref_tokens:
        return ErrorCounts(0, 0, len(hyp_tokens), 0)
    if not hyp_tokens:
        return ErrorCounts(0, len(ref_tokens), 0, 0)
    out = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))
    return ErrorCounts(out.substitutions, out.deletions, out.insertions, out.hits)


def word_error_rate(ref: str, hyp: str, language: str = "en") -> float | None:
    """WER over language-appropriate tokens (characters for zh)."""
    return token_error_counts(tokens(ref, language), tokens(hyp, language)).rate


def char_error_rate(ref: str, hyp: str, language: str = "en") -> float | None:
    ref_n = normalize_text(ref, language).replace(" ", "" if language == "zh" else " ")
    hyp_n = normalize_text(hyp, language).replace(" ", "" if language == "zh" else " ")
    if not ref_n:
        return None
    return float(jiwer.cer(ref_n, hyp_n))


def asr_metrics(ref_text: str, hyp_text: str, language: str) -> dict:
    counts = token_error_counts(tokens(ref_text, language), tokens(hyp_text, language))
    return {
        "wer": counts.rate,
        "cer": char_error_rate(ref_text, hyp_text, language),
        "wer_substitutions": counts.substitutions,
        "wer_deletions": counts.deletions,
        "wer_insertions": counts.insertions,
        "ref_token_count": counts.ref_len,
        "token_unit": "char" if language == "zh" else "word",
    }
