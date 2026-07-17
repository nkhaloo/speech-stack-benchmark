"""Speaker-attributed transcription metrics for fused (ASR + diarization)
output.

  * cpWER — concatenated-minimum-permutation WER (the standard speaker-
    attributed metric, cf. CHiME-6 / meeteval): concatenate each speaker's
    words, find the speaker mapping minimizing total token errors (Hungarian
    assignment), report total errors / total reference tokens.
  * word_attribution_accuracy — % of hypothesis words whose (mapped) speaker
    matches the reference speaker active at the word's midpoint.
  * sentence_attribution_accuracy — same at sentence level, majority-overlap
    reference speaker per sentence.

Speaker mapping for the attribution metrics is computed from time overlap
between hypothesis diarization turns and reference turns (Hungarian on
overlap duration), which mirrors how DER maps speakers.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..fusion.assign import CombinedResult
from ..schemas import Reference, ReferenceTurn, SpeakerTurn
from .asr_metrics import token_error_counts
from .text_norm import tokens


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def map_speakers_by_time(ref_turns: list[ReferenceTurn],
                         hyp_turns: list[SpeakerTurn]) -> dict[str, str]:
    """Optimal hyp-speaker -> ref-speaker mapping by total overlap duration."""
    ref_spk = sorted({t.speaker for t in ref_turns})
    hyp_spk = sorted({t.speaker for t in hyp_turns})
    if not ref_spk or not hyp_spk:
        return {}
    ov = np.zeros((len(hyp_spk), len(ref_spk)))
    for i, h in enumerate(hyp_spk):
        for j, r in enumerate(ref_spk):
            ov[i, j] = sum(
                _overlap(ht.start, ht.end, rt.start, rt.end)
                for ht in hyp_turns if ht.speaker == h
                for rt in ref_turns if rt.speaker == r
            )
    rows, cols = linear_sum_assignment(-ov)
    return {hyp_spk[i]: ref_spk[j] for i, j in zip(rows, cols) if ov[i, j] > 0}


def cpwer(reference: Reference, combined: CombinedResult, language: str) -> dict:
    """cpWER over per-speaker concatenated token streams."""
    ref_streams: dict[str, list[str]] = {}
    for t in reference.turns:
        ref_streams.setdefault(t.speaker, []).extend(tokens(t.text, language))
    hyp_streams: dict[str, list[str]] = {}
    for s in combined.sentences:
        if s.words:  # word-level speakers, falling back to the sentence's
            for w in s.words:
                spk = w.speaker or s.speaker or "<none>"
                hyp_streams.setdefault(spk, []).extend(tokens(w.text, language))
        else:
            spk = s.speaker or "<none>"
            hyp_streams.setdefault(spk, []).extend(tokens(s.text, language))

    ref_ids = sorted(ref_streams)
    hyp_ids = sorted(hyp_streams)
    n = max(len(ref_ids), len(hyp_ids))
    total_ref = sum(len(v) for v in ref_streams.values())
    if total_ref == 0:
        return {"cpwer": None, "cpwer_ref_tokens": 0}

    # Square cost matrix of token errors; unmatched ref speakers cost all
    # their tokens as deletions, unmatched hyp speakers all theirs as insertions.
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            hyp_toks = hyp_streams[hyp_ids[i]] if i < len(hyp_ids) else []
            ref_toks = ref_streams[ref_ids[j]] if j < len(ref_ids) else []
            cost[i, j] = token_error_counts(ref_toks, hyp_toks).errors
    rows, cols = linear_sum_assignment(cost)
    total_errors = cost[rows, cols].sum()
    return {
        "cpwer": float(total_errors) / total_ref,
        "cpwer_ref_tokens": total_ref,
    }


def _ref_speaker_at(ref_turns: list[ReferenceTurn], t: float) -> str | None:
    for turn in ref_turns:
        if turn.start <= t <= turn.end:
            return turn.speaker
    return None


def attribution_metrics(reference: Reference, combined: CombinedResult,
                        hyp_turns: list[SpeakerTurn]) -> dict:
    mapping = map_speakers_by_time(reference.turns, hyp_turns)

    word_total = word_correct = 0
    for s in combined.sentences:
        for w in s.words:
            if w.start is None or w.end is None:
                continue
            ref_spk = _ref_speaker_at(reference.turns, (w.start + w.end) / 2)
            if ref_spk is None:
                continue  # word not inside any reference turn (gap/insertion)
            word_total += 1
            if w.speaker is not None and mapping.get(w.speaker) == ref_spk:
                word_correct += 1

    sent_total = sent_correct = 0
    for s in combined.sentences:
        if s.start is None or s.end is None:
            continue
        best_spk, best_ov = None, 0.0
        for t in reference.turns:
            ov = _overlap(s.start, s.end, t.start, t.end)
            if ov > best_ov:
                best_spk, best_ov = t.speaker, ov
        if best_spk is None:
            continue
        sent_total += 1
        if s.speaker is not None and mapping.get(s.speaker) == best_spk:
            sent_correct += 1

    return {
        "word_attribution_accuracy": word_correct / word_total if word_total else None,
        "word_attribution_count": word_total,
        "sentence_attribution_accuracy": sent_correct / sent_total if sent_total else None,
        "sentence_count": sent_total,
    }


def combined_metrics(reference: Reference, combined: CombinedResult,
                     hyp_turns: list[SpeakerTurn], language: str) -> dict:
    out = cpwer(reference, combined, language)
    out.update(attribution_metrics(reference, combined, hyp_turns))
    return out
