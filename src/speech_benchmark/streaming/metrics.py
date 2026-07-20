"""Streaming metrics — three axes (see docs/methodology_streaming.md §7):

  A. Final accuracy — reconstruct the finalized transcript from the emission log
     and score it with the *existing* batch metrics (WER/CER, DER, cpWER,
     attribution), so the streaming penalty is directly comparable to batch.
  B. Latency — time-to-first-token, and sentence finalization delay (emission
     wall-time minus the sentence's audio end), summarized median / p90.
  C. Stability — revision rate, speaker-label churn, token flicker: how much the
     output changes before it settles.

No blended score is produced; the three axes are reported side by side.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..fusion.assign import CombinedResult, CombinedSentence
from ..metrics.asr_metrics import asr_metrics
from ..metrics.combined_metrics import combined_metrics
from ..metrics.diar_metrics import diarization_metrics
from ..schemas import Emission, Reference, SpeakerTurn, StreamingResult, Word


def _synth_words(text: str, start: Optional[float], end: Optional[float],
                 speaker: Optional[str]) -> list[Word]:
    toks = text.split()
    if not toks:
        return []
    if start is None or end is None or end <= start:
        return [Word(t, speaker=speaker) for t in toks]
    dur = (end - start) / len(toks)
    return [Word(t, start + i * dur, start + (i + 1) * dur, speaker=speaker)
            for i, t in enumerate(toks)]


def reconstruct(sr: StreamingResult) -> tuple[CombinedResult, list[SpeakerTurn], str]:
    """Build the finalized (CombinedResult, hyp speaker turns, transcript text)
    from the last-known state of each sentence."""
    finals = sr.final_emissions()
    sentences, turns, texts = [], [], []
    for e in finals:
        words = _synth_words(e.text, e.start, e.end, e.speaker)
        sentences.append(CombinedSentence(text=e.text, start=e.start, end=e.end,
                                          speaker=e.speaker, words=words))
        texts.append(e.text)
        if e.speaker is not None and e.start is not None and e.end is not None \
                and e.end > e.start:
            turns.append(SpeakerTurn(e.speaker, e.start, e.end))
    combined = CombinedResult(recording_id=sr.recording_id, asr_model_id=sr.model_id,
                              diarization_model_id=sr.model_id, sentences=sentences,
                              num_speakers=len({t.speaker for t in turns}) or None)
    return combined, turns, " ".join(texts).strip()


def _latency_metrics(sr: StreamingResult) -> dict:
    if not sr.emissions:
        return {"time_to_first_token_sec": None,
                "finalization_delay_median_sec": None,
                "finalization_delay_p90_sec": None}
    ttft = min(e.wall_time for e in sr.emissions)
    delays = [max(0.0, e.wall_time - e.end)
              for e in sr.emissions if e.is_final and e.end is not None]
    return {
        "time_to_first_token_sec": ttft,
        "finalization_delay_median_sec": float(np.median(delays)) if delays else None,
        "finalization_delay_p90_sec": float(np.percentile(delays, 90)) if delays else None,
        "finalized_sentences": len(delays),
    }


def _stability_metrics(sr: StreamingResult) -> dict:
    by_id: dict[int, list[Emission]] = {}
    for e in sr.emissions:
        by_id.setdefault(e.sentence_id, []).append(e)
    n = len(by_id)
    if n == 0:
        return {"revision_rate": None, "speaker_label_churn": None,
                "token_flicker": None, "total_emissions": 0, "num_sentences": 0}
    revised = churned = 0
    flicker_total = 0
    for hist in by_id.values():
        if max(e.revision for e in hist) > 0:
            revised += 1
        speakers = {e.speaker for e in hist if e.speaker is not None}
        if len(speakers) > 1:
            churned += 1
        flicker_total += max(0, len({e.text for e in hist}) - 1)
    return {
        "revision_rate": revised / n,
        "speaker_label_churn": churned / n,
        "token_flicker": flicker_total / n,
        "total_emissions": len(sr.emissions),
        "num_sentences": n,
    }


def streaming_metrics(sr: StreamingResult, reference: Optional[Reference],
                      language: str, collar: float = 0.5,
                      skip_overlap: bool = False) -> dict:
    """All three axes for one streaming run over one recording."""
    out: dict = {}
    out.update(_latency_metrics(sr))
    out.update(_stability_metrics(sr))
    out["streaming_rtf"] = sr.real_time_factor

    if reference is None:
        return out
    combined, hyp_turns, hyp_text = reconstruct(sr)
    # A. final accuracy — reuse the batch metric stack
    out.update(asr_metrics(reference.text, hyp_text, language))
    try:
        out.update(combined_metrics(reference, combined, hyp_turns, language))
    except Exception:  # attribution/cpWER are best-effort; never abort a row
        pass
    if hyp_turns:
        try:
            out.update(diarization_metrics(reference.turns, hyp_turns,
                                           collar=collar, skip_overlap=skip_overlap))
        except Exception:
            pass
    return out
