"""Diarization metrics via pyannote.metrics.

Scoring settings (recorded with every run, see docs/methodology.md):
  * DER collar: 0.5 s total (±0.25 s around each reference boundary) — the
    common "fair" setting; configurable via metrics config.
  * Overlap: scored (skip_overlap=False). The synthetic baseline data has no
    overlapped speech, so this matters only for real-audio anchors.
"""

from __future__ import annotations

from pyannote.core import Annotation, Segment

from ..schemas import ReferenceTurn, SpeakerTurn


def _to_annotation(turns: list, uri: str = "rec") -> Annotation:
    ann = Annotation(uri=uri)
    for i, t in enumerate(turns):
        ann[Segment(t.start, t.end), i] = t.speaker
    return ann


def diarization_metrics(
    ref_turns: list[ReferenceTurn],
    hyp_turns: list[SpeakerTurn],
    collar: float = 0.5,
    skip_overlap: bool = False,
) -> dict:
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref = _to_annotation(ref_turns)
    hyp = _to_annotation(hyp_turns)
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    ends = [t.end for t in list(ref_turns) + list(hyp_turns)] or [0.0]
    uem = Segment(0.0, max(ends))
    components = metric(ref, hyp, uem=uem, detailed=True)

    total = components.get("total", 0.0) or 0.0
    def rate(key: str) -> float | None:
        return (components.get(key, 0.0) / total) if total else None

    ref_spk = len({t.speaker for t in ref_turns})
    hyp_spk = len({t.speaker for t in hyp_turns})
    return {
        "der": components.get("diarization error rate"),
        "missed_speech": rate("missed detection"),
        "false_alarm_speech": rate("false alarm"),
        "speaker_confusion": rate("confusion"),
        "ref_speech_sec": total,
        "ref_num_speakers": ref_spk,
        "hyp_num_speakers": hyp_spk,
        "speaker_count_error": hyp_spk - ref_spk,
        "der_collar": collar,
        "der_skip_overlap": skip_overlap,
    }
