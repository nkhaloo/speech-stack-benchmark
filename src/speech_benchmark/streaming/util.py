"""Small helpers shared by the windowing/streaming adapters."""

from __future__ import annotations

from ..fusion import fuse
from ..schemas import ASRResult, DiarizationResult, SpeakerTurn


def window_start(audio_time_end: float, policy: str, window_sec: float) -> float:
    return max(0.0, audio_time_end - window_sec) if policy == "sliding" else 0.0


def absolute_sentences(asr: ASRResult, diar_turns: list[SpeakerTurn],
                       win_start: float, gap: float) -> list[dict]:
    """Shift an ASR result produced over a buffer window back to absolute stream
    time, fuse it with the given (absolute-time) diarization turns, and return
    ``{start,end,text,speaker}`` sentence dicts for the SentenceTracker."""
    for seg in asr.segments:
        for wd in seg.words:
            if wd.start is not None:
                wd.start += win_start
            if wd.end is not None:
                wd.end += win_start
        if seg.start is not None:
            seg.start += win_start
        if seg.end is not None:
            seg.end += win_start
    diar = DiarizationResult(recording_id="__win", model_id="__diar", turns=diar_turns)
    combined = fuse(asr, diar, gap_threshold_sec=gap)
    return [{"start": s.start or 0.0, "end": s.end or 0.0,
             "text": s.text, "speaker": s.speaker}
            for s in combined.sentences if s.text.strip()]
