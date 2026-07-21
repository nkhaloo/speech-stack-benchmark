"""Native streaming stack: diart + WhisperLive (large-v3-turbo)."""

from __future__ import annotations

import numpy as np

from ..schemas import ASRResult, ASRSegment, Emission, Recording, Word
from .base import StreamingAdapter
from .online_diar import OnlineDiarizer
from .tracking import SentenceTracker
from .util import absolute_sentences
from .whisperlive_client import WhisperLiveClient


class DiartWhisperLiveStreamingAdapter(StreamingAdapter):
    def _load(self) -> None:
        self._diar = OnlineDiarizer(self.config.get("diart", {}))
        self._diar.load()
        self._sr = self._diar.sample_rate
        w = self.config.get("whisperlive", {}) or {}
        self._client_factory = lambda: WhisperLiveClient(w)
        self._response_wait = float(w.get("response_wait_sec", 0.05))
        self._gap = float(self.config.get("sentence_gap_sec", 0.8))
        self._finalize_after = float(w.get("finalize_after_sec", 5.0))
        self._match_tol = float(w.get("match_tolerance_sec", 1.0))

    def _unload(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None:
            client.close()

    def reset(self, recording: Recording) -> None:
        old = getattr(self, "_client", None)
        if old is not None:
            old.close()
        self._diar.reset()
        self._buf = np.zeros(0, dtype=np.float32)
        self._tracker = SentenceTracker(self._finalize_after, self._match_tol)
        self._client = self._client_factory()
        language = recording.language if self.config.get("force_language", True) else None
        self._client.connect(language)

    def push(self, audio: np.ndarray, audio_time_end: float) -> list[Emission]:
        chunk = np.asarray(audio, dtype=np.float32)
        self._buf = np.concatenate([self._buf, chunk])
        self._diar.feed(self._buf)
        self._client.send(chunk)
        return self._emit(self._client.snapshot(self._response_wait),
                          audio_time_end, final=False)

    def flush(self) -> list[Emission]:
        total = len(self._buf) / self._sr
        self._diar.feed(self._buf)
        segments = self._client.finish()
        self._client = None
        if not segments:
            raise RuntimeError(
                "WhisperLive returned no segments. Check the server log for a "
                "CUDA or transcription error; refusing to cache an empty run."
            )
        return self._emit(segments, total, final=True)

    def _emit(self, raw: list[dict], audio_time: float,
              final: bool) -> list[Emission]:
        segments = []
        for s in raw:
            start, end = _number(s.get("start")), _number(s.get("end"))
            words = [Word(
                text=str(w.get("word", "")).strip(),
                start=_number(w.get("start")), end=_number(w.get("end")),
                confidence=_number(w.get("probability")),
            ) for w in (s.get("words") or []) if str(w.get("word", "")).strip()]
            segments.append(ASRSegment(
                text=str(s.get("text", "")).strip(), start=start, end=end,
                words=words))
        asr = ASRResult(recording_id="__stream", model_id=self.model_id,
                        text=" ".join(s.text for s in segments), segments=segments)
        current = absolute_sentences(asr, self._diar.current_turns(), 0.0, self._gap)
        return self._tracker.update(current, audio_time, final=final)


def _number(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
