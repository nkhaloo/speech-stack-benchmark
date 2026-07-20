"""Shared base for streaming stacks that pair diart *online* diarization with a
*windowed* ASR arm. Subclasses only supply the ASR arm (a local model, or an
HTTP client to a streaming server); the diart feeding, ASR windowing, fusion,
and emission bookkeeping are identical and live here.

Used by:
  * ``diart_whisper``  — ASR arm = local faster-whisper
"""

from __future__ import annotations

import tempfile
from abc import abstractmethod
from pathlib import Path

import numpy as np

from ..audio import write_wav
from ..schemas import ASRResult, Emission, Recording
from .base import StreamingAdapter
from .online_diar import OnlineDiarizer
from .tracking import SentenceTracker
from .util import absolute_sentences, window_start


class OnlineDiarWindowedAdapter(StreamingAdapter):
    # -- subclass hook ------------------------------------------------------
    @abstractmethod
    def _build_asr(self):
        """Return an object with ``transcribe(recording, language) -> ASRResult``
        and ``unload()``; raise AdapterUnavailable if unavailable."""

    # -- lifecycle ----------------------------------------------------------
    def _load(self) -> None:
        self._diar = OnlineDiarizer(self.config.get("diart", {}))
        self._diar.load()
        self._sr = self._diar.sample_rate
        self._asr = self._build_asr()
        w = self.config.get("window", {}) or {}
        self._policy = w.get("policy", "growing")
        self._emit_every = float(w.get("emit_every_sec", 2.0))
        self._window_sec = float(w.get("window_sec", 30.0))
        self._gap = float(self.config.get("sentence_gap_sec", 0.8))
        self._finalize_after = float(w.get("finalize_after_sec", 5.0))
        self._match_tol = float(w.get("match_tolerance_sec", 1.0))

    def _unload(self) -> None:
        asr = getattr(self, "_asr", None)
        if asr is not None and hasattr(asr, "unload"):
            asr.unload()

    def reset(self, recording: Recording) -> None:
        self._diar.reset()
        self._buf = np.zeros(0, dtype=np.float32)
        self._language = recording.language if self.config.get("force_language") else None
        self._last_emit_audio = -1e9
        self._tracker = SentenceTracker(self._finalize_after, self._match_tol)
        self._tmpdir = tempfile.TemporaryDirectory()

    # -- streaming ----------------------------------------------------------
    def push(self, audio: np.ndarray, audio_time_end: float) -> list[Emission]:
        self._buf = np.concatenate([self._buf, audio.astype(np.float32)])
        self._diar.feed(self._buf)
        if audio_time_end - self._last_emit_audio < self._emit_every:
            return []
        self._last_emit_audio = audio_time_end
        return self._emit(audio_time_end, final=False)

    def flush(self) -> list[Emission]:
        total = len(self._buf) / self._sr
        self._diar.feed(self._buf)
        out = self._emit(total, final=True)
        self._tmpdir.cleanup()
        return out

    def _emit(self, audio_time_end: float, final: bool) -> list[Emission]:
        if len(self._buf) == 0:
            return []
        win_start = window_start(audio_time_end, self._policy, self._window_sec)
        chunk = self._buf[int(win_start * self._sr):]
        wav = Path(self._tmpdir.name) / "asr.wav"
        write_wav(wav, chunk, self._sr)
        rec = Recording(recording_id="__stream", dataset="stream",
                        language=self._language or "und", audio_path=str(wav))
        asr: ASRResult = self._asr.transcribe(rec, language=self._language)
        current = absolute_sentences(asr, self._diar.current_turns(), win_start, self._gap)
        return self._tracker.update(current, audio_time_end, final=final)
