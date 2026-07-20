"""Generic windowing wrapper: turn any batch ASR + batch diarization pair into
a streaming baseline (`runtime: windowed_stack`).

It buffers incoming audio and, every ``emit_every_sec`` of audio, re-runs the
batch ASR and diarizer over the buffer (``policy: growing`` keeps the whole
conversation so speaker IDs stay globally consistent; ``policy: sliding`` keeps
only the last ``window_sec``), fuses them, and diffs the fused sentences against
what it has already emitted (via the shared ``SentenceTracker``). A sentence
finalizes once its audio end is older than ``finalize_after_sec``.

This is the *baseline* arm of the streaming benchmark: the exact batch models
run incrementally, so their streaming penalty (accuracy Δ, latency, label churn)
is measured against their own batch numbers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ..asr import create_asr_adapter
from ..audio import write_wav
from ..config import load_yaml, resolve_path
from ..diarization import create_diarization_adapter
from ..schemas import Emission, Recording, SpeakerTurn
from .base import AdapterUnavailable, StreamingAdapter
from .tracking import SentenceTracker
from .util import absolute_sentences, window_start


def _load_card(spec) -> dict:
    """A stack references its ASR/diarizer either inline (dict) or by config
    path (str), mirroring how track configs reference model cards."""
    if isinstance(spec, dict):
        return dict(spec)
    if isinstance(spec, str):
        return load_yaml(resolve_path(spec))
    raise AdapterUnavailable(f"windowed_stack: bad asr/diarization spec {spec!r}")


class WindowedStackStreamingAdapter(StreamingAdapter):
    def _load(self) -> None:
        if "asr" not in self.config or "diarization" not in self.config:
            raise AdapterUnavailable(
                "windowed_stack needs 'asr' and 'diarization' model cards")
        self._asr = create_asr_adapter(_load_card(self.config["asr"]))
        self._diar = create_diarization_adapter(_load_card(self.config["diarization"]))
        self._asr.load()
        self._diar.load()
        self._sr = 16000
        w = self.config.get("window", {}) or {}
        self._policy = w.get("policy", "growing")
        self._emit_every = float(w.get("emit_every_sec", 2.0))
        self._window_sec = float(w.get("window_sec", 30.0))
        self._finalize_after = float(w.get("finalize_after_sec", 5.0))
        self._match_tol = float(w.get("match_tolerance_sec", 1.0))
        self._gap = float(self.config.get("sentence_gap_sec", 0.8))

    def _unload(self) -> None:
        for a in (getattr(self, "_asr", None), getattr(self, "_diar", None)):
            if a is not None:
                a.unload()

    def reset(self, recording: Recording) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._language = recording.language if self.config.get("force_language") else None
        self._last_emit_audio = -1e9
        self._tracker = SentenceTracker(self._finalize_after, self._match_tol)
        self._tmpdir = tempfile.TemporaryDirectory()

    def push(self, audio: np.ndarray, audio_time_end: float) -> list[Emission]:
        self._buf = np.concatenate([self._buf, audio.astype(np.float32)])
        if audio_time_end - self._last_emit_audio < self._emit_every:
            return []
        self._last_emit_audio = audio_time_end
        return self._emit(audio_time_end, final=False)

    def flush(self) -> list[Emission]:
        total = len(self._buf) / self._sr
        out = self._emit(total, final=True)
        self._tmpdir.cleanup()
        return out

    def _emit(self, audio_time_end: float, final: bool) -> list[Emission]:
        if len(self._buf) == 0:
            return []
        win_start = window_start(audio_time_end, self._policy, self._window_sec)
        chunk = self._buf[int(win_start * self._sr):]
        wav = Path(self._tmpdir.name) / "win.wav"
        write_wav(wav, chunk, self._sr)
        rec = Recording(recording_id="__win", dataset="stream",
                        language=self._language or "und", audio_path=str(wav))
        asr = self._asr.transcribe(rec, language=self._language)
        diar = self._diar.diarize(rec)
        # shift diarization turns from window-relative to absolute stream time
        turns = [SpeakerTurn(t.speaker, t.start + win_start, t.end + win_start,
                             t.confidence) for t in diar.turns]
        current = absolute_sentences(asr, turns, win_start, self._gap)
        return self._tracker.update(current, audio_time_end, final=final)
