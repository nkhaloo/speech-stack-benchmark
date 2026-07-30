"""Deferred-labeling stack: streaming ASR + one whole-file diarization pass
(``runtime: batchdiar_stack``).

This models a real desktop product shape, not an ablation. A meeting recorder
shows a running transcript *while* someone is talking, then attaches speaker
labels when the user stops recording. Text is needed live; speaker attribution
is needed only at the end. So:

  * ASR re-runs every ``emit_every_sec`` over the audio buffered so far
    (``policy: growing``) or over the trailing ``window_sec``
    (``policy: sliding``) — this is the live path;
  * diarization runs ONCE per recording, in batch, over the complete audio,
    producing absolute-time turns that are never recomputed.

The implementation diarizes at ``reset`` (before the audio is streamed) rather
than at ``flush``. That is an efficiency detail, not a semantic one: the turns
are held fixed for the whole session either way, so every emission carries the
same labels the end-of-session pass would have produced. Doing it up front
means emissions are speaker-attributed as they go, which lets the existing
streaming metrics score attribution without a separate replay.

What this buys, and why it is the CPU baseline:

  * **Constant ASR cost.** A sliding window keeps per-step compute flat, which
    a desktop CPU needs — a growing buffer is quadratic in session length.
  * **No speaker-label churn.** Sliding windows normally re-cluster from
    scratch each step, so ``SPEAKER_00`` need not mean the same person twice.
    Here labels come from one global pass, so they are consistent by
    construction and ``speaker_label_churn`` should be ~0. Getting both
    constant cost and stable identities is the whole point of the pairing.
  * **Diarizer cost paid once**, not once per window — which is what makes the
    expensive-but-better diarizer (pyannote on CPU) viable at all.

LABEL LATENCY — the one thing this design gives up. Because the diarization
pass consumes the whole file, speaker labels are not available mid-session:
their latency is end-of-session, not sub-second. Every emission log and metrics
row carries ``diarization_mode: "batch_once"``, ``labels_finalized:
"end_of_session"``, and ``causal: false`` (the diarization pass is not causal,
even though the text path is) so this arm is never silently compared against a
stack that does label live. ``cpu-stream-fw-small-sherpa-windowed`` is that
live-labeling comparison, on identical models.

COST ACCOUNTING — the runner times only ``push``/``flush``, so the one-off
diarization pass is excluded from ``runtime_sec`` and therefore from
``streaming_rtf``. That is correct for judging whether the *live* path keeps up
with real time, but it means streaming_rtf is not the stack's total cost. The
pass is timed separately and reported as ``batch_diarization_sec`` in
``model_meta``; total ≈ ``streaming_rtf * duration + batch_diarization_sec``.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from ..asr import create_asr_adapter
from ..audio import write_wav
from ..diarization import create_diarization_adapter
from ..schemas import Emission, Recording, SpeakerTurn
from .base import AdapterUnavailable, StreamingAdapter
from .tracking import SentenceTracker
from .util import absolute_sentences, window_start
from .windowed import _load_card


class BatchDiarStreamingAdapter(StreamingAdapter):
    """Windowed streaming ASR fused against a fixed, whole-file diarization."""

    def _load(self) -> None:
        if "asr" not in self.config or "diarization" not in self.config:
            raise AdapterUnavailable(
                "batchdiar_stack needs 'asr' and 'diarization' model cards")
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
        self._turns: list[SpeakerTurn] = []
        self._diar_sec: float | None = None

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
        # The whole-file diarization pass: once per recording, never revisited.
        t0 = time.perf_counter()
        self._turns = list(self._diar.diarize(recording).turns)
        self._diar_sec = time.perf_counter() - t0

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
        # Fresh SpeakerTurn copies each pass: absolute_sentences hands them to
        # fuse(), and the cached turns must survive every window unmodified.
        turns = [SpeakerTurn(t.speaker, t.start, t.end, t.confidence)
                 for t in self._turns]
        current = absolute_sentences(asr, turns, win_start, self._gap)
        return self._tracker.update(current, audio_time_end, final=final)

    # -- metadata -----------------------------------------------------------
    def contract_meta(self) -> dict:
        w = self.config.get("window", {}) or {}
        return {
            "native": False,
            "policy": w.get("policy"),
            "emit_every_sec": w.get("emit_every_sec"),
            "window_sec": w.get("window_sec"),
            "finalize_after_sec": w.get("finalize_after_sec"),
            "asr_mode": "windowed",
            "diarization_mode": "batch_once",
            "labels_finalized": "end_of_session",
            "causal": False,   # the diarization pass; the text path is causal
        }

    def model_meta(self) -> dict:
        meta = super().model_meta()
        meta["diarization_mode"] = "batch_once"
        diar_sec = getattr(self, "_diar_sec", None)
        if diar_sec is not None:
            meta["batch_diarization_sec"] = round(diar_sec, 3)
        return meta
