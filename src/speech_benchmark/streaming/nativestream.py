"""Deferred-labeling stack with a *natively streaming* ASR
(``runtime: native_batchdiar_stack``).

Same product shape as ``batchdiar_stack`` — a desktop recorder that shows a
running transcript while someone is still talking and attaches speaker labels
when they stop — but the live path is a real streaming decoder instead of a
batch model re-run over a sliding window:

  * ASR keeps decoder state across frames, so every sample is decoded **once**;
  * diarization runs ONCE per recording, in batch, over the complete audio,
    producing absolute-time turns that are never recomputed.

Why this replaced the windowed-Whisper arms it is otherwise identical to: a
sliding window re-transcribes its whole buffer every emit step, which on a
desktop CPU measured above 2.0 real-time factor with faster-whisper small — the
live path falls permanently behind the speaker. A streaming decoder measured
~0.02. Windowing was never the point; it was a workaround for ASR models that
cannot stream, and the workaround is what made the CPU track unaffordable.

What carries over unchanged from ``batchdiar.py``:

  * **Diarization cost is paid once**, not once per window, which is what makes
    the expensive-but-better diarizer (pyannote on CPU) viable at all.
  * **No speaker-label churn.** Labels come from one global pass, so
    ``SPEAKER_00`` means the same person for the whole session by construction
    and ``speaker_label_churn`` should be ~0. A nonzero value indicates a bug,
    not a model quality signal.
  * **Label latency is end-of-session** — the price of the design. Rows carry
    ``diarization_mode: batch_once``, ``labels_finalized: end_of_session`` and
    ``causal: false`` (the diarization pass is not causal; the text path is), so
    this arm is never silently compared against a stack that labels live.

COST ACCOUNTING — the runner times only ``push``/``flush``, so the one-off
diarization pass is excluded from ``runtime_sec`` and therefore from
``streaming_rtf``. That is the right number for "does text keep up with speech"
(check it against 1.0), but it is not the stack's total cost. The pass is timed
separately as ``batch_diarization_sec`` in ``model_meta``; total ≈
``streaming_rtf * duration + batch_diarization_sec``. With a streaming decoder
the diarization pass is now the dominant cost, which is the opposite of the
windowed regime and worth keeping in mind when reading the numbers.

LANGUAGE COVERAGE — unlike the multilingual Whisper arms, a streaming model may
have weights for only some of the benchmark's languages. ``reset`` propagates
:class:`UnsupportedLanguage` and the runner marks that recording ``skipped``.
"""

from __future__ import annotations

import time

import numpy as np

from ..diarization import create_diarization_adapter
from ..schemas import ASRResult, ASRSegment, Emission, Recording, SpeakerTurn
from .base import AdapterUnavailable, StreamingAdapter
from .online_asr import create_online_asr
from .tracking import SentenceTracker
from .util import absolute_sentences
from .windowed import _load_card


class NativeStreamBatchDiarAdapter(StreamingAdapter):
    """Streaming ASR fused against a fixed, whole-file diarization."""

    def _load(self) -> None:
        if "asr" not in self.config or "diarization" not in self.config:
            raise AdapterUnavailable(
                "native_batchdiar_stack needs 'asr' and 'diarization' model cards")
        self._asr = create_online_asr(_load_card(self.config["asr"]))
        self._diar = create_diarization_adapter(_load_card(self.config["diarization"]))
        self._asr.load()
        self._diar.load()
        self._sr = 16000
        w = self.config.get("window", {}) or {}
        self._emit_every = float(w.get("emit_every_sec", 1.0))
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
        # Language first: an unsupported language must skip the recording
        # before any diarization time is spent on it.
        self._asr.reset(recording.language)
        self._language = recording.language
        self._last_emit_audio = -1e9
        self._tracker = SentenceTracker(self._finalize_after, self._match_tol)
        # The whole-file diarization pass: once per recording, never revisited.
        t0 = time.perf_counter()
        self._turns = list(self._diar.diarize(recording).turns)
        self._diar_sec = time.perf_counter() - t0

    def push(self, audio: np.ndarray, audio_time_end: float) -> list[Emission]:
        self._asr.accept(audio.astype(np.float32))
        if audio_time_end - self._last_emit_audio < self._emit_every:
            return []
        self._last_emit_audio = audio_time_end
        return self._emit(audio_time_end, final=False)

    def flush(self) -> list[Emission]:
        self._asr.finalize()
        words = self._asr.words()
        end = max([w.end for w in words if w.end is not None], default=0.0)
        return self._emit(max(end, self._last_emit_audio), final=True)

    def _emit(self, audio_time_end: float, final: bool) -> list[Emission]:
        words = self._asr.words()
        if not words:
            return []
        # The decoder already reports absolute stream time, so win_start is 0 —
        # absolute_sentences still owns word->speaker assignment and sentence
        # splitting, keeping this identical to every other stack's fusion.
        asr = ASRResult(recording_id="__stream", model_id=self.model_id,
                        text=" ".join(w.text for w in words),
                        segments=[ASRSegment(
                            text=" ".join(w.text for w in words),
                            start=words[0].start, end=words[-1].end,
                            words=words)])
        turns = [SpeakerTurn(t.speaker, t.start, t.end, t.confidence)
                 for t in self._turns]
        current = absolute_sentences(asr, turns, 0.0, self._gap)
        return self._tracker.update(current, audio_time_end, final=final)

    # -- metadata -----------------------------------------------------------
    def contract_meta(self) -> dict:
        w = self.config.get("window", {}) or {}
        return {
            "native": True,
            "policy": "native",           # no buffer window: audio decoded once
            "emit_every_sec": w.get("emit_every_sec"),
            "window_sec": None,
            "finalize_after_sec": w.get("finalize_after_sec"),
            "asr_mode": "native_streaming",
            "diarization_mode": "batch_once",
            "labels_finalized": "end_of_session",
            "causal": False,   # the diarization pass; the text path is causal
        }

    def model_meta(self) -> dict:
        meta = super().model_meta()
        meta["diarization_mode"] = "batch_once"
        meta["asr_mode"] = "native_streaming"
        asr = getattr(self, "_asr", None)
        if asr is not None and hasattr(asr, "model_meta"):
            meta.update(asr.model_meta())
        diar_sec = getattr(self, "_diar_sec", None)
        if diar_sec is not None:
            meta["batch_diarization_sec"] = round(diar_sec, 3)
        return meta
