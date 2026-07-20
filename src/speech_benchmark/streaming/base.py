"""Base class for streaming (incremental) speech-stack adapters.

Where batch adapters expose ``transcribe`` / ``diarize`` over a whole file, a
streaming adapter is *driven* by the runner one audio frame at a time and emits
speaker-attributed sentences that may be revised before they finalize:

  * ``load()`` / ``unload()`` — acquire/release the underlying model(s). Unlike
    the batch runner, a streaming *stack* may legitimately hold both an ASR and
    a diarization model in memory at once — that is the nature of streaming.
  * ``reset()`` — clear all per-recording streaming state before a new recording.
  * ``push(audio, audio_time_end)`` — feed the next audio frame (float32 mono @
    16 kHz) whose final sample lands at ``audio_time_end`` seconds; return zero
    or more :class:`Emission`s produced now. ``wall_time`` is left for the runner
    to stamp (it owns the simulated real-time clock).
  * ``flush()`` — end of stream; return any final emissions.

Emissions returned from ``push``/``flush`` should set ``sentence_id`` (stable
across revisions of the same sentence), ``text``, ``speaker``, ``start``/``end``
(audio time), ``audio_time`` (= ``audio_time_end`` of the triggering frame),
``is_final``, and ``revision``. The runner fills ``wall_time``.
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod

import numpy as np

from ..asr.base import AdapterUnavailable  # re-exported for adapters
from ..schemas import Emission, Recording

__all__ = ["StreamingAdapter", "AdapterUnavailable"]


class StreamingAdapter(ABC):
    def __init__(self, model_config: dict):
        self.config = model_config
        self.model_id: str = model_config["id"]
        self._loaded = False

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    def unload(self) -> None:
        if self._loaded:
            self._unload()
            self._loaded = False
        gc.collect()
        try:  # release CUDA cache if torch happens to be present
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load(self) -> None:  # pragma: no cover - trivial default
        pass

    def _unload(self) -> None:  # pragma: no cover - trivial default
        pass

    # -- streaming ----------------------------------------------------------
    @abstractmethod
    def reset(self, recording: Recording) -> None:
        """Clear per-recording state so the next recording starts fresh.

        The :class:`Recording` is provided for audio metadata; only dummy
        adapters read its reference (mirroring the batch dummies) — real
        streaming stacks must not."""

    @abstractmethod
    def push(self, audio: np.ndarray, audio_time_end: float) -> list[Emission]:
        """Feed one audio frame; return emissions produced now (may be empty)."""

    def flush(self) -> list[Emission]:
        """End of stream. Default: nothing further to emit."""
        return []

    # -- metadata -----------------------------------------------------------
    def model_meta(self) -> dict:
        keys = ("id", "family", "runtime", "asr", "diarization", "model",
                "license", "weights_license", "device", "native")
        return {k: self.config.get(k) for k in keys if self.config.get(k) is not None}

    def contract_meta(self) -> dict:
        """The streaming contract this stack runs under (recorded in every row)."""
        w = self.config.get("window", {}) or {}
        return {
            "native": bool(self.config.get("native", False)),
            "policy": w.get("policy"),
            "emit_every_sec": w.get("emit_every_sec"),
            "window_sec": w.get("window_sec"),
            "finalize_after_sec": w.get("finalize_after_sec"),
        }
