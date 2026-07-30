"""Contract and registry for *natively streaming* ASR models.

The batch :class:`~speech_benchmark.asr.base.ASRAdapter` contract is
``transcribe(recording) -> ASRResult``: it consumes a complete file and returns
a complete answer. That is the wrong shape for a model that decodes
incrementally. Feeding a batch model successive windows of a growing buffer
(what ``windowed_stack`` / ``batchdiar_stack`` do) simulates streaming but pays
to re-transcribe the same audio over and over — on a desktop CPU that is the
difference between 0.02 and >2.0 real-time factor.

An :class:`OnlineASR` instead holds decoder state across calls:

  * ``reset(language)`` — start a new stream (raises :class:`UnsupportedLanguage`
    if the model has no weights for it);
  * ``accept(audio)`` — feed one float32 mono 16 kHz frame; decode incrementally;
  * ``words()`` — the best current word hypothesis in absolute stream time,
    finalized words plus the in-flight partial. Callers must treat the tail as
    revisable: that instability is exactly what the stability metrics measure;
  * ``finalize()`` — flush the decoder at end of stream.

``words()`` returning **word-level timestamps is mandatory**, not a nicety.
Speaker attribution (``fusion/assign.py``) assigns each word to the diarization
turn it overlaps most, and cpWER builds its hypothesis stream from word-level
speaker labels. An online ASR that emits only text cannot be scored here.

Adapters are registered below and referenced by ``runtime:`` in an ASR model
card, exactly like batch adapters.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..asr.base import AdapterUnavailable
from ..schemas import Word


class UnsupportedLanguage(RuntimeError):
    """This model has no weights for the recording's language.

    Distinct from :class:`AdapterUnavailable`, which means the whole stack
    cannot run at all. This is per-recording and expected: unlike the
    multilingual Whisper arms, a streaming model may cover only part of the
    benchmark's five languages, and those recordings must be recorded as
    ``skipped`` rather than ``failed`` so they neither pollute the error log nor
    let the leaderboard read a missing language as a bad score.
    """


class OnlineASR(ABC):
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

    def _load(self) -> None:  # pragma: no cover - trivial default
        pass

    def _unload(self) -> None:  # pragma: no cover - trivial default
        pass

    # -- languages ----------------------------------------------------------
    def languages(self) -> list[str]:
        """Languages this card covers.

        ``languages:`` for a single multilingual checkpoint; the keys of a
        ``models:`` map for families that ship one checkpoint per language.
        """
        return sorted(self.config.get("languages")
                      or (self.config.get("models") or {}).keys())

    def supports_language(self, language: Optional[str]) -> bool:
        langs = self.languages()
        return bool(language) and language in langs

    # -- streaming ----------------------------------------------------------
    @abstractmethod
    def reset(self, language: Optional[str]) -> None:
        """Begin a new stream in ``language``; drop all previous decoder state."""

    @abstractmethod
    def accept(self, audio: np.ndarray) -> None:
        """Feed one float32 mono 16 kHz frame."""

    @abstractmethod
    def words(self) -> list[Word]:
        """Best current hypothesis, absolute stream time, word-level timing."""

    def finalize(self) -> None:
        """End of stream; fold any in-flight partial into the final hypothesis."""


_REGISTRY = {
    "vosk": "speech_benchmark.streaming.vosk_asr.VoskOnlineASR",
    "sherpa_streaming": (
        "speech_benchmark.streaming.sherpa_stream_asr.SherpaStreamingASR"
    ),
}


def create_online_asr(model_config: dict) -> OnlineASR:
    runtime = model_config.get("runtime")
    if runtime not in _REGISTRY:
        raise KeyError(
            f"Unknown online ASR runtime {runtime!r} for model "
            f"{model_config.get('id')!r}; known: {sorted(_REGISTRY)}"
        )
    module_path, cls_name = _REGISTRY[runtime].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(model_config)


__all__ = ["OnlineASR", "UnsupportedLanguage", "AdapterUnavailable",
           "create_online_asr"]
