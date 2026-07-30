"""Vosk online ASR (``runtime: vosk``) — the CPU streaming track's workhorse.

Kaldi's online decoder behind a small Python binding: audio goes in
chunk-by-chunk, partial hypotheses come out continuously, and a segment
finalizes when the decoder's endpointer fires. Genuinely streaming — decoder
state persists across ``accept`` calls, so audio is decoded exactly once.

Why this model family, given it is not the most accurate option available:

  * **It is the only one that covers all five target languages.** The official
    sherpa-onnx streaming zipformers cover en/fr/zh but have no Spanish or
    Arabic model, and Whisper is not a streaming model at all. Vosk ships
    Apache-2.0 weights for every language in this benchmark.
  * **It is fast enough that streaming is not the bottleneck.** Measured ~0.02
    real-time factor on a desktop CPU for the small models.

License note: the Arabic model deliberately pinned in the cards is
``vosk-model-ar-mgb2-0.4`` (Apache-2.0), **not** ``vosk-model-ar-0.22-linto``,
which is AGPL and would impose copyleft on a shipped desktop product.

Per-language weights are a first for this repo — every other card names one
multilingual model. Cards therefore carry a ``models:`` map from language code
to model directory, and a recording whose language is absent raises
:class:`UnsupportedLanguage` so the runner records it as ``skipped``.

Accuracy caveat worth remembering when reading the transcripts: the small Vosk
models emit lowercase text with no punctuation. Sentence segmentation therefore
falls entirely to silence gaps and speaker changes (``fusion/assign.py`` also
splits on final punctuation, which never fires here). WER is computed on
normalized text so casing/punctuation do not penalize it, but sentence
boundaries are looser than a Whisper-derived transcript's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import project_root, resolve_path
from ..schemas import Word
from .online_asr import AdapterUnavailable, OnlineASR, UnsupportedLanguage

ROOT = project_root()
SAMPLE_RATE = 16000


def _words_from(payload: list[dict]) -> list[Word]:
    return [Word(text=w["word"], start=w.get("start"), end=w.get("end"),
                 confidence=w.get("conf"))
            for w in payload if w.get("word")]


class VoskOnlineASR(OnlineASR):
    # -- lifecycle ----------------------------------------------------------
    def _load(self) -> None:
        try:
            import vosk
        except ImportError as e:  # pragma: no cover - env-dependent
            raise AdapterUnavailable(
                "vosk not installed. On macOS install `vosk==0.3.44` — 0.3.45 "
                "ships no macOS wheel."
            ) from e
        vosk.SetLogLevel(-1)
        self._vosk = vosk
        self._paths: dict[str, Path] = {
            lang: resolve_path(p, ROOT)
            for lang, p in (self.config.get("models") or {}).items()
        }
        if not self._paths:
            raise AdapterUnavailable(
                f"{self.model_id}: card has no `models:` language map")
        if not any(p.exists() for p in self._paths.values()):
            raise AdapterUnavailable(
                f"{self.model_id}: no model directories found (looked under "
                f"{ROOT / 'artifacts/models/vosk'}); run "
                "scripts/download_models.py --track cpu_streaming")
        self._models: dict[str, object] = {}
        self._rec = None
        self._final: list[Word] = []
        self._partial: list[Word] = []

    def _unload(self) -> None:
        self._rec = None
        self._models.clear()

    # -- languages ----------------------------------------------------------
    def languages(self) -> list[str]:
        return sorted(self.config.get("models") or {})

    def _model_for(self, language: Optional[str]):
        if not language or language not in self._paths:
            raise UnsupportedLanguage(
                f"{self.model_id} has no model for language {language!r} "
                f"(has: {', '.join(self.languages()) or 'none'})")
        path = self._paths[language]
        if not path.exists():
            raise UnsupportedLanguage(
                f"{self.model_id}: model for {language!r} not downloaded ({path})")
        if language not in self._models:
            self._models[language] = self._vosk.Model(str(path))
        return self._models[language]

    # -- streaming ----------------------------------------------------------
    def reset(self, language: Optional[str]) -> None:
        model = self._model_for(language)          # may raise UnsupportedLanguage
        self._rec = self._vosk.KaldiRecognizer(model, SAMPLE_RATE)
        self._rec.SetWords(True)
        self._rec.SetPartialWords(True)
        self._final = []
        self._partial = []

    def accept(self, audio: np.ndarray) -> None:
        if self._rec is None:
            raise RuntimeError("reset() must be called before accept()")
        pcm = (np.clip(audio.astype(np.float32), -1.0, 1.0) * 32767.0) \
            .astype("<i2").tobytes()
        if self._rec.AcceptWaveform(pcm):
            # Endpointer fired: this segment is settled, the partial is spent.
            self._final.extend(_words_from(json.loads(self._rec.Result())
                                           .get("result", [])))
            self._partial = []
        else:
            self._partial = _words_from(
                json.loads(self._rec.PartialResult()).get("partial_result", []))

    def words(self) -> list[Word]:
        # Fresh copies every call: fusion writes speaker labels onto Word
        # objects, and the decoder's own hypothesis must survive untouched.
        return [Word(w.text, w.start, w.end, w.confidence)
                for w in (self._final + self._partial)]

    def finalize(self) -> None:
        if self._rec is None:
            return
        self._final.extend(_words_from(json.loads(self._rec.FinalResult())
                                       .get("result", [])))
        self._partial = []

    # -- metadata -----------------------------------------------------------
    def model_meta(self) -> dict:
        return {"asr_id": self.model_id, "asr_runtime": "vosk",
                "asr_languages": self.languages()}
