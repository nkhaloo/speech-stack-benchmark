"""Streaming transducer ASR on sherpa-onnx (``runtime: sherpa_streaming``).

Drives NVIDIA **Nemotron 3.5 ASR streaming 0.6B** — a cache-aware
FastConformer-RNNT that processes each audio frame exactly once — through the
Apache-2.0 sherpa-onnx onnxruntime backend. The same code path also drives
sherpa-onnx's streaming zipformer transducers, which share the
``from_transducer`` factory.

Why this is the CPU track's ASR:

  * **One multilingual checkpoint covers all five benchmark languages.** Not
    five per-language models behind one runtime — a single set of weights, with
    the language selected per stream. Measured here on en/es/fr/ar/zh.
  * **Natively streaming.** Decoder state persists across ``accept`` calls, so
    audio is decoded once. Measured ~0.10 real-time factor on a desktop CPU,
    against >2.0 for re-transcribing a sliding window with faster-whisper —
    windowing was never the design, it was a workaround for ASR models that
    cannot stream, and that workaround is what made the CPU track unaffordable.
  * **Punctuation and capitalization are built in**, so the punctuation-based
    sentence splitting in ``fusion/assign.py`` actually fires. ASR families that
    emit lowercase unpunctuated text leave that path dead and fall back to
    silence gaps alone.
  * **OpenMDW-1.1** weights: permissive, commercial use explicit, no copyleft
    and no field-of-use restriction. See docs/licensing.md.

LANGUAGE SELECTION. The multilingual export takes an extra ``prompt_index``
encoder input, which sherpa-onnx auto-detects and drives from a per-stream
option: ``stream.set_option("language", "es")``. ``"auto"`` lets the model
detect. The option is re-applied after every endpoint reset, since the decoder
segment restarts there. A card lists what it supports under ``languages:``;
anything else raises :class:`UnsupportedLanguage` and the runner records that
recording as ``skipped`` rather than ``failed``.

TIMESTAMP CAVEAT. sherpa-onnx reports token timestamps relative to the *current
segment*, and the segment clock restarts every time the endpointer fires and we
reset the stream. Absolute stream time is reconstructed by adding the total
audio fed at the moment of the last reset — accurate to within one frame (0.5 s
at the track's frame size), comfortably inside the 0.5 s diarization collar.

Words are rebuilt from BPE tokens: a token beginning with ``▁`` opens a new
word, continuation tokens append to it. CJK models emit one character per token
with no word marker, so each character becomes its own word — which matches the
project's convention of scoring Chinese at character level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..config import project_root, resolve_path
from ..schemas import Word
from .online_asr import AdapterUnavailable, OnlineASR, UnsupportedLanguage

ROOT = project_root()
SAMPLE_RATE = 16000
WORD_MARK = "▁"          # BPE word-start marker
TAIL_SEC = 0.2                # assumed duration of a word with no successor


def _pick(directory: Path, stem: str, int8: bool) -> Path:
    """Find the encoder/decoder/joiner ONNX file, preferring the int8 export."""
    hits = sorted(p for p in directory.glob(f"*{stem}*.onnx"))
    if not hits:
        raise AdapterUnavailable(f"no *{stem}*.onnx under {directory}")
    quant = [p for p in hits if "int8" in p.name]
    plain = [p for p in hits if "int8" not in p.name]
    order = (quant + plain) if int8 else (plain + quant)
    return order[0]


def _tokens_to_words(tokens: list[str], stamps: list[float],
                     offset: float) -> list[Word]:
    words: list[Word] = []
    cur, start, prev_end = "", None, None
    for i, tok in enumerate(tokens):
        ts = stamps[i] if i < len(stamps) else None
        nxt = stamps[i + 1] if i + 1 < len(stamps) else None
        opens = tok.startswith(WORD_MARK) or not cur
        piece = tok[len(WORD_MARK):] if tok.startswith(WORD_MARK) else tok
        if opens and cur:
            words.append(Word(cur, start, prev_end, None))
            cur, start = "", None
        if not cur:
            start = (ts or 0.0) + offset
        cur += piece
        prev_end = ((nxt if nxt is not None else (ts or 0.0) + TAIL_SEC) + offset)
    if cur:
        words.append(Word(cur, start, prev_end, None))
    return [w for w in words if w.text.strip()]


class SherpaStreamingASR(OnlineASR):
    # -- lifecycle ----------------------------------------------------------
    def _load(self) -> None:
        try:
            import sherpa_onnx
        except ImportError as e:  # pragma: no cover - env-dependent
            raise AdapterUnavailable(
                "sherpa-onnx not installed (install the [cpu] extra)") from e
        self._sherpa = sherpa_onnx
        if not self.config.get("model"):
            raise AdapterUnavailable(
                f"{self.model_id}: card has no `model:` bundle directory")
        self._dir = resolve_path(self.config["model"], ROOT)
        if not self._dir.exists():
            raise AdapterUnavailable(
                f"{self.model_id}: model bundle not found ({self._dir}); run "
                "scripts/download_models.py --track cpu_streaming")
        tokens = self._dir / "tokens.txt"
        if not tokens.exists():
            raise AdapterUnavailable(f"tokens.txt missing under {self._dir}")
        int8 = bool(self.config.get("use_int8", True))
        self._rec = self._sherpa.OnlineRecognizer.from_transducer(
            tokens=str(tokens),
            encoder=str(_pick(self._dir, "encoder", int8)),
            decoder=str(_pick(self._dir, "decoder", int8)),
            joiner=str(_pick(self._dir, "joiner", int8)),
            num_threads=int(self.config.get("num_threads", 4)),
            sample_rate=SAMPLE_RATE,
            feature_dim=int(self.config.get("feature_dim", 80)),
            decoding_method=self.config.get("decoding_method", "greedy_search"),
            enable_endpoint_detection=True,
        )
        self._stream = None

    def _unload(self) -> None:
        self._rec = None
        self._stream = None

    # -- languages ----------------------------------------------------------
    def languages(self) -> list[str]:
        return sorted(self.config.get("languages") or [])

    # -- streaming ----------------------------------------------------------
    def reset(self, language: Optional[str]) -> None:
        langs = self.languages()
        if langs and (not language or language not in langs):
            raise UnsupportedLanguage(
                f"{self.model_id} does not cover language {language!r} "
                f"(has: {', '.join(langs)})")
        self._lang = language or "auto"
        self._stream = self._rec.create_stream()
        self._apply_language()
        self._final: list[Word] = []
        self._partial: list[Word] = []
        self._fed_sec = 0.0        # total audio handed to the decoder
        self._offset = 0.0         # stream time at which the current segment began

    def _apply_language(self) -> None:
        """Condition the encoder on the language for this stream.

        A no-op on monolingual exports (no ``prompt_index`` input), so the same
        adapter drives the streaming zipformers unchanged.
        """
        try:
            self._stream.set_option("language", self._lang)
        except Exception:  # pragma: no cover - monolingual export
            pass

    def accept(self, audio: np.ndarray) -> None:
        if self._stream is None:
            raise RuntimeError("reset() must be called before accept()")
        samples = audio.astype(np.float32)
        self._fed_sec += len(samples) / SAMPLE_RATE
        self._stream.accept_waveform(SAMPLE_RATE, samples)
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)
        self._partial = self._current_words()
        if self._rec.is_endpoint(self._stream):
            # Segment settled. Fold it into the finals and restart the segment
            # clock — everything after this is timed from _offset.
            self._final.extend(self._partial)
            self._partial = []
            self._rec.reset(self._stream)
            self._apply_language()      # reset() clears per-segment conditioning
            self._offset = self._fed_sec

    def _current_words(self) -> list[Word]:
        r = self._rec.get_result_all(self._stream)
        return _tokens_to_words(list(r.tokens), list(r.timestamps), self._offset)

    def words(self) -> list[Word]:
        # Fresh copies every call: fusion writes speaker labels onto Word
        # objects, and the decoder's own hypothesis must survive untouched.
        return [Word(w.text, w.start, w.end, w.confidence)
                for w in (self._final + self._partial)]

    def finalize(self) -> None:
        if self._stream is None:
            return
        # Tail padding flushes the last chunk out of the encoder's lookahead.
        self._stream.accept_waveform(
            SAMPLE_RATE, np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32))
        self._stream.input_finished()
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)
        self._final.extend(self._current_words())
        self._partial = []

    # -- metadata -----------------------------------------------------------
    def model_meta(self) -> dict:
        return {"asr_id": self.model_id, "asr_runtime": "sherpa_streaming",
                "asr_languages": self.languages(),
                "chunk_ms": self.config.get("chunk_ms")}
