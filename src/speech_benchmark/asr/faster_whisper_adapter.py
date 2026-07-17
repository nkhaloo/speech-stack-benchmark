"""faster-whisper (CTranslate2) adapter — primary Whisper runtime for both
the GPU track (float16 on CUDA) and the CPU track (int8).

Config keys:
  model:          Whisper size or CT2 model path ("large-v3", "large-v3-turbo",
                  "medium", "small", or a local directory)
  device:         "cuda" | "cpu" | "auto"   (default auto)
  compute_type:   "float16" | "int8" | "int8_float16" | "auto"
  beam_size:      default 5
  vad_filter:     default true
  word_timestamps: default true
  download_root:  cache dir for weights (default artifacts/models/faster-whisper)
"""

from __future__ import annotations

from typing import Optional

from ..config import project_root
from ..schemas import ASRResult, ASRSegment, Recording, Word
from .base import AdapterUnavailable, ASRAdapter


class FasterWhisperAdapter(ASRAdapter):
    def _load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:  # pragma: no cover
            raise AdapterUnavailable(
                "faster-whisper is not installed in this environment "
                "(install the [cpu] or [gpu] extra)"
            ) from e
        self._model = WhisperModel(
            self.config.get("model", "small"),
            device=self.config.get("device", "auto"),
            compute_type=self.config.get("compute_type", "auto"),
            download_root=str(
                self.config.get("download_root")
                or project_root() / "artifacts" / "models" / "faster-whisper"
            ),
        )

    def _unload(self) -> None:
        del self._model

    def transcribe(self, recording: Recording, language: Optional[str] = None) -> ASRResult:
        seg_iter, info = self._model.transcribe(
            recording.audio_path,
            language=language,
            beam_size=int(self.config.get("beam_size", 5)),
            vad_filter=bool(self.config.get("vad_filter", True)),
            word_timestamps=bool(self.config.get("word_timestamps", True)),
        )
        segments: list[ASRSegment] = []
        for seg in seg_iter:  # generator: transcription happens here
            words = [
                Word(w.word.strip(), float(w.start), float(w.end), float(w.probability))
                for w in (seg.words or [])
            ]
            segments.append(ASRSegment(
                text=seg.text.strip(), start=float(seg.start), end=float(seg.end),
                confidence=float(getattr(seg, "avg_logprob", 0.0)), words=words,
            ))
        return ASRResult(
            recording_id=recording.recording_id,
            model_id=self.model_id,
            text=" ".join(s.text for s in segments),
            segments=segments,
            language_requested=language,
            language_detected=info.language,
            language_probability=float(info.language_probability or 0.0),
            model_meta=self.model_meta(),
        )
