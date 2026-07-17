"""Voxtral Mini adapter (GPU track, experimental) — thin client for a locally
hosted vLLM server.

Voxtral Mini (Apache-2.0, open weights) runs behind vLLM in its own
environment (see scripts/setup_linux_gpu.sh --with-voxtral). vLLM exposes an
OpenAI-compatible ``/v1/audio/transcriptions`` endpoint on localhost; this
adapter posts audio to it. This is still fully self-hosted/offline — the
"server" is a local process on the same machine, not a hosted API.

Limitations (documented in docs/model_shortlist.md):
  * segment/word timestamps depend on server version; when absent, fusion
    falls back to interpolated timing, which degrades speaker attribution.
  * the realtime (streaming) mode is not exercised by this adapter; it is
    assessed separately in the streaming simulation notes.

Config keys:
  endpoint:  default "http://127.0.0.1:8000/v1"
  model:     served model name, default "mistralai/Voxtral-Mini-3B-2507"
"""

from __future__ import annotations

from typing import Optional

from ..schemas import ASRResult, ASRSegment, Recording, Word
from .base import AdapterUnavailable, ASRAdapter


class VoxtralClientAdapter(ASRAdapter):
    def _load(self) -> None:
        try:
            import requests  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise AdapterUnavailable("`requests` not installed") from e
        import requests

        self._requests = requests
        self.endpoint = self.config.get("endpoint", "http://127.0.0.1:8000/v1").rstrip("/")
        self.model = self.config.get("model", "mistralai/Voxtral-Mini-3B-2507")
        try:
            r = requests.get(f"{self.endpoint}/models", timeout=5)
            r.raise_for_status()
        except Exception as e:
            raise AdapterUnavailable(
                f"No vLLM server reachable at {self.endpoint}. Launch it first, e.g.\n"
                f"  vllm serve {self.model} --tokenizer-mode mistral "
                f"--config-format mistral --load-format mistral"
            ) from e

    def transcribe(self, recording: Recording, language: Optional[str] = None) -> ASRResult:
        with open(recording.audio_path, "rb") as f:
            resp = self._requests.post(
                f"{self.endpoint}/audio/transcriptions",
                files={"file": (recording.recording_id + ".wav", f, "audio/wav")},
                data={
                    "model": self.model,
                    **({"language": language} if language else {}),
                    "response_format": "verbose_json",
                },
                timeout=int(self.config.get("timeout_sec", 1800)),
            )
        resp.raise_for_status()
        data = resp.json()

        segments: list[ASRSegment] = []
        for seg in data.get("segments") or []:
            words = [
                Word(w.get("word", "").strip(), w.get("start"), w.get("end"))
                for w in seg.get("words") or []
            ]
            segments.append(ASRSegment(
                text=(seg.get("text") or "").strip(),
                start=seg.get("start"), end=seg.get("end"), words=words,
            ))
        text = (data.get("text") or "").strip()
        if not segments and text:
            segments = [ASRSegment(text=text)]
        return ASRResult(
            recording_id=recording.recording_id,
            model_id=self.model_id,
            text=text or " ".join(s.text for s in segments),
            segments=segments,
            language_requested=language,
            language_detected=data.get("language"),
            model_meta=self.model_meta(),
        )
