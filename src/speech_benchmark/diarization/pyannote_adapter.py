"""pyannote.audio pipeline adapter (GPU primary; also runnable on CPU).

Works with:
  * pyannote/speaker-diarization-3.1   (MIT weights, gated; pyannote.audio 3.x)
  * pyannote/speaker-diarization-community-1
        (CC-BY-4.0 weights, gated; requires pyannote.audio >= 4.0 — use the
         dedicated .venv-pyannote4 environment, see setup_linux_gpu.sh)

Access is gated on Hugging Face: accept the model conditions once and export
HF_TOKEN. After ``scripts/download_models.py`` has cached the pipeline, it
loads fully offline (HF_HUB_OFFLINE=1).

Audio is passed in-memory ({"waveform", "sample_rate"}) so no torchaudio /
torchcodec decoding path is exercised.

Config keys:
  model:         HF pipeline id or local path
  device:        "cuda" | "cpu" (default cuda if available)
  num_speakers / min_speakers / max_speakers: optional clustering hints
"""

from __future__ import annotations

import os

from ..audio import load_audio
from ..asr.base import AdapterUnavailable
from ..schemas import DiarizationResult, Recording, SpeakerTurn
from .base import DiarizationAdapter


class PyannoteAdapter(DiarizationAdapter):
    def _load(self) -> None:
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as e:  # pragma: no cover
            raise AdapterUnavailable(
                "pyannote.audio not installed in this environment"
            ) from e

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self._pipeline = Pipeline.from_pretrained(
            self.config.get("model", "pyannote/speaker-diarization-3.1"),
            use_auth_token=token,
        )
        if self._pipeline is None:
            raise AdapterUnavailable(
                f"Could not load pipeline {self.config.get('model')!r}. Gated model: "
                "accept its conditions on huggingface.co and set HF_TOKEN."
            )
        device = self.config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline.to(torch.device(device))
        self._torch = torch

    def _unload(self) -> None:
        del self._pipeline

    def diarize(self, recording: Recording) -> DiarizationResult:
        audio, sr = load_audio(recording.audio_path, 16000)
        waveform = self._torch.from_numpy(audio).unsqueeze(0)
        kwargs = {}
        for k in ("num_speakers", "min_speakers", "max_speakers"):
            if self.config.get(k) is not None:
                kwargs[k] = int(self.config[k])
        annotation = self._pipeline({"waveform": waveform, "sample_rate": sr}, **kwargs)
        # pyannote.audio 4.x returns an object holding .speaker_diarization
        if hasattr(annotation, "speaker_diarization"):
            annotation = annotation.speaker_diarization

        turns = [
            SpeakerTurn(str(label), float(segment.start), float(segment.end))
            for segment, _, label in annotation.itertracks(yield_label=True)
        ]
        return DiarizationResult(
            recording_id=recording.recording_id,
            model_id=self.model_id,
            turns=turns,
            num_speakers=len({t.speaker for t in turns}),
            model_meta=self.model_meta(),
        )
