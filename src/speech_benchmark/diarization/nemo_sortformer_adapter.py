"""NVIDIA NeMo Sortformer diarization adapter — REFERENCE ONLY.

IMPORTANT LICENSING FLAG: the released weights (nvidia/diar_sortformer_4spk-v1)
are CC-BY-NC-4.0 — **non-commercial**. This adapter exists only so accuracy
can be compared against commercially usable candidates; it is disabled by
default in every track config and must never be part of the recommended
product stack. See docs/licensing.md.

NeMo is dependency-heavy; if you enable this, install it in a dedicated
environment (.venv-nemo) and run the benchmark for this model from there.
"""

from __future__ import annotations

from ..asr.base import AdapterUnavailable
from ..audio import load_audio, write_wav
from ..schemas import DiarizationResult, Recording, SpeakerTurn
from .base import DiarizationAdapter


class NemoSortformerAdapter(DiarizationAdapter):
    def _load(self) -> None:
        if not self.config.get("acknowledge_non_commercial", False):
            raise AdapterUnavailable(
                "Sortformer weights are CC-BY-NC (non-commercial). Set "
                "`acknowledge_non_commercial: true` in the model config to run it "
                "for reference comparison only."
            )
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel
        except ImportError as e:  # pragma: no cover
            raise AdapterUnavailable(
                "NeMo not installed. Create the dedicated env: "
                "scripts/setup_linux_gpu.sh --with-nemo"
            ) from e
        self._model = SortformerEncLabelModel.from_pretrained(
            self.config.get("model", "nvidia/diar_sortformer_4spk-v1")
        )
        self._model.eval()

    def _unload(self) -> None:
        del self._model

    def diarize(self, recording: Recording) -> DiarizationResult:
        import tempfile
        from pathlib import Path

        # NeMo expects 16 kHz mono files.
        audio, sr = load_audio(recording.audio_path, 16000)
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "in.wav"
            write_wav(wav, audio, sr)
            predicted = self._model.diarize(audio=[str(wav)], batch_size=1)
        turns = []
        for line in predicted[0]:
            start_s, end_s, spk = line.split()
            turns.append(SpeakerTurn(str(spk), float(start_s), float(end_s)))
        return DiarizationResult(
            recording_id=recording.recording_id,
            model_id=self.model_id,
            turns=turns,
            num_speakers=len({t.speaker for t in turns}),
            model_meta=self.model_meta(),
        )
