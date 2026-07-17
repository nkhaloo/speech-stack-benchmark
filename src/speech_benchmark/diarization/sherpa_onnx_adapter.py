"""sherpa-onnx offline speaker diarization (CPU track).

Runs the pyannote segmentation-3.0 model (ONNX export, MIT) plus a
3D-Speaker embedding extractor (Apache-2.0) with clustering — all via the
Apache-2.0 sherpa-onnx runtime. No torch dependency; light enough for
desktop CPUs, works on macOS / Windows / Linux.

Config keys:
  segmentation_model: path to sherpa-onnx-pyannote-segmentation-3-0 model.onnx
  embedding_model:    path to speaker embedding .onnx
  num_speakers:       optional fixed speaker count (else threshold clustering)
  cluster_threshold:  default 0.5
  num_threads:        default 4
"""

from __future__ import annotations

from pathlib import Path

from ..audio import load_audio
from ..asr.base import AdapterUnavailable
from ..schemas import DiarizationResult, Recording, SpeakerTurn
from .base import DiarizationAdapter


class SherpaOnnxDiarizationAdapter(DiarizationAdapter):
    def _load(self) -> None:
        try:
            import sherpa_onnx
        except ImportError as e:  # pragma: no cover
            raise AdapterUnavailable("sherpa-onnx not installed (install [cpu] extra)") from e

        seg = Path(self.config["segmentation_model"])
        emb = Path(self.config["embedding_model"])
        for p in (seg, emb):
            if not p.exists():
                raise AdapterUnavailable(
                    f"model file missing: {p} (run scripts/download_models.py --track cpu)"
                )
        cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(seg)
                ),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(emb),
                num_threads=int(self.config.get("num_threads", 4)),
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=int(self.config.get("num_speakers", -1) or -1),
                threshold=float(self.config.get("cluster_threshold", 0.5)),
            ),
        )
        if not cfg.validate():
            raise AdapterUnavailable("invalid sherpa-onnx diarization config")
        self._sd = __import__("sherpa_onnx").OfflineSpeakerDiarization(cfg)

    def _unload(self) -> None:
        del self._sd

    def diarize(self, recording: Recording) -> DiarizationResult:
        audio, sr = load_audio(recording.audio_path, self._sd.sample_rate)
        result = self._sd.process(audio).sort_by_start_time()
        turns = [
            SpeakerTurn(f"SPK{seg.speaker:02d}", float(seg.start), float(seg.end))
            for seg in result
        ]
        return DiarizationResult(
            recording_id=recording.recording_id,
            model_id=self.model_id,
            turns=turns,
            num_speakers=len({t.speaker for t in turns}),
            model_meta=self.model_meta(),
        )
