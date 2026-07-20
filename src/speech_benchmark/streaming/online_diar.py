"""Reusable diart online-diarization component, shared by the diart_whisper and
voxtral_realtime stacks (both pair streaming ASR/windowed ASR with the same
online diarizer).

Wraps diart's ``SpeakerDiarization`` block API: audio is fed one step-sized 5 s
window at a time (exactly as diart would run live), online clustering keeps
speaker IDs stable, and per-step aggregated annotations are accumulated into a
global set of speaker turns. Requires ``diart`` + gated pyannote weights; must be
verified on the lab machine (needs weights + GPU).
"""

from __future__ import annotations

import os

import numpy as np

from ..schemas import SpeakerTurn
from .base import AdapterUnavailable


class OnlineDiarizer:
    def __init__(self, diart_config: dict):
        self.cfg = diart_config or {}

    def load(self) -> None:
        try:
            import torch
            from diart import SpeakerDiarization, SpeakerDiarizationConfig
            from diart.models import EmbeddingModel, SegmentationModel
        except ImportError as e:
            raise AdapterUnavailable(
                "diart not installed. Run scripts/setup_diart.sh to create "
                ".venv-diart (diart + gated pyannote weights, HF_TOKEN), then "
                "re-run from that env."
            ) from e

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or True
        try:
            seg = SegmentationModel.from_pyannote(
                self.cfg.get("segmentation", "pyannote/segmentation"), use_hf_token=token)
            emb = EmbeddingModel.from_pyannote(
                self.cfg.get("embedding", "pyannote/embedding"), use_hf_token=token)
        except Exception as e:
            raise AdapterUnavailable(
                f"Could not load diart pyannote models ({e}). Accept the terms on "
                "huggingface.co for pyannote/segmentation and pyannote/embedding "
                "and set HF_TOKEN.") from e

        device = self.cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        config = SpeakerDiarizationConfig(
            segmentation=seg, embedding=emb,
            duration=float(self.cfg.get("duration", 5.0)),
            step=float(self.cfg.get("step", 0.5)),
            latency=self.cfg.get("latency", "min"),
            tau_active=float(self.cfg.get("tau_active", 0.5)),
            rho_update=float(self.cfg.get("rho_update", 0.1)),
            delta_new=float(self.cfg.get("delta_new", 0.57)),
            device=torch.device(device),
        )
        self._pipeline = SpeakerDiarization(config)
        self.duration = config.duration
        self.step = config.step
        self.sample_rate = config.sample_rate

    def reset(self) -> None:
        self._pipeline.reset()
        self._fed_steps = 0
        self._global: dict[str, list[list[float]]] = {}

    def feed(self, buf: np.ndarray) -> None:
        """Feed every newly-available step-sized 5 s window from ``buf`` (the full
        audio buffer so far), advancing internal state."""
        from pyannote.core import SlidingWindow, SlidingWindowFeature

        sr = self.sample_rate
        dur_samples = int(round(self.duration * sr))
        step_samples = int(round(self.step * sr))
        res = 1.0 / sr
        while (self._fed_steps * step_samples) + dur_samples <= len(buf):
            start = self._fed_steps * step_samples
            win = buf[start:start + dur_samples].reshape(-1, 1).astype(np.float32)
            swf = SlidingWindowFeature(
                win, SlidingWindow(start=self._fed_steps * self.step,
                                   duration=res, step=res))
            for annotation, _ in self._pipeline([swf]):
                for segment, _track, label in annotation.itertracks(yield_label=True):
                    self._accumulate(str(label), float(segment.start), float(segment.end))
            self._fed_steps += 1

    def _accumulate(self, speaker: str, start: float, end: float) -> None:
        spans = self._global.setdefault(speaker, [])
        for span in spans:
            if start <= span[1] + 0.25 and end >= span[0] - 0.25:
                span[0] = min(span[0], start)
                span[1] = max(span[1], end)
                return
        spans.append([start, end])

    def current_turns(self) -> list[SpeakerTurn]:
        turns = [SpeakerTurn(spk, s, e)
                 for spk, spans in self._global.items()
                 for s, e in spans if e > s]
        return sorted(turns, key=lambda t: t.start)
