"""Deterministic dummy diarizer for plumbing tests and smoke runs.

Perturbs the reference speaker turns with seeded boundary jitter and a
configurable speaker-confusion rate, so DER / attribution metrics get
non-trivial but predictable values without any model download.
"""

from __future__ import annotations

import random

from ..schemas import DiarizationResult, Recording, SpeakerTurn
from .base import DiarizationAdapter


class DummyDiarizationAdapter(DiarizationAdapter):
    def diarize(self, recording: Recording) -> DiarizationResult:
        ref = recording.load_reference()
        if ref is None:
            raise ValueError(
                f"DummyDiarizationAdapter needs a reference for {recording.recording_id}"
            )
        jitter = float(self.config.get("boundary_jitter_sec", 0.2))
        confusion = float(self.config.get("confusion_rate", 0.05))
        rng = random.Random(f"{self.model_id}:{recording.recording_id}")

        speakers = sorted({t.speaker for t in ref.turns})
        label_map = {s: f"SPK{i:02d}" for i, s in enumerate(speakers)}

        turns: list[SpeakerTurn] = []
        for t in ref.turns:
            start = max(0.0, t.start + rng.uniform(-jitter, jitter))
            end = max(start + 0.1, t.end + rng.uniform(-jitter, jitter))
            spk = t.speaker
            if len(speakers) > 1 and rng.random() < confusion:
                spk = rng.choice([s for s in speakers if s != t.speaker])
            turns.append(SpeakerTurn(label_map[spk], start, end, confidence=0.9))

        return DiarizationResult(
            recording_id=recording.recording_id,
            model_id=self.model_id,
            turns=turns,
            num_speakers=len(speakers),
            model_meta=self.model_meta(),
        )
