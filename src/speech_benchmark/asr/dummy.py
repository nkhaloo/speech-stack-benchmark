"""Deterministic dummy ASR adapter for plumbing tests and smoke runs.

It reads the recording's reference transcript and re-emits it with a
configurable, seeded corruption rate (substitutions / deletions / insertions)
and interpolated word timestamps. This exercises the full pipeline —
caching, fusion, metrics, reporting — with zero model downloads, and the
injected error rates give known-order-of-magnitude WER values to sanity-check
the metric code against.

Never use dummy results in real comparisons; the runner tags them
``family: dummy`` and reports exclude them from leaderboards by default.
"""

from __future__ import annotations

import random
from typing import Optional

from ..schemas import ASRResult, ASRSegment, Recording, Word
from .base import ASRAdapter


class DummyASRAdapter(ASRAdapter):
    def transcribe(self, recording: Recording, language: Optional[str] = None) -> ASRResult:
        ref = recording.load_reference()
        if ref is None:
            raise ValueError(
                f"DummyASRAdapter needs a reference file for {recording.recording_id}"
            )
        wer_target = float(self.config.get("wer_target", 0.1))
        rng = random.Random(f"{self.model_id}:{recording.recording_id}")

        segments: list[ASRSegment] = []
        for turn in ref.turns:
            tokens = turn.text.split()
            out_tokens: list[str] = []
            for tok in tokens:
                r = rng.random()
                if r < wer_target * 0.5:
                    out_tokens.append(f"sub{rng.randint(0, 999)}")  # substitution
                elif r < wer_target * 0.75:
                    continue  # deletion
                elif r < wer_target:
                    out_tokens.extend([tok, f"ins{rng.randint(0, 999)}"])  # insertion
                else:
                    out_tokens.append(tok)
            if not out_tokens:
                continue
            dur = (turn.end - turn.start) / len(out_tokens)
            words = [
                Word(t, turn.start + i * dur, turn.start + (i + 1) * dur, confidence=0.9)
                for i, t in enumerate(out_tokens)
            ]
            segments.append(ASRSegment(
                text=" ".join(out_tokens), start=turn.start, end=turn.end, words=words,
            ))

        return ASRResult(
            recording_id=recording.recording_id,
            model_id=self.model_id,
            text=" ".join(s.text for s in segments),
            segments=segments,
            language_requested=language,
            language_detected=ref.language,
            language_probability=1.0,
            model_meta=self.model_meta(),
        )
