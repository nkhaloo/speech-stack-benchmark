"""Deterministic dummy streaming stack for plumbing tests and smoke runs.

Reads the recording's reference and re-emits each reference turn as a
speaker-attributed sentence, *incrementally* under the runner's clock:

  * a sentence finalizes ``latency_sec`` after its audio ends (so finalization
    delay and time-to-first-token are non-trivial and predictable);
  * text is corrupted at a seeded ``wer_target`` rate (sub/del/ins), giving
    known-magnitude WER/cpWER to sanity-check the streaming metric code;
  * speakers are relabeled and confused at a seeded ``confusion_rate``;
  * a seeded fraction of sentences emit a rougher **partial** first, then a
    corrected **final** — exercising the revision / stability metrics.

Zero downloads; never used in real comparisons (tagged ``family: dummy``).
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from ..schemas import Emission, Recording
from .base import StreamingAdapter


class DummyStreamingAdapter(StreamingAdapter):
    def _load(self) -> None:  # nothing to load
        pass

    def reset(self, recording: Recording) -> None:
        ref = recording.load_reference()
        if ref is None:
            raise ValueError(
                f"DummyStreamingAdapter needs a reference for {recording.recording_id}"
            )
        self._wer = float(self.config.get("wer_target", 0.1))
        self._confusion = float(self.config.get("confusion_rate", 0.05))
        self._latency = float(self.config.get("latency_sec", 0.6))
        self._revision_rate = float(self.config.get("revision_rate", 0.3))
        rng = random.Random(f"{self.model_id}:{recording.recording_id}")

        turns = sorted(ref.turns, key=lambda t: t.start)
        speakers = sorted({t.speaker for t in turns})
        label_map = {s: f"SPK{i:02d}" for i, s in enumerate(speakers)}

        self._plan: list[dict] = []
        for i, t in enumerate(turns):
            spk = t.speaker
            if len(speakers) > 1 and rng.random() < self._confusion:
                spk = rng.choice([s for s in speakers if s != t.speaker])
            final_text = _corrupt(t.text, self._wer, rng)
            has_partial = rng.random() < self._revision_rate
            partial_text = _corrupt(t.text, min(1.0, self._wer * 2.5), rng) \
                if has_partial else final_text
            self._plan.append({
                "id": i, "start": t.start, "end": t.end,
                "speaker": label_map[spk],
                "partial_text": partial_text, "final_text": final_text,
                "has_partial": has_partial,
                "partial_done": False, "final_done": False,
            })

    def push(self, audio: np.ndarray, audio_time_end: float) -> list[Emission]:
        out: list[Emission] = []
        for p in self._plan:
            if not p["final_done"] and p["has_partial"] and not p["partial_done"] \
                    and audio_time_end >= p["end"]:
                p["partial_done"] = True
                out.append(Emission(
                    sentence_id=p["id"], text=p["partial_text"], speaker=p["speaker"],
                    start=p["start"], end=p["end"], audio_time=audio_time_end,
                    is_final=False, revision=0))
            if not p["final_done"] and audio_time_end >= p["end"] + self._latency:
                p["final_done"] = True
                out.append(Emission(
                    sentence_id=p["id"], text=p["final_text"], speaker=p["speaker"],
                    start=p["start"], end=p["end"], audio_time=audio_time_end,
                    is_final=True, revision=1 if p["partial_done"] else 0))
        return out

    def flush(self) -> list[Emission]:
        out: list[Emission] = []
        for p in self._plan:
            if not p["final_done"]:
                p["final_done"] = True
                out.append(Emission(
                    sentence_id=p["id"], text=p["final_text"], speaker=p["speaker"],
                    start=p["start"], end=p["end"], audio_time=p["end"],
                    is_final=True, revision=1 if p["partial_done"] else 0))
        return out


def _corrupt(text: str, rate: float, rng: random.Random) -> str:
    out: list[str] = []
    for tok in text.split():
        r = rng.random()
        if r < rate * 0.5:
            out.append(f"sub{rng.randint(0, 999)}")     # substitution
        elif r < rate * 0.75:
            continue                                     # deletion
        elif r < rate:
            out.extend([tok, f"ins{rng.randint(0, 999)}"])  # insertion
        else:
            out.append(tok)
    return " ".join(out)
