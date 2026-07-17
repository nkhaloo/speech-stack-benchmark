"""Canonical output schemas shared by every ASR and diarization adapter.

All adapters normalize their native output into these dataclasses so that
cached results from different systems can be fused and scored consistently.
Serialization is plain JSON via ``to_dict``/``from_dict``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Atomic JSON I/O (interrupted writes must never look like completed results)
# ---------------------------------------------------------------------------

def atomic_write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

@dataclass
class ResourceStats:
    peak_ram_mb: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    gpu_name: Optional[str] = None
    gpu_index: Optional[int] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["ResourceStats"]:
        return cls(**d) if d else None


@dataclass
class Word:
    text: str
    start: Optional[float] = None
    end: Optional[float] = None
    confidence: Optional[float] = None
    speaker: Optional[str] = None  # filled in by fusion

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Word":
        return cls(**d)


@dataclass
class ASRSegment:
    text: str
    start: Optional[float] = None
    end: Optional[float] = None
    confidence: Optional[float] = None
    words: list[Word] = field(default_factory=list)
    speaker: Optional[str] = None  # filled in by fusion

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ASRSegment":
        words = [Word.from_dict(w) for w in d.get("words") or []]
        return cls(
            text=d["text"], start=d.get("start"), end=d.get("end"),
            confidence=d.get("confidence"), words=words, speaker=d.get("speaker"),
        )


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------

@dataclass
class ASRResult:
    recording_id: str
    model_id: str
    text: str = ""
    segments: list[ASRSegment] = field(default_factory=list)
    language_requested: Optional[str] = None
    language_detected: Optional[str] = None
    language_probability: Optional[float] = None
    runtime_sec: Optional[float] = None
    load_time_sec: Optional[float] = None
    audio_duration_sec: Optional[float] = None
    resources: Optional[ResourceStats] = None
    model_meta: dict = field(default_factory=dict)
    streaming_meta: dict = field(default_factory=dict)  # time-to-first-text etc.
    status: str = "completed"  # pending|running|completed|failed|skipped|interrupted
    error: Optional[str] = None
    created_at: str = field(default_factory=utc_now_iso)

    @property
    def real_time_factor(self) -> Optional[float]:
        if self.runtime_sec and self.audio_duration_sec:
            return self.runtime_sec / self.audio_duration_sec
        return None

    def words(self) -> list[Word]:
        """All words, falling back to whitespace-split segments with
        linearly interpolated timestamps when word timing is unavailable."""
        out: list[Word] = []
        for seg in self.segments:
            if seg.words:
                out.extend(seg.words)
            else:
                tokens = seg.text.split()
                if not tokens:
                    continue
                if seg.start is not None and seg.end is not None and seg.end > seg.start:
                    dur = (seg.end - seg.start) / len(tokens)
                    for i, tok in enumerate(tokens):
                        out.append(Word(tok, seg.start + i * dur, seg.start + (i + 1) * dur))
                else:
                    out.extend(Word(tok) for tok in tokens)
        return out

    def to_dict(self) -> dict:
        return {
            "schema": "asr_result/v1",
            "recording_id": self.recording_id,
            "model_id": self.model_id,
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "language_requested": self.language_requested,
            "language_detected": self.language_detected,
            "language_probability": self.language_probability,
            "runtime_sec": self.runtime_sec,
            "load_time_sec": self.load_time_sec,
            "audio_duration_sec": self.audio_duration_sec,
            "real_time_factor": self.real_time_factor,
            "resources": self.resources.to_dict() if self.resources else None,
            "model_meta": self.model_meta,
            "streaming_meta": self.streaming_meta,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ASRResult":
        return cls(
            recording_id=d["recording_id"], model_id=d["model_id"],
            text=d.get("text", ""),
            segments=[ASRSegment.from_dict(s) for s in d.get("segments", [])],
            language_requested=d.get("language_requested"),
            language_detected=d.get("language_detected"),
            language_probability=d.get("language_probability"),
            runtime_sec=d.get("runtime_sec"), load_time_sec=d.get("load_time_sec"),
            audio_duration_sec=d.get("audio_duration_sec"),
            resources=ResourceStats.from_dict(d.get("resources")),
            model_meta=d.get("model_meta", {}), streaming_meta=d.get("streaming_meta", {}),
            status=d.get("status", "completed"), error=d.get("error"),
            created_at=d.get("created_at", utc_now_iso()),
        )


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------

@dataclass
class SpeakerTurn:
    speaker: str
    start: float
    end: float
    confidence: Optional[float] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SpeakerTurn":
        return cls(**d)


@dataclass
class DiarizationResult:
    recording_id: str
    model_id: str
    turns: list[SpeakerTurn] = field(default_factory=list)
    num_speakers: Optional[int] = None
    runtime_sec: Optional[float] = None
    load_time_sec: Optional[float] = None
    audio_duration_sec: Optional[float] = None
    resources: Optional[ResourceStats] = None
    model_meta: dict = field(default_factory=dict)
    status: str = "completed"
    error: Optional[str] = None
    created_at: str = field(default_factory=utc_now_iso)

    @property
    def real_time_factor(self) -> Optional[float]:
        if self.runtime_sec and self.audio_duration_sec:
            return self.runtime_sec / self.audio_duration_sec
        return None

    def speaker_labels(self) -> list[str]:
        seen: dict[str, None] = {}
        for t in self.turns:
            seen.setdefault(t.speaker)
        return list(seen)

    def to_dict(self) -> dict:
        return {
            "schema": "diarization_result/v1",
            "recording_id": self.recording_id,
            "model_id": self.model_id,
            "turns": [t.to_dict() for t in self.turns],
            "num_speakers": self.num_speakers,
            "runtime_sec": self.runtime_sec,
            "load_time_sec": self.load_time_sec,
            "audio_duration_sec": self.audio_duration_sec,
            "real_time_factor": self.real_time_factor,
            "resources": self.resources.to_dict() if self.resources else None,
            "model_meta": self.model_meta,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DiarizationResult":
        return cls(
            recording_id=d["recording_id"], model_id=d["model_id"],
            turns=[SpeakerTurn.from_dict(t) for t in d.get("turns", [])],
            num_speakers=d.get("num_speakers"),
            runtime_sec=d.get("runtime_sec"), load_time_sec=d.get("load_time_sec"),
            audio_duration_sec=d.get("audio_duration_sec"),
            resources=ResourceStats.from_dict(d.get("resources")),
            model_meta=d.get("model_meta", {}),
            status=d.get("status", "completed"), error=d.get("error"),
            created_at=d.get("created_at", utc_now_iso()),
        )

    def to_rttm(self, uri: Optional[str] = None) -> str:
        uri = uri or self.recording_id
        lines = []
        for t in self.turns:
            lines.append(
                f"SPEAKER {uri} 1 {t.start:.3f} {t.duration:.3f} "
                f"<NA> <NA> {t.speaker} <NA> <NA>"
            )
        return "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------------
# Reference (ground truth) and recordings
# ---------------------------------------------------------------------------

@dataclass
class ReferenceTurn:
    speaker: str
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReferenceTurn":
        return cls(speaker=d["speaker"], start=d["start"], end=d["end"], text=d["text"])


@dataclass
class Reference:
    recording_id: str
    language: str
    turns: list[ReferenceTurn] = field(default_factory=list)
    num_speakers: Optional[int] = None
    source_meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.turns if t.text)

    def to_dict(self) -> dict:
        return {
            "schema": "reference/v1",
            "recording_id": self.recording_id,
            "language": self.language,
            "turns": [t.to_dict() for t in self.turns],
            "num_speakers": self.num_speakers,
            "source_meta": self.source_meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Reference":
        return cls(
            recording_id=d["recording_id"], language=d["language"],
            turns=[ReferenceTurn.from_dict(t) for t in d.get("turns", [])],
            num_speakers=d.get("num_speakers"), source_meta=d.get("source_meta", {}),
        )


@dataclass
class Recording:
    recording_id: str
    dataset: str
    language: str
    audio_path: str
    reference_path: Optional[str] = None
    duration_sec: Optional[float] = None
    num_speakers: Optional[int] = None
    profile: Optional[str] = None

    def load_reference(self) -> Optional[Reference]:
        if self.reference_path and Path(self.reference_path).exists():
            return Reference.from_dict(load_json(self.reference_path))
        return None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Recording":
        return cls(**{k: d.get(k) for k in (
            "recording_id", "dataset", "language", "audio_path", "reference_path",
            "duration_sec", "num_speakers", "profile")})


def load_manifest(path: str | Path) -> list[Recording]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(Recording.from_dict(json.loads(line)))
    return recs


def save_manifest(path: str | Path, recordings: list[Recording]) -> None:
    text = "".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in recordings)
    atomic_write_text(path, text)
