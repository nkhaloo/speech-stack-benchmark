"""Fusion: combine cached ASR output with cached diarization output into a
sentence-level, speaker-attributed transcript (the product's target shape —
comparable to ElevenLabs-Scribe-style speaker-labeled utterances).

Steps:
  1. Take word-level ASR output (interpolated when word timing is missing).
  2. Assign each word to the diarization turn with maximal temporal overlap.
  3. Split words into sentences on final punctuation and/or silence gaps.
  4. Give each sentence the speaker owning the largest share of word time.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

from ..schemas import (ASRResult, DiarizationResult, SpeakerTurn, Word,
                       utc_now_iso)

SENT_FINAL = set(".?!。！？؟…")


@dataclass
class CombinedSentence:
    text: str
    start: Optional[float]
    end: Optional[float]
    speaker: Optional[str]
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text, "start": self.start, "end": self.end,
            "speaker": self.speaker, "words": [w.to_dict() for w in self.words],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CombinedSentence":
        return cls(text=d["text"], start=d.get("start"), end=d.get("end"),
                   speaker=d.get("speaker"),
                   words=[Word.from_dict(w) for w in d.get("words", [])])


@dataclass
class CombinedResult:
    recording_id: str
    asr_model_id: str
    diarization_model_id: str
    sentences: list[CombinedSentence] = field(default_factory=list)
    num_speakers: Optional[int] = None
    status: str = "completed"
    error: Optional[str] = None
    created_at: str = dataclasses.field(default_factory=utc_now_iso)

    def to_dict(self) -> dict:
        return {
            "schema": "combined_result/v1",
            "recording_id": self.recording_id,
            "asr_model_id": self.asr_model_id,
            "diarization_model_id": self.diarization_model_id,
            "sentences": [s.to_dict() for s in self.sentences],
            "num_speakers": self.num_speakers,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CombinedResult":
        return cls(
            recording_id=d["recording_id"], asr_model_id=d["asr_model_id"],
            diarization_model_id=d["diarization_model_id"],
            sentences=[CombinedSentence.from_dict(s) for s in d.get("sentences", [])],
            num_speakers=d.get("num_speakers"),
            status=d.get("status", "completed"), error=d.get("error"),
            created_at=d.get("created_at", utc_now_iso()),
        )


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_word_speaker(word: Word, turns: list[SpeakerTurn],
                        tolerance_sec: float = 2.0) -> Optional[str]:
    """Speaker of the turn with maximal overlap with the word interval;
    falls back to the nearest turn within ``tolerance_sec``."""
    if word.start is None or word.end is None or not turns:
        return None
    best, best_ov = None, 0.0
    for t in turns:
        ov = _overlap(word.start, word.end, t.start, t.end)
        if ov > best_ov:
            best, best_ov = t.speaker, ov
    if best is not None:
        return best
    mid = (word.start + word.end) / 2
    nearest = min(turns, key=lambda t: min(abs(t.start - mid), abs(t.end - mid)))
    dist = min(abs(nearest.start - mid), abs(nearest.end - mid))
    return nearest.speaker if dist <= tolerance_sec else None


def split_sentences(words: list[Word], gap_threshold_sec: float = 0.8,
                    max_sentence_sec: float = 30.0,
                    split_on_speaker_change: bool = True) -> list[list[Word]]:
    """Sentence boundaries after final punctuation, on long silence gaps, on
    diarized speaker changes, or when a sentence exceeds ``max_sentence_sec``.
    Speaker-change splitting matches how the product would render output —
    one speaker per emitted sentence."""
    sentences: list[list[Word]] = []
    cur: list[Word] = []
    for i, w in enumerate(words):
        cur.append(w)
        boundary = False
        if w.text and w.text[-1] in SENT_FINAL:
            boundary = True
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt and w.end is not None and nxt.start is not None \
                and nxt.start - w.end > gap_threshold_sec:
            boundary = True
        if split_on_speaker_change and nxt is not None \
                and w.speaker is not None and nxt.speaker is not None \
                and nxt.speaker != w.speaker:
            boundary = True
        if cur[0].start is not None and w.end is not None \
                and w.end - cur[0].start > max_sentence_sec:
            boundary = True
        if boundary:
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)
    return sentences


def _sentence_speaker(words: list[Word]) -> Optional[str]:
    weight: dict[str, float] = {}
    for w in words:
        if w.speaker is None:
            continue
        dur = (w.end - w.start) if (w.start is not None and w.end is not None) else 0.0
        weight[w.speaker] = weight.get(w.speaker, 0.0) + max(dur, 1e-3)
    if not weight:
        return None
    return max(weight, key=weight.get)


def fuse(asr: ASRResult, diar: DiarizationResult,
         gap_threshold_sec: float = 0.8) -> CombinedResult:
    words = [dataclasses.replace(w) for w in asr.words()]
    for w in words:
        w.speaker = assign_word_speaker(w, diar.turns)

    sentences = []
    for group in split_sentences(words, gap_threshold_sec=gap_threshold_sec):
        text = " ".join(w.text for w in group).strip()
        if not text:
            continue
        sentences.append(CombinedSentence(
            text=text,
            start=group[0].start, end=group[-1].end,
            speaker=_sentence_speaker(group),
            words=group,
        ))
    return CombinedResult(
        recording_id=asr.recording_id,
        asr_model_id=asr.model_id,
        diarization_model_id=diar.model_id,
        sentences=sentences,
        num_speakers=diar.num_speakers,
    )
