"""Shared incremental-emission bookkeeping for streaming adapters.

Every windowing/streaming stack faces the same problem: each time it re-derives
sentences from the audio so far, it must decide which are *new*, which are
*revisions* of an already-emitted sentence, and which are old enough to
*finalize*. ``SentenceTracker`` centralizes that so the windowed, diart, and
Voxtral adapters share one implementation (and one set of revision/churn
semantics that the metrics depend on).

A "current sentence" is a dict ``{start, end, text, speaker}`` in absolute audio
time. Sentences are matched across updates by nearest start time within
``match_tolerance_sec``.
"""

from __future__ import annotations

from typing import Optional

from ..schemas import Emission


class SentenceTracker:
    def __init__(self, finalize_after_sec: float = 5.0,
                 match_tolerance_sec: float = 1.0):
        self.finalize_after = float(finalize_after_sec)
        self.match_tol = float(match_tolerance_sec)
        self._tracked: list[dict] = []
        self._next_id = 0

    def update(self, current: list[dict], audio_time_end: float,
               final: bool = False) -> list[Emission]:
        """Diff ``current`` sentences against tracked state; return new/revised
        and newly-finalized emissions. When ``final`` is True, every open
        sentence is finalized (end of stream)."""
        out: list[Emission] = []
        for cs in current:
            if not (cs.get("text") or "").strip():
                continue
            match = self._match(cs)
            if match is None:
                t = {"id": self._next_id, "start": cs["start"], "end": cs["end"],
                     "text": cs["text"], "speaker": cs.get("speaker"),
                     "final": False, "revision": 0}
                self._next_id += 1
                self._tracked.append(t)
                out.append(self._emit(t, audio_time_end, is_final=False))
            elif not match["final"] and (match["text"] != cs["text"]
                                         or match["speaker"] != cs.get("speaker")):
                match.update(text=cs["text"], speaker=cs.get("speaker"),
                             end=cs["end"], revision=match["revision"] + 1)
                out.append(self._emit(match, audio_time_end, is_final=False))

        for t in self._tracked:
            if not t["final"] and (final or audio_time_end - t["end"] > self.finalize_after):
                t["final"] = True
                out.append(self._emit(t, audio_time_end, is_final=True))
        return out

    def _match(self, cs: dict) -> Optional[dict]:
        best, best_d = None, self.match_tol
        for t in self._tracked:
            d = abs(t["start"] - cs["start"])
            if d <= best_d:
                best, best_d = t, d
        return best

    @staticmethod
    def _emit(t: dict, audio_time_end: float, is_final: bool) -> Emission:
        return Emission(sentence_id=t["id"], text=t["text"], speaker=t["speaker"],
                        start=t["start"], end=t["end"], audio_time=audio_time_end,
                        is_final=is_final, revision=t["revision"])
