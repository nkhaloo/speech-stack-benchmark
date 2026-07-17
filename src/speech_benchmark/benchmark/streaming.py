"""Sentence-level chunked-processing simulation.

The eventual product processes audio at roughly sentence/utterance level with
a small acceptable delay. Most shortlisted models are *batch* models, so we
simulate near-real-time use by re-running them over a moving window
("batch-over-moving-window") in simulated real time, and measure:

  * time_to_first_text_sec — simulated wall time from the moment audio starts
    until the first non-empty partial text is available;
  * sentence_finalization_delay_sec — mean lag between a sentence's audio end
    and the simulated wall-clock moment its text stabilized (unchanged for
    ``stabilize_steps`` consecutive steps);
  * revisions — number of times a previously emitted sentence's text changed;
  * chunked_rtf — total processing time / audio duration in chunked mode.

Simulated wall clock: audio is assumed to arrive in real time, so step ``k``'s
audio is fully available at ``(k+1)*hop_sec``; processing for that step then
takes the measured inference time. If inference is slower than the hop, delay
accumulates — exactly as it would live.

True streaming models (e.g. Voxtral Mini Realtime) should additionally be
assessed with their native streaming interface; results from this simulation
are labeled ``mode: "simulated_chunked"`` so the two are never conflated.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from ..audio import load_audio, write_wav
from ..asr.base import ASRAdapter
from ..schemas import Recording


@dataclass
class ChunkedConfig:
    window_sec: float = 20.0   # context window fed to the model
    hop_sec: float = 5.0       # new audio per step
    stabilize_steps: int = 2   # steps a sentence must stay unchanged to finalize


def run_chunked_simulation(adapter: ASRAdapter, recording: Recording,
                           cfg: ChunkedConfig, language: str | None = None) -> dict:
    audio, sr = load_audio(recording.audio_path, 16000)
    total_sec = len(audio) / sr

    sim_clock = 0.0            # simulated wall time
    step_time_total = 0.0
    first_text_at: float | None = None
    revisions = 0
    # bucket -> {"text", "stable_steps", "audio_end", "finalized_at"}
    tracked: dict[int, dict] = {}
    finalization_delays: list[float] = []

    n_steps = max(1, -(-int(total_sec * 1000) // int(cfg.hop_sec * 1000)))
    with tempfile.TemporaryDirectory() as td:
        for step in range(n_steps):
            fed_until = min(total_sec, (step + 1) * cfg.hop_sec)
            win_start = max(0.0, fed_until - cfg.window_sec)
            chunk = audio[int(win_start * sr):int(fed_until * sr)]
            wav = Path(td) / "chunk.wav"
            write_wav(wav, chunk, sr)
            fake = Recording(
                recording_id=f"{recording.recording_id}__chunk{step}",
                dataset=recording.dataset, language=recording.language,
                audio_path=str(wav), reference_path=recording.reference_path,
            )
            t0 = time.perf_counter()
            result = adapter.transcribe(fake, language=language)
            infer_sec = time.perf_counter() - t0
            step_time_total += infer_sec
            # audio for this step is complete at `fed_until`; results exist
            # after inference. If inference lags behind real time, carry over.
            sim_clock = max(sim_clock, fed_until) + infer_sec

            if result.text.strip() and first_text_at is None:
                # first text is available once this step's processing is done;
                # measured from the start of the audio stream
                first_text_at = sim_clock - 0.0

            for seg in result.segments:
                audio_end = (seg.end or 0.0) + win_start
                bucket = int(audio_end // 2.0)  # coarse identity across steps
                entry = tracked.get(bucket)
                if entry is None:
                    tracked[bucket] = {"text": seg.text, "stable_steps": 0,
                                       "audio_end": audio_end, "finalized_at": None}
                elif entry["text"] != seg.text:
                    if entry["finalized_at"] is not None:
                        pass  # already finalized; late change counts as revision
                    revisions += 1
                    entry.update(text=seg.text, stable_steps=0, finalized_at=None,
                                 audio_end=audio_end)
                else:
                    entry["stable_steps"] += 1
                    if entry["stable_steps"] == cfg.stabilize_steps \
                            and entry["finalized_at"] is None:
                        entry["finalized_at"] = sim_clock
                        finalization_delays.append(
                            max(0.0, sim_clock - entry["audio_end"]))

    mean_final = (sum(finalization_delays) / len(finalization_delays)
                  if finalization_delays else None)
    return {
        "mode": "simulated_chunked",
        "window_sec": cfg.window_sec,
        "hop_sec": cfg.hop_sec,
        "stabilize_steps": cfg.stabilize_steps,
        "time_to_first_text_sec": first_text_at,
        "sentence_finalization_delay_sec": mean_final,
        "finalized_sentences": len(finalization_delays),
        "revisions": revisions,
        "chunked_rtf": step_time_total / total_sec if total_sec else None,
        "steps": n_steps,
    }
