"""Deterministic synthetic-conversation builder.

Builds multi-speaker "conversations" by concatenating single-speaker
utterances (with silence gaps, no overlap) from a clip source. Because every
sample's transcript, speaker, and time span are known exactly, this yields
perfect ground truth for WER, DER, and speaker-attribution metrics in all
five languages under one permissive license (CC0 for Common Voice).

This is the reproducible core of the benchmark; real-audio anchor corpora
(AMI, AISHELL-4) complement it — see docs/datasets.md for the tradeoffs.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from ..audio import voiced_segments, write_wav
from ..schemas import (Recording, Reference, ReferenceTurn, atomic_write_json,
                       save_manifest)
from .sources import SR, get_source


def build_conversation(
    speaker_clips: dict[str, list],
    target_duration_sec: float,
    rng: random.Random,
    gap_range: tuple[float, float] = (0.4, 1.2),
    leading_silence: float = 0.5,
) -> tuple[np.ndarray, list[ReferenceTurn]]:
    """Concatenate clips from the given speakers until the target duration.

    Speaker order is a seeded random walk that avoids immediate repetition
    80% of the time (so consecutive same-speaker turns exist but are rare).
    Returns (audio at 16 kHz, reference turns with exact times).
    """
    speakers = sorted(speaker_clips)
    pools = {s: list(clips) for s, clips in speaker_clips.items()}
    for pool in pools.values():
        rng.shuffle(pool)

    audio_parts: list[np.ndarray] = [np.zeros(int(leading_silence * SR), dtype=np.float32)]
    turns: list[ReferenceTurn] = []
    t = leading_silence
    prev_speaker: str | None = None

    while t < target_duration_sec:
        candidates = [s for s in speakers if pools[s]]
        if not candidates:
            break
        if prev_speaker in candidates and len(candidates) > 1 and rng.random() < 0.8:
            candidates = [s for s in candidates if s != prev_speaker]
        spk = rng.choice(candidates)
        clip = pools[spk].pop()
        # Reference turns follow speech activity: drop edge silence and split on
        # silence *inside* the clip, so pauses are not labeled as speech (which
        # the diarizer would otherwise be charged with as "missed speech").
        raw = clip.audio()
        segs = voiced_segments(raw, SR)
        if not segs:
            continue
        a = int(round(segs[0][0] * SR))
        b = int(round(segs[-1][1] * SR))
        clip_audio = raw[a:b]  # trims leading/trailing silence; keeps inner pauses
        dur = len(clip_audio) / SR
        if dur < 0.2:
            continue
        clip_start = t
        offset = a / SR
        audio_parts.append(clip_audio)
        # One reference turn per voiced span; the clip's text goes on the first.
        for i, (s, e) in enumerate(segs):
            turns.append(ReferenceTurn(
                speaker=spk,
                start=round(clip_start + s - offset, 3),
                end=round(clip_start + e - offset, 3),
                text=clip.text if i == 0 else "",
            ))
        gap = rng.uniform(*gap_range)
        audio_parts.append(np.zeros(int(gap * SR), dtype=np.float32))
        t += dur + gap
        prev_speaker = spk

    return np.concatenate(audio_parts).astype(np.float32), turns


def prepare_synthetic_dataset(
    out_dir: str | Path,
    source_name: str,
    profile: str,
    languages: list[str],
    minutes_per_language: float,
    recording_minutes: float,
    speakers_range: tuple[int, int],
    seed: int,
    source_kwargs: dict | None = None,
) -> list[Recording]:
    """Generate conversations for each language and write:
         <out>/<lang>/<recording_id>.wav
         <out>/<lang>/<recording_id>.ref.json
         <out>/manifest.jsonl        (recording index for the runner)
         <out>/selection.json        (exact clip ids used, for reproducibility)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = get_source(source_name, **(source_kwargs or {}))
    dataset_name = f"synthetic_{source_name}"

    n_recordings = max(1, round(minutes_per_language / recording_minutes))
    recordings: list[Recording] = []
    selection: dict[str, dict] = {}

    for lang in languages:
        lang_dir = out_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        master_rng = random.Random(f"{seed}:{profile}:{lang}")
        max_speakers_needed = speakers_range[1]
        all_speakers = source.speakers(lang, max_speakers_needed, seed)

        for r in range(n_recordings):
            rec_id = f"{dataset_name}_{profile}_{lang}_{r:03d}"
            rng = random.Random(f"{seed}:{rec_id}")
            n_spk = rng.randint(*speakers_range)
            spk_ids = sorted(all_speakers)[:n_spk] if r % 2 == 0 else \
                rng.sample(sorted(all_speakers), n_spk)
            clips = {s: all_speakers[s] for s in spk_ids}

            audio, turns = build_conversation(clips, recording_minutes * 60.0, rng)
            duration = len(audio) / SR

            wav_path = lang_dir / f"{rec_id}.wav"
            ref_path = lang_dir / f"{rec_id}.ref.json"
            write_wav(wav_path, audio, SR)
            ref = Reference(
                recording_id=rec_id, language=lang, turns=turns,
                num_speakers=len({t.speaker for t in turns}),
                source_meta={"source": source_name, "profile": profile, "seed": seed},
            )
            atomic_write_json(ref_path, ref.to_dict())

            recordings.append(Recording(
                recording_id=rec_id, dataset=dataset_name, language=lang,
                audio_path=str(wav_path), reference_path=str(ref_path),
                duration_sec=round(duration, 3),
                num_speakers=ref.num_speakers, profile=profile,
            ))
            selection[rec_id] = {
                "language": lang, "num_speakers": ref.num_speakers,
                "clip_ids": [
                    {"speaker": t.speaker, "start": t.start} for t in turns
                ],
                "seed": seed,
            }

    save_manifest(out_dir / "manifest.jsonl", recordings)
    atomic_write_json(out_dir / "selection.json", {
        "profile": profile, "source": source_name, "seed": seed,
        "languages": languages, "minutes_per_language": minutes_per_language,
        "recordings": selection,
    })
    return recordings
