#!/usr/bin/env python3
"""Diagnose residual within-turn silence in reference labels.

Missed speech in DER is charged whenever a reference turn says "speech" but the
audio (and therefore any honest VAD) is silent. `audio.trim_silence` removes the
*edge* silence of each source clip, but silence *inside* a clip (mid-utterance
pauses, breaths) still sits within the reference turn and is labeled as speech.

For each reference turn this script measures how much of the labeled span is
actually silent (frame RMS below a peak-relative threshold), and how much of that
silence lies **beyond the DER collar** of any reference boundary -- i.e. the part
that is forced to count as missed speech regardless of the diarizer. Compare the
reported "unforgiven silent" fraction against the run's observed missed_speech.

Run on the machine that has the prepared dataset:

    python scripts/diagnose_reference_silence.py \
        --manifest artifacts/datasets/synthetic_commonvoice_mdc/manifest.jsonl

    # or point at a run's copied manifest:
    python scripts/diagnose_reference_silence.py \
        --manifest artifacts/runs/2026-07-19_gpu_baseline_v1/config/manifest.jsonl
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from speech_benchmark.audio import load_audio
from speech_benchmark.schemas import load_manifest


def frame_rms(audio: np.ndarray, sr: int, frame: int, hop: int):
    """Vectorized frame RMS and each frame's start time (seconds)."""
    if len(audio) < frame:
        return np.zeros(0), np.zeros(0)
    power = audio.astype(np.float64) ** 2
    csum = np.concatenate(([0.0], np.cumsum(power)))
    starts = np.arange(0, len(audio) - frame + 1, hop)
    fp = (csum[starts + frame] - csum[starts]) / frame
    return np.sqrt(fp + 1e-12), starts / sr


def analyze(audio, sr, turns, collar, top_db, frame_ms=25.0, hop_ms=10.0):
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    fdur = hop / sr
    rms, ftime = frame_rms(audio, sr, frame, hop)
    if rms.size == 0:
        return 0.0, 0.0, 0.0

    boundaries = np.array(sorted({t.start for t in turns} | {t.end for t in turns}))

    ref_speech = sum(max(0.0, t.end - t.start) for t in turns)
    silent_times = []
    for t in turns:
        sel = (ftime >= t.start) & (ftime < t.end)
        if not sel.any():
            continue
        r = rms[sel]
        thr = float(r.max()) * (10.0 ** (-top_db / 20.0))
        silent_times.append(ftime[sel][r < thr])

    if not silent_times:
        return ref_speech, 0.0, 0.0
    st = np.concatenate(silent_times)
    naive_silent = st.size * fdur

    # Distance from each silent frame to the nearest reference boundary.
    idx = np.searchsorted(boundaries, st)
    left = boundaries[np.clip(idx - 1, 0, len(boundaries) - 1)]
    right = boundaries[np.clip(idx, 0, len(boundaries) - 1)]
    dist = np.minimum(np.abs(st - left), np.abs(st - right))
    unforgiven_silent = int((dist > collar).sum()) * fdur
    return ref_speech, naive_silent, unforgiven_silent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="dataset/run manifest.jsonl")
    ap.add_argument("--collar", type=float, default=0.25,
                    help="half-collar in seconds (DER der_collar/2; default 0.25)")
    ap.add_argument("--top-db", type=float, default=40.0,
                    help="silence threshold, dB below each turn's peak RMS")
    args = ap.parse_args()

    recordings = load_manifest(Path(args.manifest))
    by_lang = defaultdict(lambda: [0.0, 0.0, 0.0])  # ref, naive_silent, unforgiven
    total = [0.0, 0.0, 0.0]

    for rec in recordings:
        ref = rec.load_reference()
        if ref is None or not ref.turns:
            continue
        audio, sr = load_audio(rec.audio_path, 16000)
        ref_s, naive, unforg = analyze(audio, sr, ref.turns, args.collar, args.top_db)
        for acc in (by_lang[rec.language], total):
            acc[0] += ref_s
            acc[1] += naive
            acc[2] += unforg

    def pct(part, whole):
        return 100.0 * part / whole if whole else 0.0

    print(f"\nReference-silence diagnostic  (collar={args.collar}s, top_db={args.top_db})")
    print(f"{'lang':<6}{'ref speech (s)':>16}{'silent-in-turn':>16}{'unforgiven*':>14}")
    for lang in sorted(by_lang):
        ref_s, naive, unforg = by_lang[lang]
        print(f"{lang:<6}{ref_s:>16.1f}{pct(naive, ref_s):>15.1f}%{pct(unforg, ref_s):>13.1f}%")
    ref_s, naive, unforg = total
    print(f"{'ALL':<6}{ref_s:>16.1f}{pct(naive, ref_s):>15.1f}%{pct(unforg, ref_s):>13.1f}%")
    print("\n* 'unforgiven' = silence inside reference turns that is farther than the")
    print("  collar from any reference boundary -> a floor on missed_speech forced by")
    print("  the labels alone. Compare it to the run's observed missed_speech (e.g. 0.151).")
    print("  Near it  -> residual reference silence still dominates (fix the data).")
    print("  Well below -> remaining missed speech is genuine VAD behavior (try other diarizers).")


if __name__ == "__main__":
    main()
