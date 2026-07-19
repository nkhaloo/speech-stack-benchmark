"""Audio I/O helpers: mono float32 at a target sample rate, no heavy deps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 in [-1, 1], resampled to ``target_sr``."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != target_sr:
        g = np.gcd(int(sr), int(target_sr))
        mono = resample_poly(mono, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    return mono, sr


def trim_silence(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    top_db: float = 40.0,
    frame_ms: float = 25.0,
    pad_ms: float = 30.0,
) -> np.ndarray:
    """Trim leading/trailing near-silence from a clip.

    Source clips (e.g. Common Voice) carry silence before/after the utterance.
    If a reference turn is set to the full clip span, that silence is labeled as
    reference *speech* and the diarizer is charged with "missed speech" for
    correctly detecting nothing there. Trimming makes reference turns hug the
    actual speech.

    Frame RMS is compared to the clip's peak RMS (like ``librosa.effects.trim``'s
    ``top_db``) but with no extra dependency: frames quieter than ``top_db`` dB
    below the peak count as silence. A small ``pad_ms`` margin is kept around the
    retained speech so onsets/offsets are not clipped. Deterministic. Returns the
    input unchanged when it is shorter than one frame or entirely below threshold.
    """
    audio = np.asarray(audio, dtype=np.float32)
    frame = max(1, int(sr * frame_ms / 1000.0))
    if audio.size <= frame:
        return audio
    # Vectorized frame RMS at hop=1 via a cumulative sum of squares.
    power = audio.astype(np.float64) ** 2
    csum = np.concatenate(([0.0], np.cumsum(power)))
    frame_power = (csum[frame:] - csum[:-frame]) / frame
    rms = np.sqrt(frame_power + 1e-12)
    peak = float(rms.max())
    if peak <= 0.0:
        return audio
    threshold = peak * (10.0 ** (-top_db / 20.0))
    voiced = np.nonzero(rms >= threshold)[0]
    if voiced.size == 0:
        return audio
    pad = int(sr * pad_ms / 1000.0)
    start = max(0, int(voiced[0]) - pad)
    end = min(audio.size, int(voiced[-1]) + frame + pad)
    return audio[start:end]


def voiced_segments(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    top_db: float = 40.0,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    pad_ms: float = 100.0,
    min_dur_ms: float = 100.0,
) -> list[tuple[float, float]]:
    """Return voiced ``(start, end)`` spans (seconds), splitting on internal silence.

    Frames quieter than ``top_db`` dB below the clip's peak RMS are treated as
    silence. Contiguous voiced runs are padded by ``pad_ms`` and merged, so pauses
    shorter than ~2*``pad_ms`` stay inside one span (matching a diarizer's tolerance
    for brief gaps) while longer pauses split it. Spans below ``min_dur_ms`` are
    dropped. Used to build reference turns that follow speech activity: silence
    *inside* a source clip (breaths, phrase pauses) is left unlabeled instead of
    being counted as speech — otherwise the diarizer is charged with "missed
    speech" for correctly detecting that silence. Edge silence falls outside the
    first/last span, so this subsumes ``trim_silence`` for turn construction.
    """
    audio = np.asarray(audio, dtype=np.float32)
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if audio.size < frame:
        return [(0.0, audio.size / sr)] if audio.size else []
    power = audio.astype(np.float64) ** 2
    csum = np.concatenate(([0.0], np.cumsum(power)))
    starts = np.arange(0, audio.size - frame + 1, hop)
    rms = np.sqrt((csum[starts + frame] - csum[starts]) / frame + 1e-12)
    peak = float(rms.max())
    if peak < 1e-4:  # whole clip is (near-)silent: no speech to label
        return []
    thr = peak * (10.0 ** (-top_db / 20.0))
    voiced = rms >= thr
    if not voiced.any():
        return []
    ftime = starts / sr
    fdur = frame / sr
    idx = np.nonzero(voiced)[0]
    groups = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
    pad = pad_ms / 1000.0
    total = audio.size / sr
    spans = [[max(0.0, ftime[g[0]] - pad), min(total, ftime[g[-1]] + fdur + pad)]
             for g in groups]
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    min_dur = min_dur_ms / 1000.0
    return [(round(s, 3), round(e, 3)) for s, e in merged if e - s >= min_dur]


def write_wav(path: str | Path, audio: np.ndarray, sr: int = TARGET_SR) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sr, subtype="PCM_16")


def duration_sec(path: str | Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)
