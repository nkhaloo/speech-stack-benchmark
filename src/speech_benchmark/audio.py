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


def write_wav(path: str | Path, audio: np.ndarray, sr: int = TARGET_SR) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sr, subtype="PCM_16")


def duration_sec(path: str | Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)
