import numpy as np

from speech_benchmark.audio import TARGET_SR, trim_silence, voiced_segments


def _tone(dur_sec: float, sr: int = TARGET_SR, freq: float = 220.0,
          amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(dur_sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_trim_silence_removes_edge_padding():
    sr = TARGET_SR
    sil = np.zeros(int(0.5 * sr), dtype=np.float32)  # 0.5 s of silence each side
    tone = _tone(1.0)                                 # 1.0 s of "speech"
    padded = np.concatenate([sil, tone, sil])

    trimmed = trim_silence(padded, sr=sr)

    # Result should be ~the 1.0 s of speech (plus a small pad), not the 2.0 s clip.
    assert 0.9 < len(trimmed) / sr < 1.25
    # Most of the 1.0 s of total silence must be gone.
    assert len(trimmed) < len(padded) - int(0.6 * sr)


def test_trim_silence_keeps_speech_when_no_padding():
    sr = TARGET_SR
    tone = _tone(1.0)
    trimmed = trim_silence(tone, sr=sr)
    # Nothing to trim -> length essentially unchanged.
    assert abs(len(trimmed) - len(tone)) < int(0.1 * sr)


def test_trim_silence_handles_pure_silence_and_short_input():
    sr = TARGET_SR
    # Entirely silent -> returned unchanged rather than emptied.
    silence = np.zeros(int(0.4 * sr), dtype=np.float32)
    assert len(trim_silence(silence, sr=sr)) == len(silence)
    # Shorter than one frame -> returned unchanged.
    tiny = np.ones(10, dtype=np.float32)
    assert np.array_equal(trim_silence(tiny, sr=sr), tiny)


def test_trim_silence_is_deterministic():
    padded = np.concatenate([np.zeros(3000, dtype=np.float32), _tone(0.7),
                             np.zeros(3000, dtype=np.float32)])
    assert np.array_equal(trim_silence(padded), trim_silence(padded))


def test_voiced_segments_splits_on_internal_pause():
    # 0.6s speech | 0.6s silence | 0.6s speech -> two voiced spans.
    clip = np.concatenate([_tone(0.6), np.zeros(int(0.6 * TARGET_SR), np.float32),
                           _tone(0.6)])
    segs = voiced_segments(clip)
    assert len(segs) == 2
    # First span ends before the gap; second starts after it; edges are trimmed.
    assert segs[0][0] < 0.2 and 0.5 < segs[0][1] < 0.9
    assert 1.0 < segs[1][0] < 1.4 and segs[1][1] > 1.5


def test_voiced_segments_merges_brief_gap():
    # 0.1s gap is shorter than ~2*pad, so the two tones stay one span.
    clip = np.concatenate([_tone(0.4), np.zeros(int(0.1 * TARGET_SR), np.float32),
                           _tone(0.4)])
    assert len(voiced_segments(clip)) == 1


def test_voiced_segments_single_span_for_continuous_speech():
    assert len(voiced_segments(_tone(1.0))) == 1


def test_voiced_segments_handles_silence_and_short_input():
    assert voiced_segments(np.zeros(int(0.4 * TARGET_SR), np.float32)) == []
    tiny = voiced_segments(np.ones(10, dtype=np.float32))
    assert len(tiny) == 1
