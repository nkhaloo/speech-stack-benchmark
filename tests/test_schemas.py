from speech_benchmark.schemas import (ASRResult, ASRSegment, DiarizationResult,
                                      SpeakerTurn, Word)


def test_asr_result_roundtrip():
    r = ASRResult(
        recording_id="rec1", model_id="m1", text="hello world",
        segments=[ASRSegment(text="hello world", start=0.0, end=1.5,
                             words=[Word("hello", 0.0, 0.7, 0.9),
                                    Word("world", 0.8, 1.5, 0.8)])],
        language_detected="en", runtime_sec=2.0, audio_duration_sec=4.0,
    )
    d = r.to_dict()
    r2 = ASRResult.from_dict(d)
    assert r2.text == "hello world"
    assert r2.segments[0].words[1].text == "world"
    assert r2.real_time_factor == 0.5
    assert d["real_time_factor"] == 0.5


def test_words_fallback_interpolation():
    r = ASRResult(recording_id="r", model_id="m",
                  segments=[ASRSegment(text="a b c d", start=0.0, end=4.0)])
    words = r.words()
    assert len(words) == 4
    assert words[0].start == 0.0 and abs(words[0].end - 1.0) < 1e-9
    assert abs(words[3].start - 3.0) < 1e-9


def test_diarization_roundtrip_and_rttm():
    d = DiarizationResult(
        recording_id="rec1", model_id="d1",
        turns=[SpeakerTurn("A", 0.0, 2.0), SpeakerTurn("B", 2.5, 4.0)],
        num_speakers=2,
    )
    d2 = DiarizationResult.from_dict(d.to_dict())
    assert d2.speaker_labels() == ["A", "B"]
    rttm = d.to_rttm()
    assert "SPEAKER rec1 1 0.000 2.000" in rttm
    assert rttm.count("\n") == 2
