from speech_benchmark.fusion.assign import fuse, split_sentences
from speech_benchmark.schemas import (ASRResult, ASRSegment, DiarizationResult,
                                      SpeakerTurn, Word)


def _asr():
    return ASRResult(
        recording_id="r", model_id="asr",
        segments=[
            ASRSegment(text="hello there.", start=0.0, end=2.0, words=[
                Word("hello", 0.0, 1.0), Word("there.", 1.0, 2.0)]),
            ASRSegment(text="how are you", start=5.0, end=7.0, words=[
                Word("how", 5.0, 5.5), Word("are", 5.5, 6.0), Word("you", 6.0, 7.0)]),
        ])


def _diar():
    return DiarizationResult(
        recording_id="r", model_id="diar",
        turns=[SpeakerTurn("S1", 0.0, 2.2), SpeakerTurn("S2", 4.8, 7.2)],
        num_speakers=2)


def test_fuse_assigns_speakers_and_sentences():
    combined = fuse(_asr(), _diar())
    assert len(combined.sentences) == 2
    s1, s2 = combined.sentences
    assert s1.speaker == "S1" and s1.text == "hello there."
    assert s2.speaker == "S2" and s2.text == "how are you"
    assert all(w.speaker == "S1" for w in s1.words)


def test_split_on_gap_without_punctuation():
    words = [Word("a", 0.0, 0.5), Word("b", 0.6, 1.0), Word("c", 3.0, 3.5)]
    groups = split_sentences(words, gap_threshold_sec=0.8)
    assert [len(g) for g in groups] == [2, 1]


def test_split_on_punctuation():
    words = [Word("ok.", 0.0, 0.5), Word("next", 0.6, 1.0)]
    groups = split_sentences(words)
    assert len(groups) == 2


def test_word_without_overlap_falls_back_to_nearest():
    asr = ASRResult(recording_id="r", model_id="a", segments=[
        ASRSegment(text="stray", start=2.9, end=3.1,
                   words=[Word("stray", 2.9, 3.1)])])
    combined = fuse(asr, _diar())
    # nearest turn within tolerance is S1 (ends 2.2) vs S2 (starts 4.8)
    assert combined.sentences[0].speaker == "S1"
