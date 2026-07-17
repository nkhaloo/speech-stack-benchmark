import pytest

from speech_benchmark.fusion.assign import CombinedResult, CombinedSentence
from speech_benchmark.metrics.combined_metrics import (combined_metrics, cpwer,
                                                       map_speakers_by_time)
from speech_benchmark.schemas import (Reference, ReferenceTurn, SpeakerTurn,
                                      Word)


def _reference():
    return Reference(recording_id="r", language="en", turns=[
        ReferenceTurn("alice", 0.0, 2.0, "hello there friend"),
        ReferenceTurn("bob", 3.0, 5.0, "good morning alice"),
    ], num_speakers=2)


def _combined(perfect=True):
    s1_words = [Word("hello", 0.1, 0.6, speaker="S1"),
                Word("there", 0.7, 1.2, speaker="S1"),
                Word("friend", 1.3, 1.9, speaker="S1")]
    text2 = "good morning alice" if perfect else "good evening alice"
    s2_words = [Word(t, 3.1 + i * 0.5, 3.5 + i * 0.5, speaker="S2")
                for i, t in enumerate(text2.split())]
    return CombinedResult(
        recording_id="r", asr_model_id="a", diarization_model_id="d",
        sentences=[
            CombinedSentence("hello there friend", 0.1, 1.9, "S1", s1_words),
            CombinedSentence(text2, 3.1, 4.5, "S2", s2_words),
        ], num_speakers=2)


def _hyp_turns():
    return [SpeakerTurn("S1", 0.0, 2.1), SpeakerTurn("S2", 2.9, 5.0)]


def test_speaker_mapping():
    mapping = map_speakers_by_time(_reference().turns, _hyp_turns())
    assert mapping == {"S1": "alice", "S2": "bob"}


def test_cpwer_perfect():
    assert cpwer(_reference(), _combined(True), "en")["cpwer"] == 0.0


def test_cpwer_one_substitution():
    # 1 error over 6 reference tokens
    assert cpwer(_reference(), _combined(False), "en")["cpwer"] == pytest.approx(1 / 6)


def test_cpwer_penalizes_wrong_speaker_split():
    # all words attributed to a single hyp speaker: bob's words become
    # insertions on alice's stream + deletions of bob's stream
    combined = _combined(True)
    for s in combined.sentences:
        s.speaker = "S1"
        for w in s.words:
            w.speaker = "S1"
    value = cpwer(_reference(), combined, "en")["cpwer"]
    assert value == pytest.approx(1.0)  # 3 insertions + 3 deletions over 6


def test_attribution_metrics_perfect():
    m = combined_metrics(_reference(), _combined(True), _hyp_turns(), "en")
    assert m["word_attribution_accuracy"] == 1.0
    assert m["sentence_attribution_accuracy"] == 1.0
