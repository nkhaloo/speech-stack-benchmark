import pytest

from speech_benchmark.metrics.asr_metrics import (asr_metrics, char_error_rate,
                                                  word_error_rate)
from speech_benchmark.metrics.diar_metrics import diarization_metrics
from speech_benchmark.metrics.text_norm import normalize_text, tokens
from speech_benchmark.schemas import ReferenceTurn, SpeakerTurn


def test_normalize_basic():
    assert normalize_text("Hello, World!") == "hello world"


def test_normalize_arabic_diacritics():
    assert normalize_text("كِتَابٌ", "ar") == "كتاب"


def test_tokens_chinese_chars():
    assert tokens("你好 世界。", "zh") == ["你", "好", "世", "界"]


def test_wer_exact_values():
    # 1 substitution over 4 reference words
    assert word_error_rate("the cat sat down", "the dog sat down") == 0.25
    assert word_error_rate("a b", "a b") == 0.0


def test_wer_chinese_char_level():
    # one wrong character of four
    assert word_error_rate("你好世界", "你好世间", "zh") == 0.25


def test_cer():
    assert char_error_rate("abcd", "abcx") == 0.25


def test_asr_metrics_counts():
    m = asr_metrics("one two three", "one two four", "en")
    assert m["wer"] == pytest.approx(1 / 3)
    assert m["ref_token_count"] == 3
    assert m["token_unit"] == "word"


def test_der_perfect_and_confused():
    ref = [ReferenceTurn("A", 0.0, 5.0, "x"), ReferenceTurn("B", 6.0, 10.0, "y")]
    perfect = [SpeakerTurn("s1", 0.0, 5.0), SpeakerTurn("s2", 6.0, 10.0)]
    m = diarization_metrics(ref, perfect, collar=0.0)
    assert m["der"] == pytest.approx(0.0, abs=1e-6)
    assert m["speaker_count_error"] == 0

    # both ref speakers mapped to one hyp speaker -> confusion
    confused = [SpeakerTurn("s1", 0.0, 5.0), SpeakerTurn("s1", 6.0, 10.0)]
    m2 = diarization_metrics(ref, confused, collar=0.0)
    assert m2["speaker_confusion"] > 0.3
    assert m2["speaker_count_error"] == -1
