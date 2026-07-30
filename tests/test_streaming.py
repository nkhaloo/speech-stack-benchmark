"""Streaming benchmark tests: schema round-trip, dummy streaming adapter,
end-to-end streaming runner + metrics + report, and graceful unavailability of
real/native stacks — all with zero downloads."""

import pytest

from speech_benchmark.benchmark import RunContext
from speech_benchmark.schemas import (Emission, Reference, ReferenceTurn,
                                       StreamingResult, load_json)
from speech_benchmark.streaming import (AdapterUnavailable,
                                        create_streaming_adapter)
from speech_benchmark.streaming.metrics import reconstruct, streaming_metrics
from speech_benchmark.streaming.diagnostics import generate_streaming_diagnostics
from speech_benchmark.streaming.report import generate_streaming_report
from speech_benchmark.streaming.runner import StreamingRunner

STREAM_CFG = {
    "track": "streaming",
    "languages": ["en", "zh"],
    "streaming": {"frame_sec": 0.5, "latency_budget_sec": 2.0},
    "streaming_stacks": [{
        "id": "dummy-stream", "family": "dummy", "runtime": "dummy_stream",
        "wer_target": 0.1, "confusion_rate": 0.05, "latency_sec": 0.6,
        "revision_rate": 0.3,
    }],
    "metrics": {"der_collar": 0.5, "der_skip_overlap": False},
}


def test_emission_and_streaming_result_roundtrip():
    e = Emission(sentence_id=3, text="hi there", speaker="SPK01", start=1.0,
                 end=2.0, audio_time=2.5, wall_time=3.1, is_final=True, revision=2)
    assert Emission.from_dict(e.to_dict()) == e
    sr = StreamingResult(recording_id="r1", model_id="m1", emissions=[e],
                         audio_duration_sec=10.0, runtime_sec=1.0)
    back = StreamingResult.from_dict(sr.to_dict())
    assert back.emissions[0] == e
    assert back.real_time_factor == 0.1


def test_final_emissions_prefers_final_and_orders():
    sr = StreamingResult(recording_id="r", model_id="m", emissions=[
        Emission(0, "a partial", start=0.0, end=1.0, is_final=False, revision=0),
        Emission(0, "a final", start=0.0, end=1.0, is_final=True, revision=1),
        Emission(1, "later", start=2.0, end=3.0, is_final=True),
    ])
    finals = sr.final_emissions()
    assert [f.text for f in finals] == ["a final", "later"]


def test_dummy_streaming_adapter_incremental():
    ref = Reference(recording_id="r", language="en", turns=[
        ReferenceTurn("A", 0.0, 1.0, "hello world"),
        ReferenceTurn("B", 1.5, 2.5, "goodbye now"),
    ])
    from speech_benchmark.schemas import Recording

    class _Rec(Recording):
        def load_reference(self):
            return ref

    rec = _Rec(recording_id="r", dataset="d", language="en", audio_path="x")
    ad = create_streaming_adapter(STREAM_CFG["streaming_stacks"][0])
    ad.load()
    ad.reset(rec)
    import numpy as np
    got = []
    # feed 3s of audio in 0.5s frames
    for k in range(6):
        got += ad.push(np.zeros(8000, dtype=np.float32), (k + 1) * 0.5)
    got += ad.flush()
    finals = [e for e in got if e.is_final]
    assert {e.sentence_id for e in finals} == {0, 1}
    # first turn (ends at 1.0) must finalize before the second (ends at 2.5)
    f0 = next(e for e in finals if e.sentence_id == 0)
    f1 = next(e for e in finals if e.sentence_id == 1)
    assert f0.audio_time <= f1.audio_time


def test_reconstruct_and_metrics():
    ref = Reference(recording_id="r", language="en", turns=[
        ReferenceTurn("A", 0.0, 1.0, "hello world"),
        ReferenceTurn("B", 1.5, 2.5, "goodbye now"),
    ])
    sr = StreamingResult(recording_id="r", model_id="m", audio_duration_sec=3.0,
                         runtime_sec=0.3, emissions=[
        Emission(0, "hello world", "SPK00", 0.0, 1.0, 1.0, 1.6, is_final=True),
        Emission(1, "goodbye", "SPK01", 1.5, 2.5, 2.5, 3.1, is_final=True, revision=1),
    ])
    combined, turns, text = reconstruct(sr)
    assert text == "hello world goodbye"
    assert len(turns) == 2
    m = streaming_metrics(sr, ref, "en")
    for key in ("wer", "cpwer", "der", "revision_rate", "speaker_label_churn",
                "time_to_first_token_sec", "finalization_delay_median_sec"):
        assert key in m
    assert 0.0 <= m["wer"] <= 1.0
    assert m["time_to_first_token_sec"] == 1.6


def test_streaming_end_to_end(dummy_dataset, tmp_path):
    _, recordings = dummy_dataset
    ctx = RunContext(tmp_path / "artifacts", "testrun_streaming_v1")
    StreamingRunner(STREAM_CFG, ctx, recordings).run()

    manifest = load_json(ctx.manifest_path)
    assert manifest["status"] == "completed"
    assert manifest["streaming_stacks"] == ["dummy-stream"]
    assert manifest["streaming_contract"]["frame_sec"] == 0.5

    for rec in recordings:
        sr = load_json(ctx.streaming_path("dummy-stream", rec.recording_id))
        assert sr["status"] == "completed"
        assert len(sr["emissions"]) > 0
        assert sr["resources"]["peak_ram_mb"] > 0

    rows = load_json(ctx.run_dir / "metrics/per_recording/streaming_rows.json")
    assert len(rows) == len(recordings)
    for r in rows:
        assert r["status"] == "completed"
        assert r["wer"] is not None and 0 <= r["wer"] < 0.5
        assert r["cpwer"] is not None
        assert 0.0 <= r["revision_rate"] <= 1.0
        assert r["finalization_delay_median_sec"] is not None
        assert r["latency_budget_sec"] == 2.0

    out = generate_streaming_report(ctx.run_dir)
    md = out.read_text()
    assert "Streaming speech-stack benchmark" in md
    assert "Dummy-only run" in md  # flagged, not silently ranked
    assert "Final accuracy" in md

    diagnostics = generate_streaming_diagnostics(ctx.run_dir)
    text = diagnostics.read_text()
    assert "cache-only analysis" in text
    assert "fused-timeline DER" in text
    payload = load_json(ctx.run_dir / "metrics/streaming_diagnostics.json")
    assert payload["overall"]["recordings"] == len(recordings)
    assert payload["overall"]["latency_budget_pass_rate"] is not None
    assert set(payload["per_language"]) == {"en", "zh"}


def test_streaming_resume_skips_cached(dummy_dataset, tmp_path):
    _, recordings = dummy_dataset
    ctx = RunContext(tmp_path / "artifacts", "testrun_streaming_resume_v1")
    StreamingRunner(STREAM_CFG, ctx, recordings).run()
    first = load_json(ctx.streaming_path("dummy-stream", recordings[0].recording_id))
    StreamingRunner(STREAM_CFG, ctx, recordings).run()
    second = load_json(ctx.streaming_path("dummy-stream", recordings[0].recording_id))
    assert first["created_at"] == second["created_at"]


def test_sentence_tracker_new_revise_finalize():
    from speech_benchmark.streaming.tracking import SentenceTracker
    tr = SentenceTracker(finalize_after_sec=5.0, match_tolerance_sec=1.0)

    e = tr.update([{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "A"}], 1.5)
    assert len(e) == 1 and e[0].sentence_id == 0 and not e[0].is_final and e[0].revision == 0

    # same sentence (start within tolerance), changed text -> revision, same id
    e = tr.update([{"start": 0.1, "end": 1.0, "text": "hello there", "speaker": "A"}], 2.0)
    rev = [x for x in e if not x.is_final]
    assert rev and rev[0].sentence_id == 0 and rev[0].revision == 1

    # much later -> finalization of sentence 0 (audio_time_end - end > 5)
    e = tr.update([], 7.0)
    fin = [x for x in e if x.is_final and x.sentence_id == 0]
    assert fin and fin[0].text == "hello there"

    # a new, well-separated sentence gets a fresh id
    e = tr.update([{"start": 20.0, "end": 21.0, "text": "bye", "speaker": "B"}], 21.5)
    assert any(x.sentence_id == 1 for x in e)


def test_sentence_tracker_final_flushes_all():
    from speech_benchmark.streaming.tracking import SentenceTracker
    tr = SentenceTracker()
    tr.update([{"start": 0.0, "end": 1.0, "text": "one", "speaker": "A"}], 1.2)
    out = tr.update([{"start": 5.0, "end": 6.0, "text": "two", "speaker": "B"}], 6.2,
                    final=True)
    finals = {x.sentence_id for x in out if x.is_final}
    assert finals == {0, 1}


def test_windowed_stack_requires_cards():
    ad = create_streaming_adapter({"id": "bad", "runtime": "windowed_stack"})
    with pytest.raises(AdapterUnavailable):
        ad.load()


def test_native_stubs_unavailable():
    # diart isn't installed in the base env, so the native stack reports
    # unavailable rather than crashing the run.
    ad = create_streaming_adapter({"id": "x", "runtime": "diart_whisper"})
    with pytest.raises(AdapterUnavailable):
        ad.load()


def test_whisperlive_protocol_tracks_latest_segments():
    import json
    import threading

    from speech_benchmark.streaming.whisperlive_client import WhisperLiveClient

    client = WhisperLiveClient({"model": "large-v3-turbo"})
    client._uid = "test-session"
    client._ready = threading.Event()
    client._closed = threading.Event()
    client._condition = threading.Condition()
    client._segments = []
    client._version = 0
    client._error = None

    client._on_message(None, json.dumps({
        "uid": "test-session", "message": "SERVER_READY",
        "backend": "faster_whisper",
    }))
    assert client._ready.is_set()

    segments = [{"start": "0.0", "end": "1.2", "text": " hello",
                 "completed": False}]
    client._on_message(None, json.dumps({
        "uid": "test-session", "segments": segments,
    }))
    assert client.snapshot() == segments


def test_diart_card_bounds_online_speaker_clusters():
    from speech_benchmark.config import load_yaml, project_root

    card = load_yaml(project_root() / "configs/models/stream_diart_whisper.yaml")
    assert card["diart"]["max_speakers"] == 4
    assert card["diart"]["delta_new"] > 1.0
    assert card["diart"]["latency"] == 2.0


def test_tuning_track_applies_nested_card_overrides():
    from speech_benchmark.config import load_track_config, project_root

    cfg = load_track_config(
        project_root() / "configs/streaming_diart_whisperlive_tuning.yaml")
    stacks = {s["id"]: s for s in cfg["streaming_stacks"]}
    assert len(stacks) == 7
    assert stacks["tune-control"]["diart"]["latency"] == 0.5
    assert stacks["tune-control"]["whisperlive"]["finalize_after_sec"] == 5.0
    assert stacks["tune-finalize-2s"]["whisperlive"]["finalize_after_sec"] == 2.0
    assert stacks["tune-finalize-2s"]["whisperlive"]["model"] == "large-v3-turbo"
    assert stacks["tune-diart-latency-2s"]["diart"]["latency"] == 2.0
    combined = stacks["tune-combined-candidate"]
    assert combined["diart"]["delta_new"] == 0.95
    assert combined["whisperlive"]["same_output_threshold"] == 3


# --------------------------------------------------- batchdiar (streaming ASR
# --------------------------------------------------- + one-shot batch diarization)

class _StubASR:
    """Returns one word per second of the window it is handed, so fused
    sentence text reveals exactly which window produced it."""

    def __init__(self):
        self.calls = 0

    def load(self):
        pass

    def unload(self):
        pass

    def transcribe(self, recording, language=None):
        from speech_benchmark.audio import load_audio
        from speech_benchmark.schemas import ASRResult, ASRSegment, Word
        self.calls += 1
        audio, sr = load_audio(recording.audio_path, 16000)
        n = max(1, int(len(audio) / sr))
        words = [Word(text=f"w{i}", start=float(i), end=float(i) + 0.5)
                 for i in range(n)]
        seg = ASRSegment(text=" ".join(w.text for w in words),
                         start=0.0, end=float(n), words=words)
        return ASRResult(recording_id=recording.recording_id, model_id="stub",
                         text=seg.text, segments=[seg])


class _StubDiar:
    """Whole-file diarization: two speakers, alternating 5s turns."""

    def __init__(self):
        self.calls = 0

    def load(self):
        pass

    def unload(self):
        pass

    def diarize(self, recording):
        from speech_benchmark.schemas import DiarizationResult, SpeakerTurn
        self.calls += 1
        turns = [SpeakerTurn("SPEAKER_00" if k % 2 == 0 else "SPEAKER_01",
                             5.0 * k, 5.0 * (k + 1)) for k in range(8)]
        return DiarizationResult(recording_id=recording.recording_id,
                                 model_id="stub-diar", turns=turns)


def _batchdiar_adapter(monkeypatch, policy="sliding"):
    import numpy as np
    from speech_benchmark.streaming import batchdiar

    asr, diar = _StubASR(), _StubDiar()
    monkeypatch.setattr(batchdiar, "create_asr_adapter", lambda card: asr)
    monkeypatch.setattr(batchdiar, "create_diarization_adapter", lambda card: diar)
    ad = create_streaming_adapter({
        "id": "cpu-stream-test", "family": "batchdiar",
        "runtime": "batchdiar_stack",
        "asr": {"id": "a", "runtime": "dummy"},
        "diarization": {"id": "d", "runtime": "dummy"},
        "sentence_gap_sec": 0.8,
        "window": {"policy": policy, "emit_every_sec": 2.0, "window_sec": 10.0,
                   "finalize_after_sec": 5.0, "match_tolerance_sec": 1.0},
    })
    ad.load()
    return ad, asr, diar, np


def test_batchdiar_requires_cards():
    ad = create_streaming_adapter({"id": "bad", "runtime": "batchdiar_stack"})
    with pytest.raises(AdapterUnavailable):
        ad.load()


def test_batchdiar_diarizes_once_per_recording(monkeypatch, tmp_path):
    """The defining property: diarization runs exactly once no matter how many
    windows the ASR side chews through, and re-running reset re-diarizes."""
    from speech_benchmark.schemas import Recording
    ad, asr, diar, np = _batchdiar_adapter(monkeypatch)

    rec = Recording(recording_id="r1", dataset="d", language="en",
                    audio_path=str(tmp_path / "a.wav"))
    ad.reset(rec)
    assert diar.calls == 1

    sr = 16000
    for k in range(20):  # 20 x 0.5s = 10s of audio
        ad.push(np.zeros(sr // 2, dtype=np.float32), (k + 1) * 0.5)
    ad.flush()

    assert diar.calls == 1, "diarization must not re-run per window"
    assert asr.calls > 1, "ASR must re-run incrementally"

    ad.reset(Recording(recording_id="r2", dataset="d", language="en",
                       audio_path=str(tmp_path / "b.wav")))
    assert diar.calls == 2, "each recording gets its own batch pass"


def test_batchdiar_speaker_labels_stay_globally_consistent(monkeypatch, tmp_path):
    """A sliding window re-clusters per step under windowed_stack; here labels
    come from the fixed whole-file pass, so a given audio time always maps to
    the same speaker."""
    from speech_benchmark.schemas import Recording
    ad, asr, diar, np = _batchdiar_adapter(monkeypatch, policy="sliding")

    ad.reset(Recording(recording_id="r1", dataset="d", language="en",
                       audio_path=str(tmp_path / "a.wav")))
    sr = 16000
    emissions = []
    for k in range(60):  # 30s
        emissions.extend(ad.push(np.zeros(sr // 2, dtype=np.float32), (k + 1) * 0.5))
    emissions.extend(ad.flush())

    finals = [e for e in emissions if e.is_final and e.end > e.start]
    assert finals
    # Stub diarization: SPEAKER_00 on even 5s blocks, SPEAKER_01 on odd. Judge
    # by sentence midpoint — fusion splits sentences on speaker change, so a
    # finalized sentence lies inside one block. (Transient pre-final revisions
    # are not checked: a zero-overlap word landing exactly on a turn boundary
    # ties in max-overlap fusion, which windowed_stack shares and metrics
    # never see, since they read final emissions only.)
    for e in finals:
        mid = (e.start + e.end) / 2.0
        expected = "SPEAKER_00" if int(mid // 5.0) % 2 == 0 else "SPEAKER_01"
        assert e.speaker == expected, (
            f"final sentence {e.start:.1f}-{e.end:.1f}s labelled {e.speaker}, "
            f"expected {expected}")


def test_batchdiar_contract_meta_records_deferred_labeling(monkeypatch, tmp_path):
    """The contract must state that labels land at end of session, so this arm
    is never compared against a live-labeling stack on attribution latency."""
    from speech_benchmark.schemas import Recording
    ad, asr, diar, np = _batchdiar_adapter(monkeypatch)

    meta = ad.contract_meta()
    assert meta["labels_finalized"] == "end_of_session"
    assert meta["causal"] is False
    assert meta["diarization_mode"] == "batch_once"
    assert meta["asr_mode"] == "windowed"
    assert meta["native"] is False

    ad.reset(Recording(recording_id="r1", dataset="d", language="en",
                       audio_path=str(tmp_path / "a.wav")))
    mm = ad.model_meta()
    assert mm["diarization_mode"] == "batch_once"
    assert isinstance(mm["batch_diarization_sec"], float)


# ------------------------------------------- nativestream (natively streaming
# ------------------------------------------- ASR + one-shot batch diarization)

class _StubOnlineASR:
    """A streaming decoder that emits one word per second of audio it has been
    fed, cumulatively — so ``words()`` growing across pushes proves state is
    being carried rather than the buffer re-decoded."""

    def __init__(self, languages=("en", "es", "fr", "ar", "zh")):
        self.languages_ = list(languages)
        self.resets = 0
        self.accepts = 0
        self.finalized = 0
        self._sec = 0.0

    def load(self):
        pass

    def unload(self):
        pass

    def reset(self, language):
        from speech_benchmark.streaming.online_asr import UnsupportedLanguage
        if language not in self.languages_:
            raise UnsupportedLanguage(f"no weights for {language!r}")
        self.resets += 1
        self._sec = 0.0

    def accept(self, audio):
        self.accepts += 1
        self._sec += len(audio) / 16000.0

    def words(self):
        from speech_benchmark.schemas import Word
        return [Word(text=f"w{i}", start=float(i), end=float(i) + 0.5)
                for i in range(int(self._sec))]

    def finalize(self):
        self.finalized += 1

    def model_meta(self):
        return {"asr_id": "stub-online", "asr_runtime": "stub"}


def _nativestream_adapter(monkeypatch, languages=("en", "es", "fr", "ar", "zh")):
    import numpy as np
    from speech_benchmark.streaming import nativestream

    asr, diar = _StubOnlineASR(languages), _StubDiar()
    monkeypatch.setattr(nativestream, "create_online_asr", lambda card: asr)
    monkeypatch.setattr(nativestream, "create_diarization_adapter", lambda card: diar)
    ad = create_streaming_adapter({
        "id": "cpu-stream-native-test", "family": "nativestream",
        "runtime": "native_batchdiar_stack", "native": True,
        "asr": {"id": "a", "runtime": "dummy"},
        "diarization": {"id": "d", "runtime": "dummy"},
        "sentence_gap_sec": 0.8,
        "window": {"emit_every_sec": 1.0, "finalize_after_sec": 2.0,
                   "match_tolerance_sec": 1.0},
    })
    ad.load()
    return ad, asr, diar, np


def test_nativestream_requires_cards():
    ad = create_streaming_adapter({"id": "bad", "runtime": "native_batchdiar_stack"})
    with pytest.raises(AdapterUnavailable):
        ad.load()


def test_nativestream_decodes_each_frame_once(monkeypatch, tmp_path):
    """The reason this stack exists: audio is handed to the decoder exactly
    once. One accept per pushed frame, and no re-transcription of the buffer."""
    from speech_benchmark.schemas import Recording
    ad, asr, diar, np = _nativestream_adapter(monkeypatch)

    ad.reset(Recording(recording_id="r1", dataset="d", language="en",
                       audio_path=str(tmp_path / "a.wav")))
    sr = 16000
    for k in range(20):  # 20 x 0.5s = 10s
        ad.push(np.zeros(sr // 2, dtype=np.float32), (k + 1) * 0.5)
    ad.flush()

    assert asr.accepts == 20, "each frame must be fed to the decoder exactly once"
    assert asr.finalized == 1
    assert diar.calls == 1, "diarization must not re-run per emission"


def test_nativestream_diarizes_once_per_recording(monkeypatch, tmp_path):
    from speech_benchmark.schemas import Recording
    ad, asr, diar, np = _nativestream_adapter(monkeypatch)

    ad.reset(Recording(recording_id="r1", dataset="d", language="en",
                       audio_path=str(tmp_path / "a.wav")))
    assert diar.calls == 1
    for k in range(20):
        ad.push(np.zeros(8000, dtype=np.float32), (k + 1) * 0.5)
    ad.flush()
    assert diar.calls == 1

    ad.reset(Recording(recording_id="r2", dataset="d", language="en",
                       audio_path=str(tmp_path / "b.wav")))
    assert diar.calls == 2, "each recording gets its own batch pass"


def test_nativestream_speaker_labels_stay_globally_consistent(monkeypatch, tmp_path):
    """Labels come from the fixed whole-file pass, so a given audio time always
    maps to the same speaker — the property that keeps churn at ~0."""
    from speech_benchmark.schemas import Recording
    ad, asr, diar, np = _nativestream_adapter(monkeypatch)

    ad.reset(Recording(recording_id="r1", dataset="d", language="en",
                       audio_path=str(tmp_path / "a.wav")))
    emissions = []
    for k in range(60):  # 30s
        emissions.extend(ad.push(np.zeros(8000, dtype=np.float32), (k + 1) * 0.5))
    emissions.extend(ad.flush())

    finals = [e for e in emissions if e.is_final and e.end > e.start]
    assert finals
    for e in finals:
        mid = (e.start + e.end) / 2.0
        expected = "SPEAKER_00" if int(mid // 5.0) % 2 == 0 else "SPEAKER_01"
        assert e.speaker == expected, (
            f"final sentence {e.start:.1f}-{e.end:.1f}s labelled {e.speaker}, "
            f"expected {expected}")


def test_nativestream_unsupported_language_skips_before_diarizing(monkeypatch, tmp_path):
    """A language the model has no weights for must raise UnsupportedLanguage —
    and must do so before any diarization time is spent on the recording."""
    from speech_benchmark.schemas import Recording
    from speech_benchmark.streaming.online_asr import UnsupportedLanguage
    ad, asr, diar, np = _nativestream_adapter(monkeypatch, languages=("en",))

    with pytest.raises(UnsupportedLanguage):
        ad.reset(Recording(recording_id="r1", dataset="d", language="ar",
                           audio_path=str(tmp_path / "a.wav")))
    assert diar.calls == 0, "must not diarize a recording it cannot transcribe"


def test_runner_records_unsupported_language_as_skipped(monkeypatch, tmp_path,
                                                        dummy_dataset):
    """UnsupportedLanguage is expected, not a failure: the row must read
    `skipped` so a missing language never looks like a bad score."""
    from speech_benchmark.benchmark import RunContext
    from speech_benchmark.streaming.online_asr import UnsupportedLanguage
    from speech_benchmark.streaming.runner import StreamingRunner

    _, recordings = dummy_dataset

    class _Boom:
        def load(self): pass
        def unload(self): pass
        def reset(self, rec): raise UnsupportedLanguage("no weights for 'xx'")
        def push(self, a, t): return []
        def flush(self): return []
        def contract_meta(self): return {"native": True}
        def model_meta(self): return {}

    import speech_benchmark.streaming.runner as runner_mod
    monkeypatch.setattr(runner_mod, "create_streaming_adapter", lambda cfg: _Boom())

    ctx = RunContext(str(tmp_path / "artifacts"), "skiprun")
    cfg = {"track": "cpu_streaming", "streaming": {"frame_sec": 0.5},
           "streaming_stacks": [{"id": "s1", "family": "nativestream"}],
           "metrics": {}}
    StreamingRunner(cfg, ctx, recordings).run()

    from speech_benchmark.schemas import load_json
    for rec in recordings:
        got = load_json(ctx.streaming_path("s1", rec.recording_id))
        assert got["status"] == "skipped"
        assert "unsupported language" in got["error"]
    assert not list((ctx.run_dir / "errors").glob("*.json")), \
        "an unsupported language must not be logged as an error"


def test_nativestream_contract_meta_records_regime(monkeypatch, tmp_path):
    """The contract must say both things at once: the ASR path is natively
    streaming, and speaker labels still land at end of session."""
    from speech_benchmark.schemas import Recording
    ad, asr, diar, np = _nativestream_adapter(monkeypatch)

    meta = ad.contract_meta()
    assert meta["asr_mode"] == "native_streaming"
    assert meta["native"] is True
    assert meta["window_sec"] is None, "a native stack buffers no window"
    assert meta["labels_finalized"] == "end_of_session"
    assert meta["causal"] is False
    assert meta["diarization_mode"] == "batch_once"

    ad.reset(Recording(recording_id="r1", dataset="d", language="en",
                       audio_path=str(tmp_path / "a.wav")))
    mm = ad.model_meta()
    assert mm["asr_mode"] == "native_streaming"
    assert isinstance(mm["batch_diarization_sec"], float)


def test_cpu_streaming_track_config_loads():
    from speech_benchmark.config import load_track_config, project_root

    cfg = load_track_config(project_root() / "configs/cpu_streaming.yaml")
    stacks = {s["id"]: s for s in cfg["streaming_stacks"]}
    assert cfg["track"] == "cpu_streaming"

    # The defining constraint of this track: every ASR arm is natively
    # streaming. A windowed stack here would silently reintroduce the >2.0 RTF
    # regime the track was rebuilt to escape.
    for s in cfg["streaming_stacks"]:
        assert s["runtime"] == "native_batchdiar_stack", s["id"]
        assert s.get("native") is True, s["id"]
        assert "window_sec" not in (s.get("window") or {}), s["id"]

    # The latency ladder must vary ONLY the chunk size, or its gap measures
    # something other than the price of lower latency.
    base = stacks["cpu-stream-nemotron35-560ms-sherpa"]
    fast = stacks["cpu-stream-nemotron35-160ms-sherpa"]
    assert base["diarization"] == fast["diarization"]
    assert base["window"] == fast["window"]
    assert base["asr"] != fast["asr"]

    # The diarizer swap must hold the ASR side fixed.
    swap = stacks["cpu-stream-nemotron35-560ms-pyannote31"]
    assert swap["asr"] == base["asr"]
    assert swap["diarization"] != base["diarization"]


def test_cpu_streaming_asr_cards_cover_all_benchmark_languages():
    """Single multilingual checkpoints, not one model per language: every ASR
    card must cover all five benchmark languages from one set of weights."""
    from speech_benchmark.config import load_track_config, load_yaml, project_root, resolve_path

    root = project_root()
    cfg = load_track_config(root / "configs/cpu_streaming.yaml")
    wanted = set(cfg["languages"])
    seen = set()
    for s in cfg["streaming_stacks"]:
        card = load_yaml(resolve_path(s["asr"], root))
        assert set(card["languages"]) >= wanted, (
            f"{card['id']} covers {card['languages']}, needs {sorted(wanted)}")
        assert "models" not in card, (
            f"{card['id']} uses a per-language model map; this track requires "
            "one multilingual checkpoint")
        seen.add(card["id"])
    assert seen, "no ASR cards found"
