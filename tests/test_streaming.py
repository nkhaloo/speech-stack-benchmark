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
