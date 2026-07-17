"""End-to-end pipeline test with dummy adapters: dataset -> ASR ->
diarization -> fusion -> metrics -> report, all inside a temp artifacts dir.
This is the same path the smoke test exercises."""

import pandas as pd

from speech_benchmark.benchmark import BenchmarkRunner, RunContext
from speech_benchmark.reporting import aggregate_run, generate_full_report
from speech_benchmark.schemas import load_json

TRACK_CFG = {
    "track": "smoke",
    "languages": ["en", "zh"],
    "asr_models": [{"id": "dummy-asr", "family": "dummy", "runtime": "dummy",
                    "wer_target": 0.1}],
    "diarization_models": [{"id": "dummy-diar", "family": "dummy",
                            "runtime": "dummy", "boundary_jitter_sec": 0.1,
                            "confusion_rate": 0.0}],
    "fusion": {"sentence_gap_sec": 0.8},
    "metrics": {"der_collar": 0.5, "der_skip_overlap": False},
}


def test_end_to_end(dummy_dataset, tmp_path):
    _, recordings = dummy_dataset
    ctx = RunContext(tmp_path / "artifacts", "testrun_smoke_v1")
    runner = BenchmarkRunner(TRACK_CFG, ctx, recordings)
    runner.run()

    # manifest
    manifest = load_json(ctx.manifest_path)
    assert manifest["status"] == "completed"
    assert manifest["asr_models"] == ["dummy-asr"]
    assert len(manifest["recordings"]) == 4
    assert manifest["environment"]["python_version"]

    # cached predictions exist and are complete
    for rec in recordings:
        asr = load_json(ctx.asr_path("dummy-asr", rec.recording_id))
        assert asr["status"] == "completed"
        assert asr["runtime_sec"] is not None
        assert asr["resources"]["peak_ram_mb"] > 0
        diar = load_json(ctx.diar_path("dummy-diar", rec.recording_id))
        assert diar["status"] == "completed"
        assert ctx.rttm_path("dummy-diar", rec.recording_id).exists()
        combined = load_json(
            ctx.combined_path("dummy-asr", "dummy-diar", rec.recording_id))
        assert combined["status"] == "completed"
        assert len(combined["sentences"]) > 0

    # metrics rows
    pair_rows = load_json(ctx.run_dir / "metrics/per_recording/pair_rows.json")
    assert len(pair_rows) == 4
    row = pair_rows[0]
    assert row["cpwer"] is not None and 0 <= row["cpwer"] < 0.6
    assert row["word_attribution_accuracy"] > 0.8

    asr_rows = load_json(ctx.run_dir / "metrics/per_recording/asr_rows.json")
    en_rows = [r for r in asr_rows if r["language"] == "en"]
    assert all(0 < r["wer"] < 0.4 for r in en_rows)
    assert all(r["language_detection_correct"] for r in asr_rows)

    diar_rows = load_json(ctx.run_dir / "metrics/per_recording/diar_rows.json")
    assert all(r["der"] is not None and r["der"] < 0.5 for r in diar_rows)
    assert all(r["der_collar"] == 0.5 for r in diar_rows)

    # report generation from saved results only
    generate_full_report(ctx.run_dir)
    summary = (ctx.run_dir / "reports/summary.md").read_text()
    assert "Benchmark summary" in summary
    assert (ctx.run_dir / "metrics/per_language/asr_by_language.csv").exists()
    tables = aggregate_run(ctx.run_dir)
    assert not tables["asr_by_language"].empty
    assert tables["failures"].empty

    # index + latest pointer
    idx = pd.read_csv(tmp_path / "artifacts/index.csv")
    assert (idx["run_id"] == "testrun_smoke_v1").any()


def test_resume_skips_cached(dummy_dataset, tmp_path, capsys):
    _, recordings = dummy_dataset
    ctx = RunContext(tmp_path / "artifacts", "testrun_resume_v1")
    BenchmarkRunner(TRACK_CFG, ctx, recordings).run()
    first = load_json(ctx.asr_path("dummy-asr", recordings[0].recording_id))
    # rerun: cached outputs must be reused (created_at unchanged)
    BenchmarkRunner(TRACK_CFG, ctx, recordings).run()
    second = load_json(ctx.asr_path("dummy-asr", recordings[0].recording_id))
    assert first["created_at"] == second["created_at"]


def test_streaming_simulation(dummy_dataset, tmp_path):
    _, recordings = dummy_dataset
    cfg = dict(TRACK_CFG)
    cfg["asr_models"] = [dict(cfg["asr_models"][0], chunked_eval=True)]
    cfg["chunked"] = {"window_sec": 10.0, "hop_sec": 5.0, "stabilize_steps": 1}
    ctx = RunContext(tmp_path / "artifacts", "testrun_stream_v1")
    BenchmarkRunner(cfg, ctx, recordings).run()
    stream_files = list((ctx.run_dir / "predictions/asr").rglob("*.streaming.json"))
    assert len(stream_files) == 2  # one per language
    meta = load_json(stream_files[0])
    assert meta["mode"] == "simulated_chunked"
    assert meta["chunked_rtf"] is not None
