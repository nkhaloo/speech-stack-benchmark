from speech_benchmark.benchmark import BenchmarkRunner, RunContext
from speech_benchmark.reporting import export_run, generate_full_report
from speech_benchmark.schemas import load_json
from test_runner_e2e import TRACK_CFG


def test_export_bundle(dummy_dataset, tmp_path):
    _, recordings = dummy_dataset
    ctx = RunContext(tmp_path / "artifacts", "testrun_export_v1")
    BenchmarkRunner(TRACK_CFG, ctx, recordings).run()
    generate_full_report(ctx.run_dir)

    out = tmp_path / "export"
    export_run(ctx.run_dir, out, include_predictions=True)
    assert (out / "run_manifest.json").exists()
    assert (out / "reports" / "summary.md").exists()
    assert (out / "metrics" / "per_recording" / "pair_rows.csv").exists()
    assert (out / "predictions.tar.gz").exists()
    info = load_json(out / "export_info.json")
    assert info["run_id"] == "testrun_export_v1"
    # never exports weights/datasets
    assert not (out / "models").exists() and not (out / "datasets").exists()
