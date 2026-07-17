from pathlib import Path

from speech_benchmark.audio import duration_sec
from speech_benchmark.datasets.synthetic import prepare_synthetic_dataset
from speech_benchmark.schemas import load_json, load_manifest


def test_dummy_dataset_generation(dummy_dataset):
    out, recordings = dummy_dataset
    assert len(recordings) == 4  # 2 languages x 2 recordings
    langs = {r.language for r in recordings}
    assert langs == {"en", "zh"}
    for rec in recordings:
        assert Path(rec.audio_path).exists()
        ref = rec.load_reference()
        assert ref is not None and len(ref.turns) > 0
        assert ref.num_speakers == 2
        assert abs(duration_sec(rec.audio_path) - rec.duration_sec) < 0.1
        # turns are ordered, non-overlapping, with text
        for a, b in zip(ref.turns, ref.turns[1:]):
            assert b.start >= a.end
        assert all(t.text for t in ref.turns)
    manifest = load_manifest(out / "manifest.jsonl")
    assert len(manifest) == 4
    assert (out / "selection.json").exists()


def test_deterministic_generation(tmp_path):
    kwargs = dict(source_name="dummy", profile="det", languages=["en"],
                  minutes_per_language=0.5, recording_minutes=0.5,
                  speakers_range=(2, 2), seed=99)
    r1 = prepare_synthetic_dataset(out_dir=tmp_path / "a", **kwargs)
    r2 = prepare_synthetic_dataset(out_dir=tmp_path / "b", **kwargs)
    ref1 = load_json(r1[0].reference_path)
    ref2 = load_json(r2[0].reference_path)
    assert ref1["turns"] == ref2["turns"]
    assert r1[0].duration_sec == r2[0].duration_sec
