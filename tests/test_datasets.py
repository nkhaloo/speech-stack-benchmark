import csv
from pathlib import Path

import numpy as np

from speech_benchmark.audio import duration_sec, write_wav
from speech_benchmark.datasets.sources import CommonVoiceMDCSource
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


def test_commonvoice_mdc_reads_extracted_archive(tmp_path):
    dataset_id = "test-en-dataset"
    dataset_root = tmp_path / dataset_id / "cv-corpus-test-en"
    clips_dir = dataset_root / "clips"
    clips_dir.mkdir(parents=True)
    rows = []
    for speaker in ("alice", "bob"):
        for i in range(3):
            name = f"{speaker}-{i}.wav"
            write_wav(clips_dir / name, np.zeros(4000, dtype=np.float32))
            rows.append({"client_id": speaker, "path": name,
                         "sentence": f"sample {speaker} {i}"})
    with (dataset_root / "test.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["client_id", "path", "sentence"],
                                delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / dataset_id / ".complete").touch()

    source = CommonVoiceMDCSource(
        dataset_ids={"en": dataset_id}, download_dir=str(tmp_path),
        min_clips_per_speaker=3,
    )
    speakers = source.speakers("en", num_speakers=2, seed=7)
    assert sorted(speakers) == ["en_spk00", "en_spk01"]
    assert all(len(clips) == 3 for clips in speakers.values())
    assert all(clip.audio().dtype == np.float32
               for clips in speakers.values() for clip in clips)
