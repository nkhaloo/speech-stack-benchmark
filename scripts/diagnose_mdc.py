#!/usr/bin/env python3
"""Diagnose why an MDC Common Voice archive yields 0 eligible speakers.

Run on the machine that has the extracted archive:

    python scripts/diagnose_mdc.py --lang en
    python scripts/diagnose_mdc.py --lang en --split test

It mirrors CommonVoiceMDCSource.speakers() step by step and prints where rows
are being dropped (bad columns vs. missing audio) so we know what to fix.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from speech_benchmark.config import load_yaml, project_root
from speech_benchmark.datasets.sources import (CV_LANG, _find_cv_audio,
                                               _find_cv_tsv)

ROOT = project_root()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--split", default=None, help="override split from config")
    ap.add_argument("--config", default=str(ROOT / "configs/datasets/synthetic.yaml"))
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    sk = cfg.get("source_kwargs", {})
    dataset_ids = sk.get("dataset_ids", {})
    download_dir = ROOT / sk.get("download_dir", "artifacts/datasets/mdc")
    split = args.split or sk.get("split", "test")
    min_clips = int(sk.get("min_clips_per_speaker", 12))

    cv_lang = CV_LANG.get(args.lang, args.lang)
    dataset_id = dataset_ids.get(args.lang) or dataset_ids.get(cv_lang)
    root = download_dir / dataset_id
    print(f"lang={args.lang} cv_lang={cv_lang} dataset_id={dataset_id}")
    print(f"extract root: {root}  exists={root.exists()}")

    if not root.exists():
        print("!! extract dir does not exist — nothing was downloaded/extracted here.")
        return

    # What .tsv files are actually present?
    tsvs = sorted(root.rglob("*.tsv"))
    print(f"\nTSV files under root ({len(tsvs)}):")
    for p in tsvs:
        print(f"  {p.relative_to(root)}")

    tsv_path = _find_cv_tsv(root, split)
    print(f"\n_find_cv_tsv(split={split!r}) -> {tsv_path}")
    if tsv_path is None:
        print("!! no test/validated tsv found")
        return

    with tsv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        rows = list(reader)
    print(f"\nColumns: {fieldnames}")
    print(f"Total rows: {len(rows)}")
    if rows:
        print("First row (truncated):")
        for k, v in rows[0].items():
            print(f"   {k!r}: {str(v)[:80]!r}")

    # Replicate the filtering with per-reason counts.
    drop = Counter()
    by_speaker: dict[str, int] = Counter()
    sample_missing_audio: list[str] = []
    for row in rows:
        cid = (row.get("client_id") or "").strip()
        text = (row.get("sentence") or row.get("text") or "").strip()
        relpath = (row.get("path") or "").strip()
        if not cid:
            drop["no client_id"] += 1
            continue
        if not text:
            drop["no sentence/text"] += 1
            continue
        if not relpath:
            drop["no path"] += 1
            continue
        audio_path = _find_cv_audio(tsv_path.parent, root, relpath)
        if audio_path is None:
            drop["audio not found"] += 1
            if len(sample_missing_audio) < 5:
                sample_missing_audio.append(relpath)
            continue
        by_speaker[cid] += 1

    print(f"\nDropped rows by reason: {dict(drop)}")
    if sample_missing_audio:
        print("Sample unresolved audio paths:")
        for rp in sample_missing_audio:
            print(f"   {rp!r}")
        # Show where audio actually lives, to fix _find_cv_audio.
        audio_ext = {p.suffix for p in root.rglob("*") if p.is_file()
                     and p.suffix in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}}
        print(f"Audio extensions present under root: {audio_ext or 'NONE'}")
        clips_dirs = [p for p in root.rglob("clips") if p.is_dir()]
        print(f"'clips' dirs found: {[str(p.relative_to(root)) for p in clips_dirs]}")

    eligible = [c for c, n in by_speaker.items() if n >= min_clips]
    print(f"\nSpeakers total: {len(by_speaker)}")
    print(f"Speakers with >= {min_clips} clips: {len(eligible)}")
    if by_speaker:
        top = by_speaker.most_common(5)
        print(f"Top speakers by clip count: {top}")


if __name__ == "__main__":
    main()
