#!/usr/bin/env python3
"""Prepare benchmark datasets (runnable independently on the lab computer).

Examples:
  # Local smoke data with the offline dummy source (no downloads):
  python scripts/prepare_datasets.py --profile smoke --source dummy

  # Real data from Common Voice (needs [data] extra + HF login/terms):
  python scripts/prepare_datasets.py --profile baseline
  python scripts/prepare_datasets.py --profile baseline --languages en zh
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from speech_benchmark.config import load_yaml, project_root
from speech_benchmark.datasets.synthetic import prepare_synthetic_dataset

ROOT = project_root()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="smoke",
                    choices=["smoke", "baseline", "extended"])
    ap.add_argument("--source", default=None,
                    help="clip source: commonvoice_mdc (default), commonvoice, or dummy")
    ap.add_argument("--languages", nargs="*", default=None)
    ap.add_argument("--config", default=str(ROOT / "configs/datasets/synthetic.yaml"))
    ap.add_argument("--out", default=None, help="output dir override")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    profile_cfg = cfg["profiles"][args.profile]
    source = args.source or cfg.get("source", "commonvoice_mdc")
    languages = args.languages or cfg.get("languages", ["en", "es", "fr", "ar", "zh"])
    source_kwargs = cfg.get("source_kwargs", {}) if source == cfg.get("source") else {}

    out_dir = Path(args.out) if args.out else \
        ROOT / "artifacts" / "datasets" / f"synthetic_{source}" / args.profile

    print(f"Preparing {args.profile} dataset from source={source} "
          f"languages={languages} -> {out_dir}")
    recs = prepare_synthetic_dataset(
        out_dir=out_dir,
        source_name=source,
        profile=args.profile,
        languages=languages,
        minutes_per_language=float(profile_cfg["minutes_per_language"]),
        recording_minutes=float(profile_cfg["recording_minutes"]),
        speakers_range=tuple(profile_cfg["speakers_range"]),
        seed=int(cfg.get("seed", 20260717)),
        source_kwargs=source_kwargs,
    )
    total = sum(r.duration_sec or 0 for r in recs)
    print(f"Done: {len(recs)} recordings, {total/60:.1f} min total.")
    print(f"Manifest: {out_dir / 'manifest.jsonl'}")
    print(f"Selection record: {out_dir / 'selection.json'}")


if __name__ == "__main__":
    main()
