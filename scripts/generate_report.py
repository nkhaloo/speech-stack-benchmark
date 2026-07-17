#!/usr/bin/env python3
"""(Re)generate tables, charts, and summary.md for an existing run.

Operates entirely from saved results — never loads ASR/diarization models.

  python scripts/generate_report.py --run-id 2026-07-17_gpu_baseline_v1
  python scripts/generate_report.py --track gpu        # newest gpu run
"""

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from speech_benchmark.config import project_root
from speech_benchmark.reporting import generate_full_report

ROOT = project_root()


def find_latest_run(runs_dir: Path, track: str | None) -> Path | None:
    candidates = sorted((p for p in runs_dir.iterdir()
                         if (p / "run_manifest.json").exists()),
                        key=lambda p: p.name)
    if track:
        candidates = [p for p in candidates if f"_{track}_" in p.name]
    return candidates[-1] if candidates else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--track", default=None, help="pick newest run of this track")
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    args = ap.parse_args()

    runs_dir = Path(args.artifacts) / "runs"
    if args.run_id:
        run_dir = runs_dir / args.run_id
    else:
        run_dir = find_latest_run(runs_dir, args.track)
        if run_dir is None:
            sys.exit(f"No runs found in {runs_dir}")
    if not (run_dir / "run_manifest.json").exists():
        sys.exit(f"Run not found: {run_dir}")

    print(f"Generating report for {run_dir.name} ...")
    generate_full_report(run_dir)
    print(f"Report: {run_dir / 'reports' / 'summary.md'}")


if __name__ == "__main__":
    main()
