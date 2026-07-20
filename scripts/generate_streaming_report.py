#!/usr/bin/env python3
"""Rebuild the streaming report (results_streaming.md) from a run's cached
streaming metrics. Pure post-processing — never loads a model.

  python scripts/generate_streaming_report.py --run-id <id>
  python scripts/generate_streaming_report.py --run-dir artifacts/runs/<id>
"""

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from speech_benchmark.config import project_root
from speech_benchmark.streaming.report import generate_streaming_report

ROOT = project_root()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.run_id:
        run_dir = Path(args.artifacts) / "runs" / args.run_id
    else:
        sys.exit("Provide --run-id or --run-dir")
    if not run_dir.exists():
        sys.exit(f"Run dir not found: {run_dir}")

    out = generate_streaming_report(run_dir)
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
