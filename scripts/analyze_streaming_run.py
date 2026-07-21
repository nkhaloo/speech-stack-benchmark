#!/usr/bin/env python3
"""Generate detailed diagnostics from an existing streaming run's cache."""

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from speech_benchmark.config import project_root
from speech_benchmark.streaming.diagnostics import generate_streaming_diagnostics

ROOT = project_root()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id")
    ap.add_argument("--run-dir")
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    args = ap.parse_args()
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.run_id:
        run_dir = Path(args.artifacts) / "runs" / args.run_id
    else:
        sys.exit("Provide --run-id or --run-dir")
    if not run_dir.exists():
        sys.exit(f"Run directory not found: {run_dir}")
    print(f"Diagnostics: {generate_streaming_diagnostics(run_dir)}")


if __name__ == "__main__":
    main()
