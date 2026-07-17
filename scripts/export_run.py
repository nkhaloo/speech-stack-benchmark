#!/usr/bin/env python3
"""Export a compact, portable results bundle for a run.

  python scripts/export_run.py --run-id 2026-07-17_gpu_baseline_v1 \
      --output artifacts/exports/2026-07-17_gpu_baseline_v1
  python scripts/export_run.py --run-id ... --include-predictions
"""

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from speech_benchmark.config import project_root
from speech_benchmark.reporting import export_run

ROOT = project_root()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", default=None,
                    help="default: artifacts/exports/<run-id>")
    ap.add_argument("--include-predictions", action="store_true",
                    help="add compressed raw predictions (predictions.tar.gz)")
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    args = ap.parse_args()

    run_dir = Path(args.artifacts) / "runs" / args.run_id
    if not (run_dir / "run_manifest.json").exists():
        sys.exit(f"Run not found: {run_dir}")
    out = Path(args.output) if args.output else \
        Path(args.artifacts) / "exports" / args.run_id
    export_run(run_dir, out, include_predictions=args.include_predictions)
    print(f"Exported to {out}")
    print("Copy back to the MacBook with e.g.:")
    print(f"  rsync -av <lab-host>:{out}/ ./artifacts/exports/{args.run_id}/")


if __name__ == "__main__":
    main()
