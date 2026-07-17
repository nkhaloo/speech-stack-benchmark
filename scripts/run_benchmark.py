#!/usr/bin/env python3
"""Run the benchmark for a track config against a prepared dataset manifest.

Resumable: re-running with the same --run-id skips completed work.

Examples:
  python scripts/run_benchmark.py --config configs/smoke.yaml \
      --manifest artifacts/datasets/synthetic_dummy/smoke/manifest.jsonl

  python scripts/run_benchmark.py --config configs/gpu.yaml --profile baseline
  python scripts/run_benchmark.py --config configs/gpu.yaml --profile baseline \
      --run-id 2026-07-17_gpu_baseline_v1        # resume that run
"""

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from speech_benchmark.benchmark import BenchmarkRunner, RunContext, make_run_id
from speech_benchmark.config import load_track_config, load_yaml, project_root
from speech_benchmark.reporting import generate_full_report
from speech_benchmark.schemas import load_manifest

ROOT = project_root()


def default_manifest(profile: str) -> Path:
    ds_cfg = load_yaml(ROOT / "configs/datasets/synthetic.yaml")
    source = ds_cfg.get("source", "commonvoice_mdc")
    return ROOT / "artifacts" / "datasets" / f"synthetic_{source}" / profile / "manifest.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="track config yaml (cpu/gpu/smoke)")
    ap.add_argument("--manifest", default=None, help="dataset manifest.jsonl")
    ap.add_argument("--profile", default="smoke",
                    choices=["smoke", "baseline", "extended"])
    ap.add_argument("--run-id", default=None,
                    help="run id (default: auto <date>_<track>_<profile>_v1); "
                         "reuse an id to resume")
    ap.add_argument("--tag", default="v1", help="suffix for auto run ids")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if cached outputs exist")
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    cfg = load_track_config(args.config)
    cfg["profile"] = args.profile
    manifest_path = Path(args.manifest) if args.manifest else default_manifest(args.profile)
    if not manifest_path.exists():
        sys.exit(f"Dataset manifest not found: {manifest_path}\n"
                 f"Run scripts/prepare_datasets.py --profile {args.profile} first.")
    recordings = load_manifest(manifest_path)
    if not recordings:
        sys.exit(f"No recordings in {manifest_path}")

    run_id = args.run_id or make_run_id(cfg.get("track", "cpu"), args.profile, args.tag)
    ctx = RunContext(args.artifacts, run_id)
    print(f"Run: {run_id}\nRun folder: {ctx.run_dir}")

    runner = BenchmarkRunner(
        cfg, ctx, recordings, force=args.force,
        config_paths=[args.config, str(manifest_path),
                      str(ROOT / "configs/datasets/synthetic.yaml")],
        cli_args=sys.argv[1:],
    )
    runner.run()

    if not args.no_report:
        generate_full_report(ctx.run_dir)
        print(f"Report: {ctx.run_dir / 'reports' / 'summary.md'}")


if __name__ == "__main__":
    main()
