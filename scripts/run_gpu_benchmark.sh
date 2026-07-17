#!/usr/bin/env bash
# GPU-track benchmark on the Linux lab machine. Default profile: baseline.
# Resumable: re-running continues an interrupted run (same day/tag => same id).
# Usage: ./scripts/run_gpu_benchmark.sh [profile] [extra run_benchmark.py args...]
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

PROFILE="${1:-baseline}"; shift || true

python scripts/run_benchmark.py \
  --config configs/gpu.yaml \
  --profile "$PROFILE" \
  "$@"

python scripts/generate_report.py --track gpu
