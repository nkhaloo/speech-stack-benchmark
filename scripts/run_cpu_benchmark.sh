#!/usr/bin/env bash
# CPU-track benchmark (desktop-class models). Default profile: baseline.
# Usage: ./scripts/run_cpu_benchmark.sh [profile] [extra run_benchmark.py args...]
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

PROFILE="${1:-baseline}"; shift || true

python scripts/run_benchmark.py \
  --config configs/cpu.yaml \
  --profile "$PROFILE" \
  "$@"

python scripts/generate_report.py --track cpu
