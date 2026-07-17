#!/usr/bin/env bash
# Minimal end-to-end smoke test: dummy data + dummy adapters, no downloads.
# Validates dataset generation, caching, fusion, metrics, and reporting.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

python scripts/prepare_datasets.py --profile smoke --source dummy
python scripts/run_benchmark.py \
  --config configs/smoke.yaml \
  --profile smoke \
  --manifest artifacts/datasets/synthetic_dummy/smoke/manifest.jsonl \
  --tag smoketest

echo
echo "Smoke test complete. Inspect:"
echo "  artifacts/latest/reports/summary.md"
