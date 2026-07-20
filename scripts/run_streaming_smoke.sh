#!/usr/bin/env bash
# Streaming smoke test: dummy data + dummy streaming stack, no downloads.
# Validates the streaming runner, emission cache, streaming metrics, and report.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

python scripts/prepare_datasets.py --profile smoke --source dummy
python scripts/run_streaming_benchmark.py \
  --config configs/streaming_smoke.yaml \
  --profile smoke \
  --manifest artifacts/datasets/synthetic_dummy/smoke/manifest.jsonl \
  --tag streamsmoke

echo
echo "Streaming smoke test complete. Inspect:"
echo "  artifacts/latest/reports/results_streaming.md"
