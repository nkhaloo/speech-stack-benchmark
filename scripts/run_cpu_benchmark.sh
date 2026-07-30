#!/usr/bin/env bash
# CPU-track benchmark (desktop-class models). Default profile: baseline.
#
# The CPU track now runs in the STREAMING regime by default: natively streaming
# ASR (Nemotron 3.5, one multilingual checkpoint) with speaker labels attached
# from one whole-file diarization pass at end of session. See
# configs/cpu_streaming.yaml for how to read the numbers — notably that
# streaming_rtf covers the live path only, and that with a native decoder the
# one-off diarization pass is now the dominant cost.
#
# Usage:
#   ./scripts/run_cpu_benchmark.sh [profile] [extra args...]   # streaming (default)
#   ./scripts/run_cpu_benchmark.sh --batch [profile] [extra args...]
#
# --batch runs the original batch ASR x batch diarization track
# (configs/cpu.yaml). That baseline is what the streaming penalty is measured
# against, so keep it current: run it whenever the streaming arms change.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

MODE="streaming"
if [[ "${1:-}" == "--batch" ]]; then
  MODE="batch"; shift
elif [[ "${1:-}" == "--streaming" ]]; then
  shift
fi

PROFILE="${1:-baseline}"; shift || true

if [[ "$MODE" == "batch" ]]; then
  python scripts/run_benchmark.py \
    --config configs/cpu.yaml \
    --profile "$PROFILE" \
    "$@"
  python scripts/generate_report.py --track cpu
else
  python scripts/run_streaming_benchmark.py \
    --config configs/cpu_streaming.yaml \
    --profile "$PROFILE" \
    --tag cpustream \
    "$@"
fi
