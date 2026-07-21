#!/usr/bin/env bash
# Run the resumable one-variable diart + WhisperLive tuning ladder on smoke.
set -euo pipefail
cd "$(dirname "$0")/.."

export STREAMING_CONFIG=configs/streaming_diart_whisperlive_tuning.yaml
export STREAMING_TAG=diart-whisperlive-tuning-v1
export STREAMING_FORCE=0

exec ./scripts/run_diart_whisperlive.sh smoke
