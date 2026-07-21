#!/usr/bin/env bash
# Create a separate WhisperLive server environment. Keeping it separate from
# diart avoids forcing their torch/pyannote dependency trees into one venv.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
"$PY" -m venv .venv-whisperlive
.venv-whisperlive/bin/pip install --upgrade pip
.venv-whisperlive/bin/pip install whisper-live

echo
echo ".venv-whisperlive ready. Run scripts/run_diart_whisperlive.sh <profile>."
