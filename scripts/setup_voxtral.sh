#!/usr/bin/env bash
# Create .venv-voxtral for the Voxtral Mini Realtime vLLM server (streaming ASR).
# Voxtral is Apache-2.0 & ungated — no HF token required. Needs a GPU >= ~16 GB.
#
# The Realtime WebSocket API (/v1/realtime) is recent, so we install the latest
# vLLM + transformers per the model card.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
"$PY" -m venv .venv-voxtral
.venv-voxtral/bin/pip install --upgrade pip
.venv-voxtral/bin/pip install --upgrade vllm transformers
.venv-voxtral/bin/pip install soxr librosa soundfile "mistral-common[audio]"

echo
echo ".venv-voxtral ready. Start the server (separate terminal):"
echo "  ./scripts/run_voxtral_server.sh"
echo "First launch downloads the model (~GBs) and compiles — give it a few minutes."
echo "Ready when the log prints:  Route: /v1/realtime"
