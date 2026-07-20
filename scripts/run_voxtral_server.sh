#!/usr/bin/env bash
# Launch the local vLLM server hosting Voxtral Mini Realtime for the streaming
# ASR arm of the `voxtral-realtime` stack. Runs in .venv-voxtral (created by
# scripts/setup_linux_gpu.sh --with-voxtral). Needs a GPU with >= ~16 GB VRAM.
#
# The benchmark's voxtral_realtime adapter is an HTTP client to this server; run
# the benchmark from .venv-diart (which has diart + the project), pointing at
# http://127.0.0.1:8000/v1. Keep this server running in a separate terminal.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${VOXTRAL_MODEL:-mistralai/Voxtral-Mini-4B-Realtime-2602}"   # Apache-2.0, ungated
PORT="${VOXTRAL_PORT:-8000}"

if [[ ! -x .venv-voxtral/bin/vllm ]]; then
  echo "No .venv-voxtral found. Create it first:" >&2
  echo "  ./scripts/setup_linux_gpu.sh --with-voxtral" >&2
  exit 1
fi

echo "Serving $MODEL on port $PORT (Ctrl-C to stop)…"
exec .venv-voxtral/bin/vllm serve "$MODEL" \
  --tokenizer-mode mistral \
  --config-format mistral \
  --load-format mistral \
  --port "$PORT"
