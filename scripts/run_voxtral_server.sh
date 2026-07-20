#!/usr/bin/env bash
# Launch the local vLLM server hosting Voxtral Mini Realtime for the STREAMING
# ASR arm of the `voxtral-realtime` stack. Exposes vLLM's WebSocket Realtime API
# at ws://127.0.0.1:8000/v1/realtime (OpenAI-Realtime-compatible). Needs a GPU
# with >= ~16 GB VRAM. Voxtral is Apache-2.0 & ungated (no HF token needed).
#
# Set up .venv-voxtral first (see scripts/setup_voxtral.sh). Keep this running in
# a separate terminal; run the benchmark from .venv-diart (the client).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${VOXTRAL_MODEL:-mistralai/Voxtral-Mini-4B-Realtime-2602}"
PORT="${VOXTRAL_PORT:-8000}"

if [[ ! -x .venv-voxtral/bin/vllm ]]; then
  echo "No .venv-voxtral found. Create it first:  ./scripts/setup_voxtral.sh" >&2
  exit 1
fi

echo "Serving $MODEL (Realtime API) on port $PORT (Ctrl-C to stop)…"
echo "Watch the startup log for:  Route: /v1/realtime"
# VLLM_USE_FLASHINFER_SAMPLER=0: FlashInfer JIT-compiles a sampler kernel and
# needs the CUDA toolkit (nvcc); many boxes have only the driver/runtime. The
# native sampler needs no nvcc. Drop it if your box has a full CUDA toolkit.
exec env VLLM_DISABLE_COMPILE_CACHE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv-voxtral/bin/vllm serve "$MODEL" \
  --tokenizer-mode mistral \
  --compilation_config '{"cudagraph_mode":"PIECEWISE"}' \
  --port "$PORT"
