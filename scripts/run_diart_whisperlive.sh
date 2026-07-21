#!/usr/bin/env bash
# Run the focused diart + WhisperLive/large-v3-turbo dataset test.
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE="${1:-baseline}"
MANIFEST="artifacts/datasets/synthetic_commonvoice_mdc/$PROFILE/manifest.jsonl"
PORT="${WHISPERLIVE_PORT:-9090}"
CONFIG="${STREAMING_CONFIG:-configs/streaming_diart_whisperlive.yaml}"
RUN_TAG="${STREAMING_TAG:-diart-whisperlive-turbo}"
FORCE="${STREAMING_FORCE:-1}"
export SPEECH_BENCHMARK_WHISPERLIVE_PORT="$PORT"

if [[ ! -x .venv-whisperlive/bin/python ]]; then
  echo "Missing .venv-whisperlive; run scripts/setup_whisperlive.sh first." >&2
  exit 1
fi
if [[ ! -x .venv-diart/bin/python ]]; then
  echo "Missing .venv-diart; run scripts/setup_diart.sh first." >&2
  exit 1
fi
if [[ -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]]; then
  echo "Missing Hugging Face token; export HF_TOKEN with a valid replacement token." >&2
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing dataset: $MANIFEST" >&2
  echo "Prepare it first: python scripts/prepare_datasets.py --profile $PROFILE" >&2
  exit 1
fi

# CTranslate2/faster-whisper wheels use CUDA 12 even when pip's current torch
# wheel pulls CUDA 13. The cu12 wheels install without root; expose their shared
# libraries to the WhisperLive server process.
WHISPERLIVE_SITE="$(
  .venv-whisperlive/bin/python -c 'import site; print(site.getsitepackages()[0])'
)"
WHISPERLIVE_CUDA_LIBS="$(
  find "${WHISPERLIVE_SITE}/nvidia" -type f -name '*.so*' -printf '%h\n' 2>/dev/null \
    | sort -u | paste -sd: -
)"
export LD_LIBRARY_PATH="${WHISPERLIVE_CUDA_LIBS}:${LD_LIBRARY_PATH:-}"

# Fail before loading diart if CTranslate2's required CUDA library is not
# actually resolvable in this environment.
.venv-whisperlive/bin/python -c \
  "import ctypes; ctypes.CDLL('libcublas.so.12'); print('CUDA 12 runtime ready')"

.venv-whisperlive/bin/python scripts/run_whisperlive_server.py \
  --port "$PORT" --model deepdml/faster-whisper-large-v3-turbo-ct2 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT INT TERM

# The WebSocket adapter performs the definitive readiness check. A short pause
# only avoids racing the server process before it binds the port.
sleep 2

RUN_ARGS=(
  scripts/run_streaming_benchmark.py
  --config "$CONFIG"
  --profile "$PROFILE"
  --manifest "$MANIFEST"
  --tag "$RUN_TAG"
)
if [[ "$FORCE" == "1" ]]; then
  RUN_ARGS+=(--force)
fi
.venv-diart/bin/python "${RUN_ARGS[@]}"
