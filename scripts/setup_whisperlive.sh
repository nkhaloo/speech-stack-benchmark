#!/usr/bin/env bash
# Create a separate WhisperLive server environment. Keeping it separate from
# diart avoids forcing their torch/pyannote dependency trees into one venv.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
"$PY" -m venv .venv-whisperlive
.venv-whisperlive/bin/pip install --upgrade pip
# The benchmark sends raw PCM itself, so WhisperLive's microphone-only PyAudio
# dependency is intentionally omitted (it otherwise needs system PortAudio
# headers). OpenVINO dependencies are also unnecessary for faster-whisper.
.venv-whisperlive/bin/pip install whisper-live --no-deps
.venv-whisperlive/bin/pip install \
  "faster-whisper==1.2.0" "numpy==1.26.4" "tokenizers==0.20.3" \
  torch torchaudio websockets websocket-client scipy soundfile librosa \
  onnxruntime numba kaldialign fastapi uvicorn python-multipart \
  "openai-whisper==20250625" nvidia-cublas-cu12 nvidia-cudnn-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12

echo
echo ".venv-whisperlive ready. Run scripts/run_diart_whisperlive.sh <profile>."
