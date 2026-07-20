#!/usr/bin/env bash
# Create .venv-diart for the native streaming diarization stack (diart_whisper).
#
# diart pulls an incompatible torch/torchaudio/torchvision by default (its deps
# resolve to a torch that breaks pyannote.audio 3.x import); we pin a compatible
# set. Also installs the project (editable) so `speech_benchmark` imports, and
# faster-whisper for the ASR arm.
#
# After setup you MUST, once:
#   1. accept terms on huggingface.co for BOTH gated models used by diart's
#      defaults:  pyannote/segmentation  and  pyannote/embedding
#   2. export HF_TOKEN=<your token>
# Then diart downloads them once and runs offline.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
"$PY" -m venv .venv-diart
.venv-diart/bin/pip install --upgrade pip

# Project (for speech_benchmark) + ASR arms.
#   faster-whisper -> diart_whisper stack; requests -> voxtral_realtime HTTP client.
.venv-diart/bin/pip install -e ".[dev]"
#   faster-whisper -> diart_whisper stack; requests + websocket-client -> voxtral client.
.venv-diart/bin/pip install faster-whisper requests websocket-client

# diart + a torch stack compatible with pyannote.audio 3.x (verified importable).
.venv-diart/bin/pip install diart
.venv-diart/bin/pip install "torch==2.2.2" "torchaudio==2.2.2" "torchvision==0.17.2"
# diart uses matplotlib.cm.get_cmap, removed in matplotlib 3.9.
.venv-diart/bin/pip install "matplotlib<3.9"

echo
.venv-diart/bin/python - <<'PY'
import diart
from diart import SpeakerDiarization, SpeakerDiarizationConfig  # noqa: F401
from diart.inference import StreamingInference  # noqa: F401
print("diart import OK")
PY

cat <<'EOF'

.venv-diart ready. This env runs BOTH native streaming stacks (it holds diart +
the project; the Voxtral arm is just an HTTP client to the vLLM server):
  * diart-whisper       (needs only this env)
  * voxtral-realtime    (also needs the vLLM server: scripts/run_voxtral_server.sh)

Before the first run (one time):
  * accept terms at https://huggingface.co/pyannote/segmentation
                and https://huggingface.co/pyannote/embedding
  * export HF_TOKEN=<your token>
  * verify readiness:  .venv-diart/bin/python scripts/check_streaming_env.py

Then contribute a native stack to a streaming run (same --run-id resumes/extends):
  # 1. enable it in configs/models/stream_diart_whisper.yaml (and/or
  #    stream_voxtral_realtime.yaml): set `enabled: true`
  # 2. run from this env against an existing run id:
  .venv-diart/bin/python scripts/run_streaming_benchmark.py \
      --config configs/streaming.yaml --profile baseline --run-id <existing_run_id>
EOF
