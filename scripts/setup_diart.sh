#!/usr/bin/env bash
# Create .venv-diart for the native streaming diarization stack (diart_whisper).
#
# diart pulls an incompatible torch/torchaudio/torchvision by default (its deps
# resolve to a torch that breaks pyannote.audio 3.x import); we pin a compatible
# set. Also installs the project (editable) so `speech_benchmark` imports, and
# websocket-client for the WhisperLive ASR arm (the server has its own env).
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

# Project (for speech_benchmark) + the diart_whisper ASR arm (faster-whisper).
# websocket-client is kept for streaming-ASR clients (e.g. WhisperLive).
.venv-diart/bin/pip install -e ".[dev]"
.venv-diart/bin/pip install requests websocket-client

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

.venv-diart ready (holds diart + the benchmark's WhisperLive client).

Before the first run (one time):
  * accept terms at https://huggingface.co/pyannote/segmentation
                and https://huggingface.co/pyannote/embedding
  * export HF_TOKEN=<your token>
  * verify readiness:  .venv-diart/bin/python scripts/check_streaming_env.py

Then set up and run WhisperLive:
  ./scripts/setup_whisperlive.sh
  ./scripts/run_diart_whisperlive.sh baseline
EOF
