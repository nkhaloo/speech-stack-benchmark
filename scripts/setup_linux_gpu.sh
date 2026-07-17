#!/usr/bin/env bash
# Linux GPU lab setup: verify NVIDIA environment, create isolated env(s),
# install pinned GPU deps, prepare directories, run a GPU validation test.
# No root required.
#
# Usage: ./scripts/setup_linux_gpu.sh [--with-pyannote4] [--with-voxtral] [--with-nemo]
#   --with-pyannote4  extra env (.venv-pyannote4) for pyannote community-1
#   --with-voxtral    extra env (.venv-voxtral) with vLLM for Voxtral Mini
#   --with-nemo       extra env (.venv-nemo) for NeMo (Sortformer — CC-BY-NC,
#                     reference comparison only!)
set -euo pipefail
cd "$(dirname "$0")/.."

WITH_PYANNOTE4=0; WITH_VOXTRAL=0; WITH_NEMO=0
for arg in "$@"; do
  case "$arg" in
    --with-pyannote4) WITH_PYANNOTE4=1 ;;
    --with-voxtral)   WITH_VOXTRAL=1 ;;
    --with-nemo)      WITH_NEMO=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

echo "== Verifying Linux + NVIDIA environment =="
if [[ "$(uname)" != "Linux" ]]; then
  echo "This script is for Linux." >&2; exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found — NVIDIA driver missing or not on PATH." >&2; exit 1
fi
echo "-- GPUs --"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo

# uv if available (no root needed to install: curl -LsSf https://astral.sh/uv/install.sh | sh)
if command -v uv >/dev/null 2>&1; then
  PYTOOL="uv"
  echo "using uv $(uv --version)"
else
  PYTOOL="venv"
  PY=""
  for cand in python3.12 python3.11 python3.10; do
    command -v "$cand" >/dev/null 2>&1 && { PY="$cand"; break; }
  done
  [[ -z "$PY" ]] && { echo "Need Python >= 3.10 or uv." >&2; exit 1; }
fi

make_env () {  # make_env <dir> <extras...>
  local dir="$1"; shift
  echo "== Creating environment $dir =="
  if [[ "$PYTOOL" == "uv" ]]; then
    uv venv --python 3.12 "$dir"
    VIRTUAL_ENV="$PWD/$dir" uv pip install -e ".[$*]"
  else
    "$PY" -m venv "$dir"
    "$dir/bin/pip" install --upgrade pip
    "$dir/bin/pip" install -e ".[$*]"
  fi
}

# Main environment: framework + faster-whisper + pyannote.audio 3.x (CUDA torch).
make_env .venv "dev,gpu,data"

echo "== Preparing model/dataset directories =="
mkdir -p artifacts/{models,datasets,cache,runs,exports}

echo "== GPU validation test =="
.venv/bin/python scripts/validate_gpu.py

if [[ $WITH_PYANNOTE4 -eq 1 ]]; then
  echo "== Extra env: pyannote.audio 4.x (community-1) =="
  if [[ "$PYTOOL" == "uv" ]]; then
    uv venv --python 3.12 .venv-pyannote4
    VIRTUAL_ENV="$PWD/.venv-pyannote4" uv pip install -e ".[dev]" "pyannote.audio>=4.0" torch
  else
    "$PY" -m venv .venv-pyannote4
    .venv-pyannote4/bin/pip install --upgrade pip
    .venv-pyannote4/bin/pip install -e ".[dev]" "pyannote.audio>=4.0" torch
  fi
  echo "Run community-1 with: .venv-pyannote4/bin/python scripts/run_benchmark.py ..."
fi

if [[ $WITH_VOXTRAL -eq 1 ]]; then
  echo "== Extra env: vLLM for Voxtral Mini =="
  if [[ "$PYTOOL" == "uv" ]]; then
    uv venv --python 3.12 .venv-voxtral
    VIRTUAL_ENV="$PWD/.venv-voxtral" uv pip install "vllm>=0.10" "mistral-common[audio]"
  else
    "$PY" -m venv .venv-voxtral
    .venv-voxtral/bin/pip install --upgrade pip
    .venv-voxtral/bin/pip install "vllm>=0.10" "mistral-common[audio]"
  fi
  cat <<'EOF'
Launch the local Voxtral server before enabling the voxtral-mini-vllm model:
  .venv-voxtral/bin/vllm serve mistralai/Voxtral-Mini-3B-2507 \
    --tokenizer-mode mistral --config-format mistral --load-format mistral
EOF
fi

if [[ $WITH_NEMO -eq 1 ]]; then
  echo "== Extra env: NeMo (Sortformer — NON-COMMERCIAL weights, reference only) =="
  if [[ "$PYTOOL" == "uv" ]]; then
    uv venv --python 3.12 .venv-nemo
    VIRTUAL_ENV="$PWD/.venv-nemo" uv pip install -e ".[dev]" "nemo_toolkit[asr]>=2.0"
  else
    "$PY" -m venv .venv-nemo
    .venv-nemo/bin/pip install --upgrade pip
    .venv-nemo/bin/pip install -e ".[dev]" "nemo_toolkit[asr]>=2.0"
  fi
fi

cat <<'EOF'

Linux GPU setup complete. Typical workflow:
  source .venv/bin/activate
  export HF_TOKEN=...                                   # for gated pyannote
  python scripts/prepare_datasets.py --profile baseline
  python scripts/download_models.py --track gpu
  ./scripts/run_gpu_benchmark.sh
  python scripts/generate_report.py --track gpu
  python scripts/export_run.py --run-id <run-id>
EOF
