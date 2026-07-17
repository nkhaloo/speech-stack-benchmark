#!/usr/bin/env bash
# MacBook development setup: isolated env, dev deps, unit tests, dummy smoke test.
# Usage: ./scripts/setup_macos.sh [--with-cpu-models]
#   --with-cpu-models  also install the CPU model runtimes (faster-whisper,
#                      sherpa-onnx) for real local CPU testing.
set -euo pipefail
cd "$(dirname "$0")/.."

WITH_CPU=0
for arg in "$@"; do
  case "$arg" in
    --with-cpu-models) WITH_CPU=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

echo "== Checking prerequisites =="
if [[ "$(uname)" != "Darwin" ]]; then
  echo "This script is for macOS. Use scripts/setup_linux_gpu.sh on the lab machine." >&2
  exit 1
fi

# Prefer uv (fast, manages its own Pythons); fall back to python3 -m venv.
if command -v uv >/dev/null 2>&1; then
  echo "using uv $(uv --version)"
  uv venv --python 3.12 .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  EXTRAS="dev"
  [[ $WITH_CPU -eq 1 ]] && EXTRAS="dev,cpu"
  uv pip install -e ".[${EXTRAS}]"
else
  PY=""
  for cand in python3.12 python3.11 python3.10; do
    command -v "$cand" >/dev/null 2>&1 && { PY="$cand"; break; }
  done
  if [[ -z "$PY" ]]; then
    echo "Need Python >= 3.10 (or install uv: https://docs.astral.sh/uv/)." >&2
    exit 1
  fi
  "$PY" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  EXTRAS="dev"
  [[ $WITH_CPU -eq 1 ]] && EXTRAS="dev,cpu"
  pip install -e ".[${EXTRAS}]"
fi

echo
echo "== Running unit tests =="
python -m pytest

echo
echo "== Running minimal smoke test (dummy models, no downloads) =="
./scripts/run_smoke_test.sh

cat <<'EOF'

macOS setup complete.
  Activate the env:        source .venv/bin/activate
  Unit tests:              python -m pytest
  Smoke test:              ./scripts/run_smoke_test.sh
  Real CPU-track models:   ./scripts/setup_macos.sh --with-cpu-models
                           (whisper.cpp binary: brew install whisper-cpp)
EOF
