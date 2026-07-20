#!/usr/bin/env python3
"""Preflight check for the diart-whisper streaming stack. Reports exactly what is
ready and what is missing so you can get to a run on the lab machine with no
guesswork. Read-only; downloads nothing.

  .venv-diart/bin/python scripts/check_streaming_env.py
"""

import argparse
import os

import _bootstrap  # noqa: F401

OK, BAD, WARN = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[33m!\033[0m"


def check(label: str, fn):
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"{type(e).__name__}: {e}"
    mark = OK if ok is True else (WARN if ok is None else BAD)
    print(f"  {mark} {label}: {detail}")
    return ok


def _diart_import():
    import diart  # noqa: F401
    from diart import SpeakerDiarization  # noqa: F401
    return True, "diart importable"


def _torch_cuda():
    import torch
    if torch.cuda.is_available():
        return True, f"CUDA available ({torch.cuda.get_device_name(0)})"
    return None, "no CUDA — will run on CPU (slow; fine for a small smoke)"


def _hf_token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return (True, "HF_TOKEN set") if tok else (False, "HF_TOKEN not set (export it)")


def _gated(repo: str):
    def _fn():
        from huggingface_hub import HfApi
        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        HfApi().model_info(repo, token=tok)
        return True, f"{repo} accessible"
    return _fn


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    print("diart-whisper stack:")
    d_ok = check("diart install", _diart_import)
    check("GPU", _torch_cuda)
    t_ok = check("HF token", _hf_token)
    s_ok = check("pyannote/segmentation (gated)", _gated("pyannote/segmentation"))
    e_ok = check("pyannote/embedding (gated)", _gated("pyannote/embedding"))

    diart_ready = all([d_ok, t_ok, s_ok, e_ok])
    print("\nSummary:")
    print(f"  diart-whisper:    {'READY' if diart_ready else 'NOT READY'}")
    if not diart_ready:
        print("\n  → run scripts/setup_diart.sh, accept the two pyannote pages, "
              "export HF_TOKEN.")


if __name__ == "__main__":
    main()
