#!/usr/bin/env python3
"""Small GPU validation test for the Linux lab machine.

Checks: NVML GPU inventory, torch CUDA, a tiny CUDA tensor op, ctranslate2
CUDA support, and pyannote import. Exits non-zero on hard failures.
"""

import sys

import _bootstrap  # noqa: F401
from speech_benchmark.benchmark.resources import gpu_inventory, nvidia_driver_version

ok = True

gpus = gpu_inventory()
if gpus:
    print(f"NVIDIA driver: {nvidia_driver_version()}")
    for g in gpus:
        print(f"GPU {g['index']}: {g['name']} ({g['vram_total_mb']/1000:.1f} GB VRAM)")
else:
    print("!! No NVIDIA GPUs visible via NVML")
    ok = False

try:
    import torch

    print(f"torch {torch.__version__}, CUDA build: {torch.version.cuda}, "
          f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).sum().item()
        print(f"CUDA matmul OK (checksum {y:.1f})")
    else:
        ok = False
except Exception as e:
    print(f"!! torch check failed: {e}")
    ok = False

try:
    import ctranslate2

    n = ctranslate2.get_cuda_device_count()
    print(f"ctranslate2 {ctranslate2.__version__}, CUDA devices: {n}")
    if n == 0:
        ok = False
except Exception as e:
    print(f"!! ctranslate2 check failed: {e}")
    ok = False

try:
    import pyannote.audio

    print(f"pyannote.audio {pyannote.audio.__version__}")
except Exception as e:
    print(f"(warn) pyannote.audio import failed: {e}")

print("GPU validation:", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
