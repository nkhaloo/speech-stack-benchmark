"""Capture the execution environment for the run manifest."""

from __future__ import annotations

import os
import platform
import subprocess
import sys

import psutil

from .resources import gpu_inventory, nvidia_driver_version


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=30).stdout.strip() or None
    except Exception:
        return None


def git_commit() -> str | None:
    return _run(["git", "rev-parse", "HEAD"])


def cuda_version() -> str | None:
    try:
        import torch
        return torch.version.cuda
    except Exception:
        smi = _run(["nvidia-smi"])
        if smi and "CUDA Version:" in smi:
            return smi.split("CUDA Version:")[1].split()[0]
        return None


def pip_freeze() -> list[str]:
    out = _run([sys.executable, "-m", "pip", "freeze"])
    return out.splitlines() if out else []


def collect_environment() -> dict:
    vm = psutil.virtual_memory()
    gpus = gpu_inventory()
    return {
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "hostname": platform.node(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_mb": round(vm.total / 1e6),
        "gpus": gpus,
        "gpu_count": len(gpus),
        "nvidia_driver_version": nvidia_driver_version(),
        "cuda_version": cuda_version(),
        "git_commit": git_commit(),
        "env_markers": {
            k: os.environ.get(k) for k in
            ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "OMP_NUM_THREADS")
            if os.environ.get(k) is not None
        },
    }
