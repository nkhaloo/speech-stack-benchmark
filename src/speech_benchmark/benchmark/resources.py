"""Peak RAM / VRAM sampling around inference calls.

RAM: RSS of this process + children, sampled by a background thread.
VRAM: via NVML when available (Linux + NVIDIA); we record the peak *used*
memory on the configured device. On machines without NVML the VRAM fields
stay None — CPU and GPU reports are kept separate, so this is expected.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import psutil

from ..schemas import ResourceStats

try:
    import pynvml  # provided by nvidia-ml-py

    pynvml.nvmlInit()
    _NVML = True
except Exception:  # pragma: no cover - absent on macOS
    _NVML = False


class ResourceMonitor:
    """Context manager sampling peak RSS (and VRAM if available)."""

    def __init__(self, gpu_index: int = 0, interval_sec: float = 0.2):
        self.gpu_index = gpu_index
        self.interval = interval_sec
        self.stats = ResourceStats()

    def _sample(self) -> None:
        proc = psutil.Process()
        peak_ram = 0
        peak_vram = 0
        gpu_name = None
        handle = None
        if _NVML:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
                name = pynvml.nvmlDeviceGetName(handle)
                gpu_name = name.decode() if isinstance(name, bytes) else str(name)
            except Exception:
                handle = None
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except psutil.Error:
                        pass
                peak_ram = max(peak_ram, rss)
                if handle is not None:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    peak_vram = max(peak_vram, mem.used)
            except Exception:
                pass
            self._stop.wait(self.interval)
        self.stats.peak_ram_mb = round(peak_ram / 1e6, 1)
        if handle is not None:
            self.stats.peak_vram_mb = round(peak_vram / 1e6, 1)
            self.stats.gpu_name = gpu_name
            self.stats.gpu_index = self.gpu_index

    def __enter__(self) -> "ResourceMonitor":
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def gpu_inventory() -> list[dict]:
    """List available NVIDIA GPUs (empty on machines without NVML)."""
    if not _NVML:
        return []
    out = []
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out.append({
                "index": i,
                "name": name.decode() if isinstance(name, bytes) else str(name),
                "vram_total_mb": round(mem.total / 1e6),
            })
    except Exception:
        pass
    return out


def nvidia_driver_version() -> Optional[str]:
    if not _NVML:
        return None
    try:
        v = pynvml.nvmlSystemGetDriverVersion()
        return v.decode() if isinstance(v, bytes) else str(v)
    except Exception:
        return None
