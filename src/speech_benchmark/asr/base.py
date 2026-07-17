"""Base class for ASR adapters.

Adapter contract:
  * ``load()`` acquires the model (once); ``unload()`` releases it.
  * ``transcribe(recording, language)`` returns an :class:`ASRResult` with
    text/segments/detected language filled in. The benchmark runner adds
    timing, resource stats, and status around the call.
  * Only one model is loaded at a time — the runner guarantees this.
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from typing import Optional

from ..schemas import ASRResult, Recording


class ASRAdapter(ABC):
    def __init__(self, model_config: dict):
        self.config = model_config
        self.model_id: str = model_config["id"]
        self._loaded = False

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    def unload(self) -> None:
        if self._loaded:
            self._unload()
            self._loaded = False
        gc.collect()
        try:  # release CUDA cache if torch happens to be present
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load(self) -> None:  # pragma: no cover - trivial default
        pass

    def _unload(self) -> None:  # pragma: no cover - trivial default
        pass

    # -- inference ----------------------------------------------------------
    @abstractmethod
    def transcribe(self, recording: Recording, language: Optional[str] = None) -> ASRResult:
        ...

    def model_meta(self) -> dict:
        keys = ("id", "family", "runtime", "model", "revision", "license",
                "weights_license", "device", "compute_type", "quantization")
        return {k: self.config.get(k) for k in keys if self.config.get(k) is not None}


class AdapterUnavailable(RuntimeError):
    """Raised when an adapter's runtime/deps are not installed in this env."""
