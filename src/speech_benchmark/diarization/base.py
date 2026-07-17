"""Base class for speaker-diarization adapters. Same contract as ASR:
load once, process recordings one at a time, unload before the next model."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod

from ..schemas import DiarizationResult, Recording


class DiarizationAdapter(ABC):
    def __init__(self, model_config: dict):
        self.config = model_config
        self.model_id: str = model_config["id"]
        self._loaded = False

    def load(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    def unload(self) -> None:
        if self._loaded:
            self._unload()
            self._loaded = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load(self) -> None:  # pragma: no cover
        pass

    def _unload(self) -> None:  # pragma: no cover
        pass

    @abstractmethod
    def diarize(self, recording: Recording) -> DiarizationResult:
        ...

    def model_meta(self) -> dict:
        keys = ("id", "family", "runtime", "model", "revision", "license",
                "weights_license", "device")
        return {k: self.config.get(k) for k in keys if self.config.get(k) is not None}
