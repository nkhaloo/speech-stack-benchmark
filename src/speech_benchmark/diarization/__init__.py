"""Diarization adapter registry."""

from __future__ import annotations

from .base import DiarizationAdapter

_REGISTRY = {
    "dummy": "speech_benchmark.diarization.dummy.DummyDiarizationAdapter",
    "pyannote": "speech_benchmark.diarization.pyannote_adapter.PyannoteAdapter",
    "sherpa_onnx": "speech_benchmark.diarization.sherpa_onnx_adapter.SherpaOnnxDiarizationAdapter",
    "nemo_sortformer": "speech_benchmark.diarization.nemo_sortformer_adapter.NemoSortformerAdapter",
}


def create_diarization_adapter(model_config: dict) -> DiarizationAdapter:
    runtime = model_config.get("runtime")
    if runtime not in _REGISTRY:
        raise KeyError(
            f"Unknown diarization runtime {runtime!r} for model "
            f"{model_config.get('id')!r}; known: {sorted(_REGISTRY)}"
        )
    module_path, cls_name = _REGISTRY[runtime].rsplit(".", 1)
    import importlib

    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(model_config)


__all__ = ["DiarizationAdapter", "create_diarization_adapter"]
