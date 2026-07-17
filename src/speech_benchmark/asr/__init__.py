"""ASR adapter registry."""

from __future__ import annotations

from .base import AdapterUnavailable, ASRAdapter

_REGISTRY = {
    "dummy": "speech_benchmark.asr.dummy.DummyASRAdapter",
    "faster_whisper": "speech_benchmark.asr.faster_whisper_adapter.FasterWhisperAdapter",
    "whisper_cpp": "speech_benchmark.asr.whisper_cpp_adapter.WhisperCppAdapter",
    "voxtral_vllm": "speech_benchmark.asr.voxtral_adapter.VoxtralClientAdapter",
}


def create_asr_adapter(model_config: dict) -> ASRAdapter:
    runtime = model_config.get("runtime")
    if runtime not in _REGISTRY:
        raise KeyError(
            f"Unknown ASR runtime {runtime!r} for model {model_config.get('id')!r}; "
            f"known: {sorted(_REGISTRY)}"
        )
    module_path, cls_name = _REGISTRY[runtime].rsplit(".", 1)
    import importlib

    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(model_config)


__all__ = ["ASRAdapter", "AdapterUnavailable", "create_asr_adapter"]
