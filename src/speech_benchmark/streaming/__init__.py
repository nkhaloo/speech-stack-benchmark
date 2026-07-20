"""Streaming stack adapter registry.

A *streaming stack* consumes audio incrementally and emits speaker-attributed
sentences (see docs/methodology_streaming.md). Cards are referenced by
``runtime:`` exactly like batch model cards.
"""

from __future__ import annotations

from .base import AdapterUnavailable, StreamingAdapter

_REGISTRY = {
    "dummy_stream": "speech_benchmark.streaming.dummy.DummyStreamingAdapter",
    "windowed_stack": "speech_benchmark.streaming.windowed.WindowedStackStreamingAdapter",
    "diart_whisper": "speech_benchmark.streaming.diart_adapter.DiartWhisperStreamingAdapter",
}


def create_streaming_adapter(model_config: dict) -> StreamingAdapter:
    runtime = model_config.get("runtime")
    if runtime not in _REGISTRY:
        raise KeyError(
            f"Unknown streaming runtime {runtime!r} for stack "
            f"{model_config.get('id')!r}; known: {sorted(_REGISTRY)}"
        )
    module_path, cls_name = _REGISTRY[runtime].rsplit(".", 1)
    import importlib

    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(model_config)


__all__ = ["StreamingAdapter", "AdapterUnavailable", "create_streaming_adapter"]
