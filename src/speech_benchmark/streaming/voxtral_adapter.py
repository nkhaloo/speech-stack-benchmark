"""Fully-streaming stack: Voxtral Mini Realtime (streaming ASR via local vLLM) +
diart online diarization.

The ASR arm talks to a local vLLM server hosting ``Voxtral-Mini-4B-Realtime-2602``
(Apache-2.0, ungated, 13 languages incl. ar+zh — the batch Voxtral-Mini-3B-2507
does NOT cover ar/zh). Two ASR modes:

  * ``chunked`` (default, runs today): the buffer is re-transcribed over a sliding
    window via vLLM's OpenAI-compatible ``/audio/transcriptions`` — the same
    battle-tested path as the batch Voxtral adapter. Gives Voxtral *accuracy*
    numbers but not its native <500 ms latency.
  * ``native`` (TODO on lab): a true incremental streaming client against
    Voxtral Realtime's low-latency interface. Not yet implemented — the exact
    vLLM streaming-transcription protocol must be confirmed on the lab machine
    (use ``scripts/check_streaming_env.py`` to probe the server). Selecting it
    raises ``AdapterUnavailable`` until implemented.

diart arm + fusion + emission logic are shared via ``OnlineDiarWindowedAdapter``.
Needs the vLLM server (``.venv-voxtral``) reachable and diart + gated pyannote
weights in the client env (``.venv-diart``).
"""

from __future__ import annotations

from ..asr import create_asr_adapter
from .base import AdapterUnavailable
from .online_base import OnlineDiarWindowedAdapter


class VoxtralRealtimeStreamingAdapter(OnlineDiarWindowedAdapter):
    def _build_asr(self):
        vx = self.config.get("voxtral", {}) or {}
        mode = vx.get("mode", "chunked")
        if mode == "native":
            raise AdapterUnavailable(
                "voxtral mode 'native' (true streaming client) not yet "
                "implemented — set mode: chunked to run now, or implement the "
                "incremental vLLM streaming client on the lab machine "
                "(scripts/check_streaming_env.py probes the server API).")
        if mode != "chunked":
            raise AdapterUnavailable(f"unknown voxtral mode {mode!r}")
        card = {
            "id": self.model_id + "-asr",
            "runtime": "voxtral_vllm",
            "endpoint": vx.get("endpoint", "http://127.0.0.1:8000/v1"),
            "model": vx.get("model", "mistralai/Voxtral-Mini-4B-Realtime-2602"),
            "timeout_sec": int(vx.get("timeout_sec", 120)),
        }
        asr = create_asr_adapter(card)
        asr.load()  # verifies the vLLM server is reachable
        return asr
