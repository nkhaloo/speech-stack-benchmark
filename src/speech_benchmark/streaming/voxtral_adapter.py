"""Fully-streaming stack: Voxtral Mini Realtime (native streaming ASR over
vLLM's WebSocket Realtime API) + diart online diarization.

Unlike the windowed stacks, nothing is re-transcribed: audio is streamed once
into Voxtral's ``/v1/realtime`` endpoint, which returns append-only
``transcription.delta`` events (<500 ms behind the audio). Each sentence is
emitted exactly once when its terminating punctuation arrives — so there is no
re-transcription churn, no sentence-duplication, and near-zero revision/flicker.
That is the whole point of a streaming ASR.

Speaker labels come from diart's online diarization run on the same audio; since
the realtime events carry no word timestamps, each delta is timestamped by the
audio position fed so far (good enough for sentence-level attribution at diart's
~0.5 s resolution).

Protocol (vLLM Realtime API, OpenAI-compatible):
  connect ws://host:port/v1/realtime
  -> recv {"type":"session.created"}
  <- {"type":"session.update","model":<model>}
  <- {"type":"input_audio_buffer.append","audio":<base64 pcm16@16k mono>}   (repeated)
  <- {"type":"input_audio_buffer.commit","final":true}                       (end)
  -> {"type":"transcription.delta","delta":<partial text>}                   (streamed)
  -> {"type":"transcription.done","text":<full text>}

Runs from .venv-diart (client: diart + websocket-client); needs the vLLM server
(scripts/run_voxtral_server.sh) reachable. Must be verified on the lab.
"""

from __future__ import annotations

import base64
import json

import numpy as np

from ..fusion.assign import SENT_FINAL
from ..schemas import Emission, Recording, SpeakerTurn
from .base import AdapterUnavailable, StreamingAdapter
from .online_diar import OnlineDiarizer


class VoxtralRealtimeStreamingAdapter(StreamingAdapter):
    def _load(self) -> None:
        try:
            import websocket  # websocket-client (sync)
        except ImportError as e:
            raise AdapterUnavailable(
                "websocket-client not installed in this env. "
                "pip install websocket-client (it's added to scripts/setup_diart.sh)."
            ) from e
        self._websocket = websocket

        vx = self.config.get("voxtral", {}) or {}
        self._ws_url = vx.get("ws_url", "ws://127.0.0.1:8000/v1/realtime")
        self._model = vx.get("model", "mistralai/Voxtral-Mini-4B-Realtime-2602")
        self._connect_timeout = float(vx.get("connect_timeout_sec", 10.0))
        # fail fast if the server isn't up
        try:
            probe = websocket.create_connection(self._ws_url, timeout=self._connect_timeout)
            probe.close()
        except Exception as e:
            raise AdapterUnavailable(
                f"Voxtral Realtime server not reachable at {self._ws_url}. "
                f"Start it with scripts/run_voxtral_server.sh. ({e})") from e

        self._diar = OnlineDiarizer(self.config.get("diart", {}))
        self._diar.load()
        self._sr = self._diar.sample_rate

    def reset(self, recording: Recording) -> None:
        self._diar.reset()
        self._buf = np.zeros(0, dtype=np.float32)
        self._ws = self._websocket.create_connection(
            self._ws_url, timeout=self._connect_timeout)
        # drain the initial session.created, then configure the session
        self._ws.settimeout(0.05)
        self._drain_raw()
        self._ws.send(json.dumps({"type": "session.update", "model": self._model}))
        self._sent_text = ""          # text already committed into emitted sentences
        self._pending = ""            # buffered delta text not yet a full sentence
        self._pending_start = None    # audio time when the pending sentence began
        self._next_id = 0

    def _unload(self) -> None:
        pass

    # -- streaming ----------------------------------------------------------
    def push(self, audio: np.ndarray, audio_time_end: float) -> list[Emission]:
        audio = audio.astype(np.float32)
        self._buf = np.concatenate([self._buf, audio])
        self._diar.feed(self._buf)
        # stream this frame to Voxtral as base64 PCM16
        pcm16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16).tobytes()
        self._ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm16).decode("ascii")}))
        return self._collect(audio_time_end, final=False)

    def flush(self) -> list[Emission]:
        total = len(self._buf) / self._sr
        self._diar.feed(self._buf)
        try:
            self._ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        except Exception:
            pass
        out = self._collect(total, final=True)
        # flush any trailing text that never got sentence-final punctuation
        if self._pending.strip():
            out.append(self._emit_sentence(self._pending, self._pending_start, total))
            self._pending = ""
        try:
            self._ws.close()
        except Exception:
            pass
        return out

    # -- internals ----------------------------------------------------------
    def _collect(self, audio_time_end: float, final: bool) -> list[Emission]:
        """Drain available realtime events; cut completed sentences and emit."""
        deadline_iters = 2000 if final else 1
        out: list[Emission] = []
        for _ in range(deadline_iters):
            got = self._drain_raw()
            if not got and not final:
                break
            for msg in got:
                mtype = msg.get("type")
                if mtype == "transcription.delta":
                    text = msg.get("delta", "")
                    if text and self._pending_start is None:
                        self._pending_start = audio_time_end
                    self._pending += text
                    out.extend(self._cut_sentences(audio_time_end))
                elif mtype == "transcription.done":
                    if final:
                        return out
                elif mtype == "error":
                    raise RuntimeError(f"voxtral realtime error: {msg.get('error')}")
            if final and not got:
                # give the model a moment to finish after commit
                import time as _t
                _t.sleep(0.05)
        return out

    def _drain_raw(self) -> list[dict]:
        msgs: list[dict] = []
        while True:
            try:
                raw = self._ws.recv()
            except self._websocket.WebSocketTimeoutException:
                break
            except Exception:
                break
            if not raw:
                break
            try:
                msgs.append(json.loads(raw))
            except (ValueError, TypeError):
                continue
        return msgs

    def _cut_sentences(self, audio_time_end: float) -> list[Emission]:
        out: list[Emission] = []
        while True:
            idx = _last_sentence_end(self._pending)
            if idx is None:
                break
            sentence = self._pending[:idx + 1].strip()
            self._pending = self._pending[idx + 1:]
            if sentence:
                out.append(self._emit_sentence(sentence, self._pending_start, audio_time_end))
            self._pending_start = audio_time_end if self._pending.strip() else None
        return out

    def _emit_sentence(self, text: str, start, end: float) -> Emission:
        start = start if start is not None else end
        speaker = self._speaker_at(start, end)
        e = Emission(sentence_id=self._next_id, text=text, speaker=speaker,
                     start=start, end=end, audio_time=end, is_final=True, revision=0)
        self._next_id += 1
        return e

    def _speaker_at(self, start: float, end: float):
        best, best_ov = None, 0.0
        for t in self._diar.current_turns():
            ov = min(end, t.end) - max(start, t.start)
            if ov > best_ov:
                best, best_ov = t.speaker, ov
        return best


def _last_sentence_end(text: str):
    """Index of the last sentence-final punctuation in text, or None."""
    idx = None
    for i, ch in enumerate(text):
        if ch in SENT_FINAL:
            idx = i
    return idx
