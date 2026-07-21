"""Small push-style client for Collabora WhisperLive's WebSocket protocol.

The upstream high-level client is oriented around microphones and files.  The
benchmark already owns the audio clock, so this client only opens a session,
sends float32 mono/16 kHz frames, and exposes the latest segment snapshot.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Optional

import numpy as np

from .base import AdapterUnavailable


class WhisperLiveClient:
    END_OF_AUDIO = b"END_OF_AUDIO"

    def __init__(self, config: dict):
        self.cfg = config or {}
        self.host = self.cfg.get("host", "127.0.0.1")
        self.port = int(os.environ.get("SPEECH_BENCHMARK_WHISPERLIVE_PORT",
                                       self.cfg.get("port", 9090)))
        self.ready_timeout = float(self.cfg.get("ready_timeout_sec", 60.0))
        self.flush_timeout = float(self.cfg.get("flush_timeout_sec", 20.0))

    def connect(self, language: Optional[str]) -> None:
        try:
            import websocket
        except ImportError as e:
            raise AdapterUnavailable(
                "websocket-client is required for WhisperLive; run "
                "scripts/setup_diart.sh."
            ) from e

        self._websocket = websocket
        self._uid = str(uuid.uuid4())
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._condition = threading.Condition()
        self._segments: list[dict] = []
        self._version = 0
        self._error: Optional[str] = None
        scheme = "wss" if self.cfg.get("use_wss", False) else "ws"
        url = f"{scheme}://{self.host}:{self.port}"
        self._options = {
            "uid": self._uid,
            "language": language,
            "task": "transcribe",
            "model": self.cfg.get("model", "large-v3-turbo"),
            "use_vad": bool(self.cfg.get("use_vad", True)),
            "send_last_n_segments": int(self.cfg.get("send_last_n_segments", 20)),
            "no_speech_thresh": float(self.cfg.get("no_speech_thresh", 0.45)),
            "clip_audio": bool(self.cfg.get("clip_audio", False)),
            "same_output_threshold": int(self.cfg.get("same_output_threshold", 5)),
            "word_timestamps": True,
        }
        self._ws = websocket.WebSocketApp(
            url, on_open=self._on_open, on_message=self._on_message,
            on_error=self._on_error, on_close=self._on_close)
        self._thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._thread.start()
        if not self._ready.wait(self.ready_timeout):
            self.close(send_end=False)
            detail = f": {self._error}" if self._error else ""
            raise AdapterUnavailable(
                f"WhisperLive did not become ready at {url}{detail}. Start its "
                "faster-whisper server with large-v3-turbo first."
            )
        if self._error:
            self.close(send_end=False)
            raise AdapterUnavailable(f"WhisperLive server error: {self._error}")

    def _on_open(self, ws) -> None:
        ws.send(json.dumps(self._options))

    def _on_message(self, _ws, payload: str) -> None:
        msg = json.loads(payload)
        if msg.get("uid") != self._uid:
            return
        if msg.get("status") == "ERROR":
            self._error = str(msg.get("message", "server error"))
            self._ready.set()
            return
        if msg.get("message") == "SERVER_READY":
            self._ready.set()
            return
        if "segments" in msg:
            with self._condition:
                self._segments = [dict(s) for s in msg["segments"]]
                self._version += 1
                self._condition.notify_all()

    def _on_error(self, _ws, error) -> None:
        self._error = str(error)
        self._ready.set()
        with self._condition:
            self._condition.notify_all()

    def _on_close(self, _ws, _code, _message) -> None:
        self._closed.set()
        with self._condition:
            self._condition.notify_all()

    def send(self, audio: np.ndarray) -> None:
        if self._error:
            raise RuntimeError(f"WhisperLive WebSocket error: {self._error}")
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("WhisperLive expects mono audio")
        self._ws.send(samples.tobytes(), opcode=self._websocket.ABNF.OPCODE_BINARY)

    def snapshot(self, wait_sec: float = 0.0) -> list[dict]:
        """Return the newest server snapshot, optionally waiting for an update."""
        with self._condition:
            version = self._version
            if wait_sec > 0 and not self._error and not self._closed.is_set():
                self._condition.wait_for(
                    lambda: self._version != version or self._error
                    or self._closed.is_set(), timeout=wait_sec)
            return [dict(s) for s in self._segments]

    def finish(self) -> list[dict]:
        self._ws.send(self.END_OF_AUDIO, opcode=self._websocket.ABNF.OPCODE_BINARY)
        deadline = time.monotonic() + self.flush_timeout
        last_version = -1
        stable_since = time.monotonic()
        while time.monotonic() < deadline and not self._closed.is_set():
            with self._condition:
                self._condition.wait(timeout=0.1)
                if self._version != last_version:
                    last_version = self._version
                    stable_since = time.monotonic()
            # Some WhisperLive releases do not close after END_OF_AUDIO.
            if last_version >= 0 and time.monotonic() - stable_since >= 0.5:
                break
        out = self.snapshot()
        self.close(send_end=False)
        return out

    def close(self, send_end: bool = False) -> None:
        ws = getattr(self, "_ws", None)
        if ws is None:
            return
        if send_end:
            try:
                ws.send(self.END_OF_AUDIO, opcode=self._websocket.ABNF.OPCODE_BINARY)
            except Exception:
                pass
        ws.close()
        thread = getattr(self, "_thread", None)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
