"""whisper.cpp adapter (CPU track) — shells out to the ``whisper-cli`` binary.

whisper.cpp is the reference for desktop CPU deployment (macOS/Windows,
quantized GGML weights, Metal acceleration on Apple Silicon). The adapter
runs the CLI with full JSON output and normalizes it.

Config keys:
  binary:      path to whisper-cli (default: "whisper-cli" on PATH)
  model_path:  path to a ggml .bin model file (required)
  threads:     CPU threads (default: os.cpu_count())
  extra_args:  list of additional CLI arguments
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ..audio import load_audio, write_wav
from ..schemas import ASRResult, ASRSegment, Recording, Word
from .base import AdapterUnavailable, ASRAdapter


class WhisperCppAdapter(ASRAdapter):
    def _load(self) -> None:
        self.binary = self.config.get("binary", "whisper-cli")
        if shutil.which(self.binary) is None and not Path(self.binary).exists():
            raise AdapterUnavailable(
                f"whisper.cpp binary not found: {self.binary!r}. "
                "Install with `brew install whisper-cpp` or build from source."
            )
        self.model_path = Path(self.config["model_path"])
        if not self.model_path.exists():
            raise AdapterUnavailable(
                f"ggml model not found: {self.model_path} "
                "(run scripts/download_models.py --track cpu)"
            )

    def transcribe(self, recording: Recording, language: Optional[str] = None) -> ASRResult:
        # whisper-cli wants 16 kHz mono wav; convert defensively.
        audio, sr = load_audio(recording.audio_path, 16000)
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "in.wav"
            write_wav(wav, audio, sr)
            out_base = Path(td) / "out"
            cmd = [
                self.binary,
                "-m", str(self.model_path),
                "-f", str(wav),
                "-l", language or "auto",
                "-t", str(self.config.get("threads") or os.cpu_count() or 4),
                "-ojf",  # full JSON incl. token timestamps
                "-of", str(out_base),
                "--no-prints",
            ] + list(self.config.get("extra_args", []))
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"whisper-cli failed (rc={proc.returncode}): {proc.stderr[-2000:]}"
                )
            with open(f"{out_base}.json", encoding="utf-8") as f:
                data = json.load(f)

        segments: list[ASRSegment] = []
        for seg in data.get("transcription", []):
            offsets = seg.get("offsets", {})
            start = float(offsets.get("from", 0)) / 1000.0
            end = float(offsets.get("to", 0)) / 1000.0
            words: list[Word] = []
            for tok in seg.get("tokens", []):
                text = tok.get("text", "")
                if text.startswith("[_"):  # control tokens like [_BEG_]
                    continue
                t_off = tok.get("offsets", {})
                words.append(Word(
                    text.strip(),
                    float(t_off.get("from", 0)) / 1000.0,
                    float(t_off.get("to", 0)) / 1000.0,
                    confidence=tok.get("p"),
                ))
            segments.append(ASRSegment(
                text=seg.get("text", "").strip(), start=start, end=end, words=words,
            ))

        detected = (data.get("result", {}) or {}).get("language")
        return ASRResult(
            recording_id=recording.recording_id,
            model_id=self.model_id,
            text=" ".join(s.text for s in segments),
            segments=segments,
            language_requested=language,
            language_detected=detected,
            model_meta=self.model_meta(),
        )
