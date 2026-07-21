#!/usr/bin/env python3
"""Minimal local WhisperLive server launcher for the focused benchmark."""

import argparse

from whisper_live.server import TranscriptionServer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument(
        "--model", default="deepdml/faster-whisper-large-v3-turbo-ct2",
        help="Local CTranslate2 model directory or Hugging Face repository ID",
    )
    args = ap.parse_args()
    TranscriptionServer().run(
        "127.0.0.1", port=args.port, backend="faster_whisper",
        faster_whisper_custom_model_path=args.model,
        single_model=True, max_clients=1, max_connection_time=3600,
    )


if __name__ == "__main__":
    main()
