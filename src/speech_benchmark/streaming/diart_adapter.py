"""Native streaming diarization via diart, fused with windowed Whisper ASR.

diart runs an *online* speaker-diarization pipeline (segmentation + embedding +
incremental clustering) that keeps speaker identities stable as audio arrives —
unlike batch pyannote-3.1, which clusters a finished file. The ASR arm is a
locally-loaded faster-whisper windowed over the buffer. Shared feeding / fusion /
emission logic lives in ``OnlineDiarWindowedAdapter``.

Requires ``diart`` + gated pyannote weights in ``.venv-diart``
(``scripts/setup_diart.sh``). diart's defaults (pyannote/segmentation +
pyannote/embedding) differ from batch pyannote-3.1 — accuracy does not transfer.
Must be verified end-to-end on the lab machine (needs gated weights + a GPU).
"""

from __future__ import annotations

from ..asr import create_asr_adapter
from .online_base import OnlineDiarWindowedAdapter
from .windowed import _load_card


class DiartWhisperStreamingAdapter(OnlineDiarWindowedAdapter):
    def _build_asr(self):
        asr = create_asr_adapter(_load_card(self.config["asr"]))
        asr.load()
        return asr
