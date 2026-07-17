"""Clip sources for the synthetic-conversation builder.

A *source* yields, per language, a mapping ``speaker_id -> [Clip, ...]`` of
single-speaker utterances with verified transcripts. The builder concatenates
clips from several speakers into a conversation with exact reference turns.

Sources:
  * DummySource       — locally generated "speech-like" tones + codeword
                        transcripts; zero downloads; for plumbing/smoke tests.
  * CommonVoiceSource — Mozilla Common Voice (CC0), real speech in all five
                        target languages with per-clip speaker ids (client_id).
                        Requires `datasets` ([data] extra), a Hugging Face
                        account, and one-time acceptance of the CV terms.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

SR = 16000

# Language codes as used by Common Voice (zh uses zh-CN by default).
CV_LANG = {"en": "en", "es": "es", "fr": "fr", "ar": "ar", "zh": "zh-CN"}

# Codeword vocabulary for dummy transcripts, per language (a few real words so
# text normalization paths — Arabic script, Chinese chars — get exercised).
_DUMMY_VOCAB = {
    "en": "alpha bravo charlie delta echo foxtrot golf hotel india juliet".split(),
    "es": "casa perro gato libro mesa cielo tierra fuego agua viento".split(),
    "fr": "maison chien chat livre table ciel terre feu eau vent".split(),
    "ar": "بيت كلب قط كتاب طاولة سماء أرض نار ماء ريح".split(),
    "zh": list("房屋小狗小猫书桌天空土地火焰清水微风"),
}


@dataclass
class Clip:
    clip_id: str
    speaker_id: str
    text: str
    _loader: Callable[[], np.ndarray] = field(repr=False, default=None)  # type: ignore

    def audio(self) -> np.ndarray:
        return self._loader()


class DummySource:
    """Synthesizes distinguishable per-speaker audio: each speaker gets a base
    frequency; each 'word' is a short modulated tone burst followed by a gap.
    Not speech — only for exercising the pipeline with the dummy adapters."""

    name = "dummy"

    def __init__(self, clips_per_speaker: int = 40, words_per_clip: tuple[int, int] = (4, 10)):
        self.clips_per_speaker = clips_per_speaker
        self.words_per_clip = words_per_clip

    def speakers(self, language: str, num_speakers: int, seed: int) -> dict[str, list[Clip]]:
        vocab = _DUMMY_VOCAB.get(language, _DUMMY_VOCAB["en"])
        rng = random.Random(f"dummy:{language}:{seed}")
        out: dict[str, list[Clip]] = {}
        for s in range(num_speakers):
            spk = f"{language}_spk{s:02d}"
            f0 = 110.0 * (1.3 ** s)  # distinct base pitch per speaker
            clips = []
            for c in range(self.clips_per_speaker):
                n_words = rng.randint(*self.words_per_clip)
                if language == "zh":
                    text = "".join(rng.choice(vocab) for _ in range(n_words))
                else:
                    text = " ".join(rng.choice(vocab) for _ in range(n_words))
                clip_seed = rng.randint(0, 2**31)
                clips.append(Clip(
                    clip_id=f"{spk}_c{c:03d}", speaker_id=spk, text=text,
                    _loader=_make_tone_loader(f0, n_words, clip_seed),
                ))
            out[spk] = clips
        return out


def _make_tone_loader(f0: float, n_words: int, seed: int) -> Callable[[], np.ndarray]:
    def _load() -> np.ndarray:
        rng = np.random.default_rng(seed)
        pieces = []
        for _ in range(n_words):
            dur = rng.uniform(0.15, 0.35)
            t = np.arange(int(dur * SR)) / SR
            f = f0 * rng.uniform(0.9, 1.15)
            tone = 0.3 * np.sin(2 * np.pi * f * t) * np.hanning(len(t))
            gap = np.zeros(int(rng.uniform(0.05, 0.15) * SR))
            pieces.extend([tone, gap])
        return np.concatenate(pieces).astype(np.float32)

    return _load


class CommonVoiceSource:
    """Groups Common Voice test-split clips by speaker (client_id).

    Deterministic: speakers are ordered by a stable hash of client_id and the
    first ``num_speakers`` with enough clips are taken, so the same seed and
    dataset version reproduce the same selection. Selected clip ids are saved
    by the preparation step for the record.
    """

    name = "commonvoice"

    def __init__(self, dataset_name: str = "mozilla-foundation/common_voice_17_0",
                 split: str = "test", min_clips_per_speaker: int = 12):
        self.dataset_name = dataset_name
        self.split = split
        self.min_clips = min_clips_per_speaker

    def speakers(self, language: str, num_speakers: int, seed: int) -> dict[str, list[Clip]]:
        try:
            from datasets import Audio, load_dataset
        except ImportError as e:
            raise RuntimeError(
                "The `datasets` package is required for Common Voice preparation. "
                "Install with: uv pip install -e '.[data]'"
            ) from e

        cv_lang = CV_LANG.get(language, language)
        ds = load_dataset(self.dataset_name, cv_lang, split=self.split,
                          trust_remote_code=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=SR))

        by_speaker: dict[str, list[int]] = {}
        for i, cid in enumerate(ds["client_id"]):
            by_speaker.setdefault(cid, []).append(i)

        eligible = [cid for cid, idxs in by_speaker.items() if len(idxs) >= self.min_clips]
        # Stable pseudo-random order independent of dict/dataset ordering.
        eligible.sort(key=lambda cid: hashlib.sha256(f"{seed}:{cid}".encode()).hexdigest())
        chosen = eligible[:num_speakers]
        if len(chosen) < num_speakers:
            raise RuntimeError(
                f"Common Voice {cv_lang}/{self.split}: only {len(chosen)} speakers "
                f"with >= {self.min_clips} clips (need {num_speakers})"
            )

        out: dict[str, list[Clip]] = {}
        for s_idx, cid in enumerate(chosen):
            spk = f"{language}_spk{s_idx:02d}"
            clips = []
            for i in by_speaker[cid][:60]:  # cap per-speaker clips
                row_idx = i

                def _loader(row_idx: int = row_idx) -> np.ndarray:
                    arr = ds[row_idx]["audio"]["array"]
                    return np.asarray(arr, dtype=np.float32)

                clips.append(Clip(
                    clip_id=f"cv:{cv_lang}:{ds[row_idx].get('path') or row_idx}",
                    speaker_id=spk,
                    text=(ds[row_idx].get("sentence") or "").strip(),
                    _loader=_loader,
                ))
            out[spk] = [c for c in clips if c.text]
        return out


def get_source(name: str, **kwargs):
    if name == "dummy":
        return DummySource(**kwargs)
    if name == "commonvoice":
        return CommonVoiceSource(**kwargs)
    raise KeyError(f"unknown dataset source {name!r}")
