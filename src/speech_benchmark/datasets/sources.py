"""Clip sources for the synthetic-conversation builder.

A *source* yields, per language, a mapping ``speaker_id -> [Clip, ...]`` of
single-speaker utterances with verified transcripts. The builder concatenates
clips from several speakers into a conversation with exact reference turns.

Sources:
  * DummySource       — locally generated "speech-like" tones + codeword
                        transcripts; zero downloads; for plumbing/smoke tests.
  * CommonVoiceMDCSource — Mozilla Common Voice (CC0), real speech in all five
                           target languages with per-clip speaker ids. Uses
                           Mozilla Data Collective's official download client.
  * CommonVoiceSource    — legacy Hugging Face loader retained for old local
                           caches; Mozilla retired those hosted datasets.
"""

from __future__ import annotations

import csv
import hashlib
import random
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..audio import load_audio

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


class CommonVoiceMDCSource:
    """Common Voice archives hosted by Mozilla Data Collective (MDC).

    MDC distributes one archive per locale.  Archives are downloaded with the
    official ``datacollective`` SDK, cached, and extracted once.  The source
    then reads Common Voice's TSV metadata directly instead of loading the
    retired Hugging Face dataset repositories.
    """

    name = "commonvoice_mdc"

    def __init__(
        self,
        dataset_ids: dict[str, str],
        download_dir: str = "artifacts/datasets/mdc",
        split: str = "test",
        min_clips_per_speaker: int = 12,
    ):
        self.dataset_ids = dataset_ids
        self.download_dir = Path(download_dir)
        self.split = split
        self.min_clips = min_clips_per_speaker

    def _dataset_root(self, language: str) -> Path:
        cv_lang = CV_LANG.get(language, language)
        dataset_id = self.dataset_ids.get(language) or self.dataset_ids.get(cv_lang)
        if not dataset_id:
            raise RuntimeError(f"No MDC dataset id configured for language {language!r}")

        extract_dir = self.download_dir / dataset_id
        if (extract_dir / ".complete").exists() and \
                _find_cv_tsv(extract_dir, self.split) is not None:
            return extract_dir

        try:
            from datacollective import download_dataset
        except ImportError as e:
            raise RuntimeError(
                "The `datacollective` package is required for Mozilla Data "
                "Collective downloads. Install with: pip install -e '.[data]'"
            ) from e

        self.download_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Common Voice {cv_lang} from MDC ({dataset_id})")
        try:
            archive = Path(download_dataset(
                dataset_id, download_directory=str(self.download_dir)
            ))
        except PermissionError as e:
            raise RuntimeError(
                f"MDC denied access to {dataset_id}. Set MDC_API_KEY and accept "
                "this dataset's conditions on mozilladatacollective.com."
            ) from e

        print(f"Extracting {archive.name} -> {extract_dir}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(extract_dir, filter="data")
        except Exception:
            # An incomplete extraction must not look reusable on the next run.
            marker = extract_dir / ".complete"
            marker.unlink(missing_ok=True)
            raise
        (extract_dir / ".complete").touch()
        return extract_dir

    def speakers(self, language: str, num_speakers: int, seed: int) -> dict[str, list[Clip]]:
        cv_lang = CV_LANG.get(language, language)
        root = self._dataset_root(language)
        tsv_path = _find_cv_tsv(root, self.split)
        if tsv_path is None:
            raise RuntimeError(
                f"MDC archive for {cv_lang} contains neither {self.split}.tsv "
                f"nor validated.tsv under {root}"
            )

        by_speaker: dict[str, list[dict[str, str]]] = {}
        with tsv_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                cid = (row.get("client_id") or "").strip()
                text = (row.get("sentence") or row.get("text") or "").strip()
                relpath = (row.get("path") or "").strip()
                if not cid or not text or not relpath:
                    continue
                audio_path = _find_cv_audio(tsv_path.parent, root, relpath)
                if audio_path is not None:
                    row["_audio_path"] = str(audio_path)
                    row["_text"] = text
                    by_speaker.setdefault(cid, []).append(row)

        eligible = [cid for cid, rows in by_speaker.items()
                    if len(rows) >= self.min_clips]
        eligible.sort(key=lambda cid: hashlib.sha256(
            f"{seed}:{cid}".encode()).hexdigest())
        chosen = eligible[:num_speakers]
        if len(chosen) < num_speakers:
            raise RuntimeError(
                f"Common Voice MDC {cv_lang}/{tsv_path.name}: only "
                f"{len(chosen)} speakers with >= {self.min_clips} clips "
                f"(need {num_speakers})"
            )

        out: dict[str, list[Clip]] = {}
        for s_idx, cid in enumerate(chosen):
            spk = f"{language}_spk{s_idx:02d}"
            clips: list[Clip] = []
            for row in by_speaker[cid][:60]:
                audio_path = Path(row["_audio_path"])

                def _loader(audio_path: Path = audio_path) -> np.ndarray:
                    audio, _ = load_audio(audio_path, SR)
                    return audio

                clips.append(Clip(
                    clip_id=f"mdc:{cv_lang}:{row['path']}",
                    speaker_id=spk,
                    text=row["_text"],
                    _loader=_loader,
                ))
            out[spk] = clips
        return out


def _find_cv_tsv(root: Path, split: str) -> Path | None:
    """Find the requested Common Voice split, falling back to validated."""
    if not root.exists():
        return None
    for name in dict.fromkeys((f"{split}.tsv", "validated.tsv")):
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    return None


def _find_cv_audio(tsv_dir: Path, root: Path, relpath: str) -> Path | None:
    candidates = (tsv_dir / "clips" / relpath, tsv_dir / relpath,
                  root / "clips" / relpath, root / relpath)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = list(root.rglob(relpath))
    return matches[0] if matches else None


def get_source(name: str, **kwargs):
    if name == "dummy":
        return DummySource(**kwargs)
    if name == "commonvoice":
        return CommonVoiceSource(**kwargs)
    if name == "commonvoice_mdc":
        return CommonVoiceMDCSource(**kwargs)
    raise KeyError(f"unknown dataset source {name!r}")
