# Model shortlist

Research date: **2026-07-17**. Licenses were verified against the current
Hugging Face model cards / vendor announcements on that date (see
`licensing.md` for the full table). Every shortlisted candidate:

* has downloadable open weights,
* runs fully offline after download,
* has at least plausibly commercial-compatible licensing (or is explicitly
  flagged otherwise),
* looks realistically packageable later (container / desktop app).

Target languages: **en, es, fr, ar, zh** — this constraint eliminated more
candidates than anything else.

---

## GPU ASR (primary track)

| id | Model / runtime | Why it's in | VRAM (est.) | Timestamps |
|---|---|---|---|---|
| `fw-large-v3` | Whisper large-v3 via faster-whisper (CTranslate2, fp16) | Accuracy reference; best-documented multilingual coverage of all 5 languages; MIT everything | ~10 GB | segment + word |
| `fw-large-v3-turbo` | Whisper large-v3-turbo via faster-whisper | ~4× faster decoder, small quality loss; likely "recommended" config | ~6 GB | segment + word |
| `fw-medium` | Whisper medium via faster-whisper | Minimum-viable GPU config; fits ~4 GB VRAM | ~4 GB | segment + word |
| `voxtral-mini-vllm` | Voxtral Mini (Mistral) via local vLLM — **experimental** | Apache-2.0, 13 languages incl. all 5 targets; the *Realtime-2602* variant is a true streaming ASR (240 ms–2.4 s configurable delay) — the main modern non-Whisper option | ≥16 GB | limited (verify per vLLM version) |

Notes:
* faster-whisper was chosen over openai/whisper (reference impl) and
  transformers-Whisper because it is markedly faster and lighter at equal
  accuracy, needs no torch, and quantizes for the CPU track with the same
  adapter. WhisperX was skipped: its main add-on (forced alignment) pulls
  per-language wav2vec2 models with mixed licenses; faster-whisper's native
  word timestamps are sufficient for sentence-level attribution.
* Voxtral is behind a local vLLM server in its own env (`.venv-voxtral`);
  the batch `Voxtral-Mini-3B-2507` is evaluated by the standard pipeline,
  and the Realtime variant should be assessed separately for streaming
  latency. Its diarization-enabled "Transcribe 2" sibling is API-only — not
  eligible here.

## CPU ASR (desktop track)

| id | Model / runtime | Why it's in | RAM (est.) |
|---|---|---|---|
| `wcpp-large-v3-turbo-q5` | whisper.cpp, ggml large-v3-turbo q5_0 | Desktop quality ceiling; Metal on Apple Silicon; excellent Windows story; ~600 MB file | ~2 GB |
| `wcpp-small-q5` | whisper.cpp, ggml small q5_1 | Minimum footprint (~200 MB) | ~1 GB |
| `fw-small-int8-cpu` | faster-whisper small, int8 | Same runtime as GPU track ⇒ simpler product code-path; solid CPU speed | ~1.5 GB |
| `fw-medium-int8-cpu` | faster-whisper medium, int8 | Desktop quality/speed middle ground | ~2.5 GB |

## GPU diarization

| id | System | Why it's in | Flag |
|---|---|---|---|
| `pyannote-3.1` | pyannote/speaker-diarization-3.1 | De-facto open standard; MIT weights; language-independent | gated download (HF terms + token) |
| `pyannote-community-1` | pyannote/speaker-diarization-community-1 (2025) | Successor with better overlap handling | CC-BY-4.0 (attribution), gated, needs pyannote.audio ≥ 4 → separate env |
| `nemo-sortformer-4spk` | NVIDIA Sortformer 4spk-v1 | **Reference only — CC-BY-NC weights (non-commercial), disabled by default.** Included solely to see how far the commercial candidates are from it, if desired | ≤4 speakers; separate NeMo env |

## CPU diarization

| id | System | Why it's in |
|---|---|---|
| `sherpa-diar-cpu` | sherpa-onnx offline diarization (pyannote segmentation-3.0 ONNX + 3D-Speaker ERes2NetV2 embeddings) | Torch-free, Apache-2.0 runtime, small models, runs on macOS/Windows/Linux CPUs — best desktop packaging story |
| `pyannote-3.1-cpu` | pyannote 3.1 forced to CPU | Quality reference on desktop; measures how slow "the good one" is without a GPU |

---

## Evaluated and rejected

| Candidate | Reason |
|---|---|
| Meta MMS / SeamlessM4T | CC-BY-NC weights — non-commercial |
| NVIDIA Canary (1b, 1b-v2) / Parakeet | Strong models, but no Arabic + Chinese coverage (European-language focus); Canary 1b original also CC-BY-NC |
| distil-whisper | English-only (fails 4 of 5 languages) |
| SenseVoice-small (FunASR) | zh/en/ja/ko/yue only — no es/fr/ar. Worth revisiting as a *Chinese-specific* booster in the extended phase |
| FireRedASR | zh/en only |
| Qwen2-Audio / Qwen3-Omni | Apache-2.0 but audio-LLMs: no reliable timestamps, heavy VRAM, hallucination risk in pure-ASR use |
| Vosk / Kaldi models | Aging accuracy vs Whisper family; per-language model zoo with mixed provenance |
| whisper.cpp tinydiarize | English-only, experimental |
| NeMo MSDD / clustering diarizer | Superseded by Sortformer in NeMo itself; heavy dependency for no expected quality win over pyannote |
| ElevenLabs Scribe | Hosted API only — used strictly as an output-format reference (sentence-level, speaker-labeled transcripts), never for inference |

## Configuration ladder (hypotheses to verify with the benchmark)

* **GPU minimum:** `fw-medium` + `pyannote-3.1` (~6 GB VRAM total, single mid-range GPU)
* **GPU recommended:** `fw-large-v3-turbo` + `pyannote-3.1` (~8 GB VRAM)
* **GPU highest-quality:** `fw-large-v3` + `pyannote-community-1` (~13 GB VRAM; CC-BY attribution obligation)
* **CPU minimum:** `wcpp-small-q5` + `sherpa-diar-cpu`
* **CPU recommended:** `wcpp-large-v3-turbo-q5` (or `fw-medium-int8-cpu`) + `sherpa-diar-cpu`

The benchmark exists to confirm or overturn these hypotheses — see
`results.md` after the Linux run.
