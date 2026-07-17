# Licensing review

Verified **2026-07-17** against Hugging Face model cards, GitHub repos, and
vendor announcements. Library and weight licenses are recorded separately —
they frequently differ. Re-verify before any production commitment; model
cards can change between revisions (pin revisions at download time).

Legend: ✔ yes ✘ no ⚠ conditional.

## ASR

| Component | Library license | Weights license | Commercial use | Redistribution | Attribution required | Extra terms / gating | Offline after download |
|---|---|---|---|---|---|---|---|
| OpenAI Whisper weights (large-v3, turbo, medium, small) | — | MIT | ✔ | ✔ | ✘ (MIT notice only) | none | ✔ |
| faster-whisper + CTranslate2 | MIT | (uses Whisper weights) | ✔ | ✔ | ✘ | none | ✔ |
| whisper.cpp + ggml conversions | MIT | MIT (converted Whisper) | ✔ | ✔ | ✘ | none | ✔ |
| Voxtral Mini 3B (2507) / Mini 4B Realtime (2602) | vLLM: Apache-2.0 | Apache-2.0 | ✔ | ✔ | ✘ | none | ✔ |
| Meta MMS / SeamlessM4T | — | CC-BY-**NC**-4.0 | ✘ | ⚠ | ✔ | — | — |
| NVIDIA Canary-1b (original) | Apache-2.0 (NeMo) | CC-BY-**NC**-4.0 | ✘ | ⚠ | ✔ | — | — |

## Diarization

| Component | Library license | Weights license | Commercial use | Redistribution | Attribution required | Extra terms / gating | Offline after download |
|---|---|---|---|---|---|---|---|
| pyannote.audio | MIT | — | ✔ | ✔ | ✘ | none | ✔ |
| pyannote/speaker-diarization-3.1 (+ segmentation-3.0, wespeaker embedding) | — | MIT | ✔ | ✔ | ✘ | ⚠ **gated**: accept conditions + HF token to download | ✔ |
| pyannote/speaker-diarization-community-1 | — | CC-BY-4.0 | ✔ | ✔ | ✔ (**attribution**) | ⚠ gated; requires pyannote.audio ≥ 4 | ✔ |
| sherpa-onnx runtime | Apache-2.0 | — | ✔ | ✔ | ✘ | none | ✔ |
| sherpa-onnx pyannote segmentation-3.0 ONNX | — | MIT | ✔ | ✔ | ✘ | none (un-gated re-export of MIT model) | ✔ |
| 3D-Speaker ERes2NetV2 embeddings | Apache-2.0 | Apache-2.0 | ✔ | ✔ | ✘ | none | ✔ |
| **NVIDIA diar_sortformer_4spk-v1** | Apache-2.0 (NeMo) | **CC-BY-NC-4.0 — NON-COMMERCIAL** | **✘** | ⚠ | ✔ | — | ✔ |

## Datasets

| Dataset | License | Commercial-adjacent use for evaluation | Notes |
|---|---|---|---|
| Mozilla Common Voice | CC0-1.0 | ✔ | ⚠ HF download gated behind free terms acceptance; also downloadable from Mozilla directly |
| AMI Meeting Corpus | CC-BY-4.0 | ✔ (attribution) | real English meetings |
| AISHELL-4 | CC-BY-**SA**-4.0 | ✔ for evaluation; **share-alike** if redistributing derivatives | real Mandarin meetings |
| FLEURS | CC-BY-4.0 | ✔ | no per-speaker IDs → unusable for our synthetic-conversation builder |
| CALLHOME / CallFriend / DIHARD | LDC proprietary | ✘ (paid, non-redistributable) | excluded |
| MGB-2 (Arabic) | research agreement | ⚠ excluded for this stage | |

## Hard conclusions

1. **Sortformer is out** of any commercial recommendation (CC-BY-NC). It
   stays in the repo only as an optional, disabled reference point.
2. The **all-MIT stack** (Whisper weights + faster-whisper/whisper.cpp +
   pyannote 3.1) is the cleanest possible licensing position: no attribution
   chain, no share-alike, no gating **after** the one-time pyannote terms
   acceptance.
3. `community-1` (CC-BY-4.0) is fine commercially but adds a visible
   attribution obligation to the product — acceptable, worth flagging to
   whoever owns product legal review.
4. Gated ≠ non-open: pyannote pipelines are MIT/CC-BY but need a one-time
   Hugging Face terms acceptance and token for the *download*; inference is
   fully offline afterwards, and weights can be mirrored into the private
   environment (MIT permits redistribution — keep the license file).
5. Voxtral (Apache-2.0) is currently the only strong non-Whisper multilingual
   open-weights ASR covering all five languages.
