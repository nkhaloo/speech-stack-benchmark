# Speech-stack benchmark report: batch baseline and streaming segmentation 3.0

## Executive summary

This report combines the corrected GPU batch baseline with the strongest native
streaming run completed so far. Both runs use the same five-language,
Common-Voice-derived baseline corpus, but they are separate benchmark tracks and
must not be merged into one leaderboard.

- **Recommended batch deployment stack:** faster-whisper `large-v3-turbo` plus
  `pyannote/speaker-diarization-3.1`. It achieved macro WER 0.1584 and macro
  cpWER 0.3487 at an end-to-end RTF of 0.021.
- **Best native streaming stack tested:** WhisperLive/faster-whisper
  `large-v3-turbo` plus diart using `pyannote/segmentation-3.0` and
  `pyannote/embedding`. It achieved macro WER 0.239, macro cpWER 0.525, and
  macro DER 0.345.
- Segmentation 3.0 improved the native streaming control's cpWER from 0.558 to
  0.525 and DER from 0.368 to 0.345 without a meaningful compute penalty.
- The streaming system remains outside the 2.0-second target: median first-token
  latency was 2.46 seconds and median finalization delay was 5.29 seconds.

## 1. Evaluation dataset

The benchmark corpus contains deterministic, constructed multi-speaker
conversations made from real Mozilla Common Voice Scripted Speech 26.0 clips. It
does not use synthetic speech.

| Property | Baseline profile |
|:--|:--|
| Languages | Arabic, English, Spanish, French, Chinese (zh-CN) |
| Recordings | 45 total, approximately 9 per language |
| Audio per language | Approximately 45 minutes |
| Total audio | Approximately 3.75 hours |
| Recording duration | Approximately 5 minutes |
| Speakers | 2–4 per recording |
| Source split | Full Common Voice `validated` pool |
| Construction seed | `20260717` |
| Turn gaps | Random 0.4–1.2 seconds |
| Overlap | None intentionally introduced |

Clips from known individual speakers are concatenated into conversations. This
provides exact transcript, speaker, and timing references while keeping the
construction reproducible through the fixed seed and `selection.json` record.

### Silence-reference correction

Common Voice clips often contain silence at their edges. Labelling each complete
clip as speech caused diarizers to be penalized for correctly detecting that
silence. The preparation pipeline therefore uses a clip-relative RMS gate:

- 25 ms analysis frames;
- silence threshold 40 dB below the clip's peak RMS;
- silent leading and trailing regions removed;
- the current voiced-span builder uses a 10 ms hop and 100 ms boundary padding;
- internal pauses long enough to separate voiced regions are left unlabelled;
- voiced regions shorter than 100 ms are discarded.

The original edge-only correction reduced batch pyannote DER from 0.3722 to
0.2459, primarily by reducing incorrectly measured missed speech from 0.3031 to
0.1514. This is a reference-ground-truth correction, not speech enhancement.

### Dataset limitations

The source audio is real, but the conversations are constructed from scripted,
separately recorded speech. They do not reproduce natural interruption,
backchannels, overlapping speech, meeting acoustics, or long-term speaker drift.
Results are most useful for controlled model comparisons and should be treated as
a best-case accuracy floor rather than a field-performance estimate.

## 2. Batch GPU benchmark

**Run:** `2026-07-19_gpu_baseline_v1`  
**Hardware:** NVIDIA GeForce RTX 4090 (26 GB), Linux, Python 3.12.3

The batch runner caches each ASR and diarization result once and then evaluates
all ASR–diarizer combinations from cache. Chinese text is scored at character
level; the other languages are scored at word level. DER uses a 0.5-second total
collar (±0.25 seconds), with overlap scored.

### ASR results

| ASR model | Macro WER ↓ | Worst-language WER ↓ | WER std | RTF ↓ | Peak VRAM |
|:--|--:|--:|--:|--:|--:|
| `large-v3` | **0.1527** | 0.2225 | 0.0663 | 0.040 | 6022.7 MB |
| `large-v3-turbo` | 0.1584 | **0.1891** | **0.0484** | **0.015** | 3438.3 MB |
| `medium` | 0.1883 | 0.3417 | 0.1068 | 0.019 | 3424.1 MB |

All three models achieved 100% language-detection accuracy on this corpus.

### Batch diarization results

The batch track used `pyannote/speaker-diarization-3.1`.

| Metric | Macro result |
|:--|--:|
| DER ↓ | 0.2459 |
| Missed speech ↓ | 0.1514 |
| False alarm ↓ | 0.0006 |
| Speaker confusion ↓ | 0.0939 |
| Mean speaker-count error | -0.0222 |

### Combined batch stacks

| ASR + `pyannote-3.1` | Macro cpWER ↓ | Worst cpWER ↓ | Word attribution ↑ | Sentence attribution ↑ | RTF ↓ | Peak VRAM |
|:--|--:|--:|--:|--:|--:|--:|
| `large-v3` | **0.3465** | 0.6624 | 0.8895 | 0.8529 | 0.046 | 6022.7 MB |
| `large-v3-turbo` | 0.3487 | **0.6546** | **0.8902** | 0.8517 | **0.021** | 3438.3 MB |
| `medium` | 0.3676 | 0.6605 | 0.8883 | **0.8599** | 0.026 | 3424.2 MB |

`large-v3-turbo` is recommended because its 0.0022 absolute cpWER difference
from `large-v3` is negligible, while it is about twice as fast, uses roughly 40%
less VRAM, and has better worst-language consistency.

### Best batch result observed per language

These are the best values among the three batch ASR stacks; the winning ASR model
is not necessarily the same in every row.

| Language | WER ↓ | DER (`pyannote-3.1`) ↓ | cpWER ↓ | Winning ASR for cpWER |
|:--|--:|--:|--:|:--|
| Arabic | 0.1659 | 0.1983 | 0.4543 | `large-v3-turbo` |
| English | 0.0535 | 0.2021 | 0.1689 | `large-v3` |
| Spanish | 0.1219 | 0.3517 | 0.1565 | `medium` |
| French | 0.1757 | 0.3676 | 0.6546 | `large-v3-turbo` |
| Chinese | 0.1801 | 0.1097 | 0.2081 | `large-v3-turbo` |

## 3. Best native streaming run

**Run:** `2026-07-21_streaming_baseline_diart-whisperlive-segmentation3-latency2-v1`  
**Stack:** `diart-segmentation-3.0-whisperlive-large-v3-turbo`

The stack combines:

- WhisperLive with faster-whisper `large-v3-turbo`;
- diart online diarization;
- `pyannote/segmentation-3.0` segmentation;
- `pyannote/embedding` speaker embeddings;
- a 5.0-second diarization window, 0.5-second hop, and 2.0-second diart latency;
- AMI clustering thresholds `tau_active=0.507`, `rho_update=0.006`, and
  `delta_new=1.057`;
- a four-speaker clustering cap, matching this corpus's 2–4 speaker design;
- 0.5-second input frames and a 2.0-second product latency target.

### Streaming aggregate results

| Axis | Metric | Result |
|:--|:--|--:|
| Accuracy | Macro WER ↓ | 0.239 |
| Accuracy | Macro cpWER ↓ | 0.525 |
| Accuracy | Macro DER ↓ | 0.345 |
| Accuracy | Worst-language cpWER ↓ | 0.764 |
| Latency | First-token median ↓ | 2.46 s |
| Latency | Finalization median ↓ | 5.29 s |
| Latency | Finalization p90 ↓ | 15.82 s |
| Stability | Revision rate ↓ | 0.700 |
| Stability | Speaker-label churn ↓ | 0.053 |
| Stability | Token flicker ↓ | 1.895 |
| Efficiency | Streaming RTF ↓ | 0.110 |

### Streaming results by language

The source metrics did not contain a false-alarm column, so DER components below
show the recorded missed-speech and speaker-confusion values only.

| Language | WER ↓ | cpWER ↓ | DER ↓ | Missed speech ↓ | Speaker confusion ↓ | First token ↓ | Finalization median ↓ | RTF ↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Arabic | 0.3535 | 0.7642 | 0.5020 | 0.0325 | 0.2253 | 2.2574 s | 5.2891 s | 0.1087 |
| English | **0.1112** | 0.5528 | 0.3769 | 0.1061 | 0.2008 | 2.5862 s | 5.3016 s | 0.1098 |
| Spanish | 0.1937 | **0.3310** | **0.2426** | 0.1153 | **0.0628** | 3.1538 s | 5.3051 s | **0.1079** |
| French | 0.2191 | 0.5041 | 0.3101 | 0.0398 | 0.1781 | **2.0403 s** | **5.2671 s** | 0.1116 |
| Chinese | 0.3182 | 0.4739 | 0.2922 | 0.0626 | 0.0951 | 2.2533 s | 5.3019 s | 0.1116 |

Bold values identify the best language result within a column. Cross-language
WER comparisons require care because Chinese uses character-level scoring.

### Stability by language

| Language | Revision rate ↓ | Speaker-label churn ↓ | Token flicker ↓ |
|:--|--:|--:|--:|
| Arabic | **0.5893** | **0.0299** | **1.4002** |
| English | 0.7754 | 0.0636 | 2.0325 |
| Spanish | 0.6523 | 0.0535 | 1.4779 |
| French | 0.7147 | 0.0697 | 2.0622 |
| Chinese | 0.7659 | 0.0476 | 2.5030 |

### Effect of segmentation 3.0

The control used `pyannote/segmentation` with otherwise equivalent promoted
settings. Replacing only the segmentation model produced:

| Metric | Original segmentation | Segmentation 3.0 | Outcome |
|:--|--:|--:|:--|
| Macro WER ↓ | 0.243 | **0.239** | Slight improvement |
| Macro cpWER ↓ | 0.558 | **0.525** | 5.9% relative improvement |
| Macro DER ↓ | 0.368 | **0.345** | 6.3% relative improvement |
| Worst-language cpWER ↓ | 0.827 | **0.764** | 7.6% relative improvement |
| First-token median ↓ | 2.46 s | 2.46 s | Unchanged |
| Finalization median ↓ | 5.31 s | **5.29 s** | Essentially unchanged |
| Finalization p90 ↓ | 21.22 s | **15.82 s** | 25.4% relative improvement |
| Revision rate ↓ | 0.708 | **0.700** | Slight improvement |
| Speaker churn ↓ | 0.058 | **0.053** | Improvement |
| Token flicker ↓ | 1.949 | **1.895** | Improvement |
| Streaming RTF ↓ | 0.112 | **0.110** | Essentially unchanged |

Segmentation 3.0 is therefore the strongest native streaming configuration tested
on this corpus. Its main unresolved issues are accuracy behind the batch stack,
first-token latency above the 2-second target, long-tail finalization delay, and a
pyannote pooling warning about mismatched frame and weight counts observed during
inference. Results completed successfully despite the warning, but compatibility
and threshold calibration should be investigated on a natural diarization corpus.

## 4. Access, gating, tokens, and licenses

Gating controls model download access; it is not the same as a restrictive model
license. Once the gated pyannote weights are downloaded and cached, both evaluated
stacks can run fully offline.

| Component | Used in | License | Approval required? | Credential required? | Offline after download? |
|:--|:--|:--|:--|:--|:--|
| Common Voice Scripted Speech 26.0 | Dataset | CC0-1.0 | Accept each locale's Mozilla Data Collective conditions | `MDC_API_KEY` for preparation through MDC; no HF token | Yes |
| Whisper `large-v3-turbo` weights | Batch + streaming ASR | MIT | No | No HF token for the normal faster-whisper download | Yes |
| faster-whisper / CTranslate2 | Batch + streaming runtime | MIT | No | None | Yes |
| WhisperLive | Streaming server | Open-source runtime; verify installed release | No | None | Yes |
| diart | Streaming diarization runtime | MIT | No | None for code | Yes |
| `pyannote/speaker-diarization-3.1` | Batch diarization | MIT weights | **Yes** | **HF token required to download** | Yes |
| `pyannote/segmentation-3.0` | Batch dependency + best streaming run | MIT weights | **Yes** | **HF token required to download** | Yes |
| `pyannote/embedding` | Streaming speaker embeddings | MIT weights per project card | **Yes** | **HF token required to download** | Yes |
| `deepdml/faster-whisper-large-v3-turbo-ct2` | Streaming ASR model conversion | Whisper-derived MIT weights | No observed approval gate | No HF token normally required | Yes |

For the segmentation-3.0 streaming run, accept the model conditions for both
`pyannote/segmentation-3.0` and `pyannote/embedding`, then export a token that has
access:

```bash
export HF_TOKEN=<hugging-face-token>
```

The token is needed for initial access/download and cache resolution. It is not
needed for offline inference when all required snapshots are already present and
the runtime is configured not to contact Hugging Face.

## 5. Conclusions and next steps

1. Use faster-whisper `large-v3-turbo` plus pyannote 3.1 for the current batch
   deployment candidate. It provides the strongest speed, memory, consistency,
   and accuracy trade-off.
2. Promote the segmentation-3.0 configuration as the leading experimental native
   streaming stack, but do not compare its metrics directly with batch as though
   both tracks had the same inference contract.
3. Validate the segmentation change on a natural, overlapping diarization corpus
   such as VoxConverse. VoxConverse can validate DER but lacks reference
   transcripts for WER and cpWER.
4. Investigate Arabic streaming DER/cpWER, English speaker attribution, Chinese
   token flicker, and Spanish first-token latency.
5. Investigate the segmentation-3.0 frame/weight mismatch warning and retune diart
   clustering thresholds rather than assuming the AMI thresholds transfer
   perfectly.
6. Pin Hugging Face model revisions and record installed diart, pyannote.audio,
   WhisperLive, and faster-whisper versions for stronger reproducibility.

---

Batch source: `docs/results.md`, run `2026-07-19_gpu_baseline_v1`. Streaming
source: run
`2026-07-21_streaming_baseline_diart-whisperlive-segmentation3-latency2-v1`
and its per-recording language aggregation supplied from the completed run cache.
