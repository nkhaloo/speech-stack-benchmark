# Benchmark summary — `2026-07-17_gpu_baseline_v1`

- **Track:** gpu
- **Profile:** baseline
- **Languages:** ar, en, es, fr, zh
- **Recordings:** 45
- **ASR models:** fw-large-v3, fw-large-v3-turbo, fw-medium
- **Diarization models:** pyannote-3.1
- **Status:** completed
- **GPUs:** NVIDIA GeForce RTX 4090 (26 GB)
- **Host:** Linux 6.8.0-134-generic / Python 3.12.3

> Chinese scores use character-level error (token_unit=char); WER values are not directly comparable across languages — compare models *within* a language, and use the macro/consistency columns across.

## Best ASR by language (WER ↓)

| language   | asr_model         |    wer |
|:-----------|:------------------|-------:|
| ar         | fw-large-v3       | 0.2011 |
| en         | fw-medium         | 0.039  |
| es         | fw-medium         | 0.1238 |
| fr         | fw-large-v3-turbo | 0.1907 |
| zh         | fw-large-v3-turbo | 0.1703 |

## Best diarizer by language (DER ↓)

| language   | diar_model   |    der |
|:-----------|:-------------|-------:|
| ar         | pyannote-3.1 | 0.4191 |
| en         | pyannote-3.1 | 0.295  |
| es         | pyannote-3.1 | 0.4464 |
| fr         | pyannote-3.1 | 0.458  |
| zh         | pyannote-3.1 | 0.2427 |

## Best combined stack by language (cpWER ↓)

| language   | asr_model         | diar_model   |   cpwer |
|:-----------|:------------------|:-------------|--------:|
| ar         | fw-medium         | pyannote-3.1 |  0.4205 |
| en         | fw-medium         | pyannote-3.1 |  0.1044 |
| es         | fw-medium         | pyannote-3.1 |  0.191  |
| fr         | fw-medium         | pyannote-3.1 |  0.6838 |
| zh         | fw-large-v3-turbo | pyannote-3.1 |  0.2008 |

## GPU stack leaderboard (macro across languages)

Sorted by macro cpWER; check the worst-language and std columns for cross-language consistency before picking a winner.

| asr_model         | diar_model   |   cpwer |   cpwer_worst_language |   cpwer_std_across_languages |   word_attribution_accuracy |   sentence_attribution_accuracy |   real_time_factor |   peak_ram_mb |   peak_vram_mb |   n_languages |
|:------------------|:-------------|--------:|-----------------------:|-----------------------------:|----------------------------:|--------------------------------:|-------------------:|--------------:|---------------:|--------------:|
| fw-large-v3-turbo | pyannote-3.1 |  0.3341 |                 0.6908 |                       0.2279 |                      0.8974 |                          0.8595 |             0.0188 |       1786.82 |        3439.81 |             5 |
| fw-medium         | pyannote-3.1 |  0.353  |                 0.6838 |                       0.2248 |                      0.8965 |                          0.8659 |             0.0223 |       1786.82 |        3411.47 |             5 |
| fw-large-v3       | pyannote-3.1 |  0.3554 |                 0.6915 |                       0.2227 |                      0.8962 |                          0.859  |             0.0383 |       1786.82 |        5980.96 |             5 |

## ASR models (macro across languages)

| asr_model         |    wer |   wer_worst_language |   wer_std_across_languages |    cer |   real_time_factor |   model_load_time_sec |   peak_ram_mb |   peak_vram_mb |   time_to_first_text_sec |   sentence_finalization_delay_sec |
|:------------------|-------:|---------------------:|---------------------------:|-------:|-------------------:|----------------------:|--------------:|---------------:|-------------------------:|----------------------------------:|
| fw-large-v3-turbo | 0.1551 |               0.225  |                     0.0657 | 0.1052 |             0.0122 |                1.2034 |       1275.42 |        3439.81 |                   5.1958 |                           14.0086 |
| fw-large-v3       | 0.1705 |               0.2578 |                     0.0789 | 0.1244 |             0.0317 |                2.3794 |       1238.42 |        5980.96 |                   5.2831 |                           13.8815 |
| fw-medium         | 0.1937 |               0.3488 |                     0.1176 | 0.1402 |             0.0158 |                1.0889 |       1295.1  |        3411.47 |                 nan      |                          nan      |

## Diarization models (macro across languages)

| diar_model   |    der |   der_worst_language |   der_std_across_languages |   missed_speech |   false_alarm_speech |   speaker_confusion |   speaker_count_error |   real_time_factor |   peak_ram_mb |   peak_vram_mb |
|:-------------|-------:|---------------------:|---------------------------:|----------------:|---------------------:|--------------------:|----------------------:|-------------------:|--------------:|---------------:|
| pyannote-3.1 | 0.3722 |                0.458 |                     0.0972 |          0.3031 |               0.0002 |               0.069 |                0.2444 |             0.0066 |       1786.82 |        3269.82 |

## Language detection accuracy

| asr_model         | language   |   detection_accuracy |
|:------------------|:-----------|---------------------:|
| fw-large-v3       | ar         |                    1 |
| fw-large-v3       | en         |                    1 |
| fw-large-v3       | es         |                    1 |
| fw-large-v3       | fr         |                    1 |
| fw-large-v3       | zh         |                    1 |
| fw-large-v3-turbo | ar         |                    1 |
| fw-large-v3-turbo | en         |                    1 |
| fw-large-v3-turbo | es         |                    1 |
| fw-large-v3-turbo | fr         |                    1 |
| fw-large-v3-turbo | zh         |                    1 |
| fw-medium         | ar         |                    1 |
| fw-medium         | en         |                    1 |
| fw-medium         | es         |                    1 |
| fw-medium         | fr         |                    1 |
| fw-medium         | zh         |                    1 |

## License / deployment flags

- `fw-large-v3` (MIT): Batch model; sentence-level streaming via chunked windows.
- `fw-large-v3-turbo` (MIT): 4x faster decoder than large-v3 with small quality loss.
- `pyannote-3.1` (MIT): gated download (accept terms + HF token); Requires pyannote.audio 3.x; language-independent.

## Failed / incomplete experiments

_none_

---
_Not a single blended score by design — see docs/methodology.md. Generated from cached results only._
