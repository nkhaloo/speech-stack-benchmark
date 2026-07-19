# Benchmark summary — `2026-07-19_gpu_baseline_v1`

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
| ar         | fw-large-v3       | 0.1719 |
| en         | fw-large-v3       | 0.0448 |
| es         | fw-large-v3-turbo | 0.0878 |
| fr         | fw-large-v3       | 0.1927 |
| zh         | fw-large-v3-turbo | 0.1909 |

## Best diarizer by language (DER ↓)

| language   | diar_model   |    der |
|:-----------|:-------------|-------:|
| ar         | pyannote-3.1 | 0.1017 |
| en         | pyannote-3.1 | 0.1403 |
| es         | pyannote-3.1 | 0.1422 |
| fr         | pyannote-3.1 | 0.3131 |
| zh         | pyannote-3.1 | 0.0844 |

## Best combined stack by language (cpWER ↓)

| language   | asr_model         | diar_model   |   cpwer |
|:-----------|:------------------|:-------------|--------:|
| ar         | fw-large-v3-turbo | pyannote-3.1 |  0.2838 |
| en         | fw-large-v3       | pyannote-3.1 |  0.1287 |
| es         | fw-large-v3-turbo | pyannote-3.1 |  0.1566 |
| fr         | fw-large-v3       | pyannote-3.1 |  0.6886 |
| zh         | fw-large-v3-turbo | pyannote-3.1 |  0.2472 |

## GPU stack leaderboard (macro across languages)

Sorted by macro cpWER; check the worst-language and std columns for cross-language consistency before picking a winner.

| asr_model         | diar_model   |   cpwer |   cpwer_worst_language |   cpwer_std_across_languages |   word_attribution_accuracy |   sentence_attribution_accuracy |   real_time_factor |   peak_ram_mb |   peak_vram_mb |   n_languages |
|:------------------|:-------------|--------:|-----------------------:|-----------------------------:|----------------------------:|--------------------------------:|-------------------:|--------------:|---------------:|--------------:|
| fw-large-v3-turbo | pyannote-3.1 |  0.3117 |                 0.7055 |                       0.2267 |                      0.9058 |                          0.858  |             0.0202 |       2135.25 |        3439.81 |             5 |
| fw-large-v3       | pyannote-3.1 |  0.3213 |                 0.6886 |                       0.2168 |                      0.9086 |                          0.867  |             0.0448 |       2135.25 |        6050.31 |             5 |
| fw-medium         | pyannote-3.1 |  0.3507 |                 0.6959 |                       0.217  |                      0.9037 |                          0.8641 |             0.025  |       2135.25 |        3408.54 |             5 |

## ASR models (macro across languages)

| asr_model         |    wer |   wer_worst_language |   wer_std_across_languages |    cer |   real_time_factor |   model_load_time_sec |   peak_ram_mb |   peak_vram_mb |   time_to_first_text_sec |   sentence_finalization_delay_sec |
|:------------------|-------:|---------------------:|---------------------------:|-------:|-------------------:|----------------------:|--------------:|---------------:|-------------------------:|----------------------------------:|
| fw-large-v3-turbo | 0.146  |               0.1963 |                     0.0567 | 0.1041 |             0.0138 |                0.9524 |       1262.87 |        3439.81 |                   5.202  |                           13.621  |
| fw-large-v3       | 0.1552 |               0.2199 |                     0.0673 | 0.1107 |             0.0384 |                1.7733 |       1222.03 |        6050.31 |                   5.2924 |                           14.7819 |
| fw-medium         | 0.1989 |               0.3443 |                     0.0997 | 0.149  |             0.0186 |                0.8093 |       1272.97 |        3408.49 |                 nan      |                          nan      |

## Diarization models (macro across languages)

| diar_model   |    der |   der_worst_language |   der_std_across_languages |   missed_speech |   false_alarm_speech |   speaker_confusion |   speaker_count_error |   real_time_factor |   peak_ram_mb |   peak_vram_mb |
|:-------------|-------:|---------------------:|---------------------------:|----------------:|---------------------:|--------------------:|----------------------:|-------------------:|--------------:|---------------:|
| pyannote-3.1 | 0.1563 |               0.3131 |                     0.0911 |           0.066 |               0.0006 |              0.0897 |                0.1333 |             0.0064 |       2135.25 |         3276.3 |

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
