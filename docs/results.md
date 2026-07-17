# Results and recommendations

> **Status: template.** Populated after the baseline runs on the Linux lab
> machine (GPU track) and a desktop-class CPU machine. Numbers below must
> come from `artifacts/runs/<run_id>/reports/` — link the run ids here so
> every claim is traceable.

## Runs

| Run id | Track | Profile | Host | Date | Status |
|---|---|---|---|---|---|
| _e.g._ `2026-07-XX_gpu_baseline_v1` | gpu | baseline | lab | | |
| _e.g._ `2026-07-XX_cpu_baseline_v1` | cpu | baseline | | | |

## GPU track (primary)

### Leaderboard
Paste/summarize `reports/leaderboard_gpu.csv`. Do **not** pick a winner from
the macro average alone — check `cpwer_worst_language` and
`cpwer_std_across_languages` for cross-language consistency, and the
per-language table for Arabic and Chinese specifically (weakest coverage
risk for Whisper-family models).

### Recommendation (to fill)
| Config | ASR | Diarizer | VRAM | Expected use |
|---|---|---|---|---|
| Minimum viable | | | | |
| **Recommended** | | | | |
| Highest quality | | | | |

Pre-benchmark hypotheses (from `model_shortlist.md`): minimum
`fw-medium + pyannote-3.1`; recommended `fw-large-v3-turbo + pyannote-3.1`;
highest `fw-large-v3 + pyannote-community-1` (CC-BY attribution). Confirm or
overturn with data, and justify with: per-language WER/CER, DER,
cpWER/attribution, RTF, load time, VRAM, licensing, streaming-simulation
latency.

## CPU track (desktop)

Same structure, from `leaderboard_cpu.csv`. Keep MacBook-vs-other-CPU
context explicit (hosts are in each run manifest). Note RAM ceilings for
target desktops and whisper.cpp-vs-faster-whisper packaging implications
(single binary + ggml file vs Python runtime).

## Cross-cutting findings (to fill)

* Language-detection accuracy per model (`metrics/per_model/language_detection.csv`).
* Streaming suitability: time-to-first-text, finalization delay, revisions —
  which candidates can plausibly do sentence-level near-real-time.
* Failure modes: OOM, crashes, degenerate outputs (`reports/failures.csv`).
* License/deployment concerns carried into the recommendation
  (`summary.md` flags section).

## Threats to validity (keep with any shared conclusions)

* Synthetic read-speech conversations → absolute WER/DER optimistic;
  rankings more reliable than magnitudes (see `datasets.md`).
* No overlapping speech in the baseline data.
* Arabic rests entirely on synthetic data at this stage.
* CPU/GPU tracks are separate experiments; never merged.
