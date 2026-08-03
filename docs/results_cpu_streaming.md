# CPU Streaming Track: Native-Streaming ASR with Deferred Speaker Labels

**Run:** `2026-07-29_cpu_streaming_subset6min` · **Track:** `cpu_streaming` · **Profile:** `smoke` (reduced) · **Status:** completed

*Hardware:* MacBook Air, Apple silicon (8 physical cores, 8 GB RAM) · macOS 15.6.1 · Python 3.12.2 · no GPU
*Code:* commit `993eae3` · *Wall clock:* ~8 min for the full matrix

---

## Scope, and what this run is not

This is the **complete stack matrix** — all three CPU streaming arms, all five
languages — run against a **deliberately reduced dataset**: the `smoke` dataset
profile, ~6 minutes per language (10 recordings, 3 minutes each, 2–3 speakers),
versus ~45 min/language for the `baseline` profile used in the GPU write-up.

The reduction was a deliberate choice, not an aborted run: the baseline profile
does not finish in reasonable time on a laptop CPU. Every number below comes from
**real Nemotron 3.5 weights on real Common Voice speech** (`synthetic_commonvoice_mdc`) —
none of it is dummy-adapter output.

Treat these as directional. They are sufficient to answer "does the live text path
keep up on desktop CPU" (it does, decisively) and insufficient to settle small
per-language differences. See [§ The French reversal](#the-french-reversal).

> The `smoke` in the profile name refers to **dataset size**, not to the
> dummy-adapter smoke test in `scripts/run_smoke_test.sh`. These are unrelated
> things that unfortunately share a word. The run id was previously
> `2026-07-29_streaming_smoke_cpustream`, which made real results read as a
> plumbing check; it has been renamed, and `scripts/run_streaming_benchmark.py`
> now derives the track component from the config.

## A. Final accuracy (macro across languages, lower is better)

| Stack | Macro WER ↓ | Macro cpWER ↓ | Macro DER ↓ | Worst-lang cpWER |
|:--|--:|--:|--:|--:|
| `cpu-stream-nemotron35-560ms-sherpa` | 0.331 | 0.778 | 0.596 | 1.080 |
| `cpu-stream-nemotron35-160ms-sherpa` | 0.481 | 0.861 | 0.618 | 1.085 |
| `cpu-stream-nemotron35-560ms-pyannote31` | 0.331 | **0.497** | **0.497** | 0.883 |

## B. Latency (seconds, lower is better)

| Stack | Time-to-first-token (median) | Finalization delay (median) | Finalization delay (p90) |
|:--|--:|--:|--:|
| `cpu-stream-nemotron35-560ms-sherpa` | 2.77 | 2.53 | 2.91 |
| `cpu-stream-nemotron35-160ms-sherpa` | 2.87 | 2.65 | 3.18 |
| `cpu-stream-nemotron35-560ms-pyannote31` | 2.76 | 2.53 | 2.90 |

## C. Stability (lower is better)

| Stack | Revision rate | Speaker-label churn | Token flicker | Streaming RTF |
|:--|--:|--:|--:|--:|
| `cpu-stream-nemotron35-560ms-sherpa` | 0.851 | 0.011 | 2.389 | 0.125 |
| `cpu-stream-nemotron35-160ms-sherpa` | 0.827 | 0.013 | 2.315 | 0.363 |
| `cpu-stream-nemotron35-560ms-pyannote31` | 0.851 | 0.007 | 2.389 | **0.107** |

`streaming_rtf` covers the **live path only**; the one-off end-of-session
diarization pass is excluded and recorded separately. All arms sit well under
1.0, which is the result this track existed to establish: a native streaming
decoder keeps up with the speaker on desktop CPU, where the retired windowed
faster-whisper arms measured above 2.0 and fell permanently behind.

## Per-language cpWER ↓

| Stack | ar | en | es | fr | zh |
|:--|--:|--:|--:|--:|--:|
| `cpu-stream-nemotron35-560ms-sherpa` | 0.913 | 0.570 | 0.739 | 0.588 | 1.080 |
| `cpu-stream-nemotron35-160ms-sherpa` | 1.038 | 0.661 | 0.833 | 0.689 | 1.085 |
| `cpu-stream-nemotron35-560ms-pyannote31` | 0.502 | 0.139 | 0.408 | **0.883** | 0.553 |

Chinese is scored at character level; WER/cpWER are not comparable across
languages. Compare within a language, or use the macro and worst-language columns.

## Findings

**1. The chunk-size ladder behaves as pre-registered.** The 160 ms arm is the same
weights at a tighter chunk, and it is worse on every accuracy axis (WER 0.481 vs
0.331) while costing ~3× the live-path RTF (0.363 vs 0.125). That gap is the price
of lower latency with nothing else varying — and the latency it buys is marginal
here (time-to-first-token 2.87 vs 2.77 s). On this evidence the 560 ms chunk is
the better operating point.

**2. The pre-registered primary candidate lost on diarization.** `configs/cpu_streaming.yaml`
names `560ms-sherpa` as "the PRIMARY candidate — the design we expect to ship,"
largely because it is the only fully torch-free arm. Swapping *only* the diarizer
to pyannote-3.1 cut macro cpWER from 0.778 to 0.497. The ASR is identical and WER
is identical at 0.331, so the entire 0.281 gap is diarizer quality — sherpa's DER
of 0.596 is doing the damage. Pyannote also came out marginally cheaper on the
live path (RTF 0.107 vs 0.125), since the diarization cost is deferred out of it.

The cost is a ~2 GB PyTorch dependency, on a machine with 8 GB RAM. That is the
trade to decide, and it is a deployment question as much as an accuracy one.

### The French reversal

Pyannote wins in Arabic, English, Spanish, and Chinese — by large margins (English
0.139 vs 0.570). It **loses in French**: 0.883 vs 0.588.

At ~6 minutes of French, this is exactly the kind of single-language reversal the
reduced dataset cannot adjudicate. It could be a genuine pyannote weakness on
French conversational audio, or it could be one or two bad recordings. Note that
the GPU baseline write-up also found French the weak language across every stack,
which makes a real effect somewhat more plausible — but that run used a different
ASR family, so it is suggestive, not confirmatory.

**This should be resolved on the baseline profile before the primary CPU candidate
is formally switched.** The macro gap is large enough that pyannote is the leading
candidate; it is not yet large enough, on this sample, to close the question.

## Reproducing this run

```bash
source .venv/bin/activate
python scripts/prepare_datasets.py --profile smoke      # needs MDC_API_KEY
python scripts/download_models.py --track cpu_streaming
python scripts/run_streaming_benchmark.py \
    --config configs/cpu_streaming.yaml --profile smoke --tag subset6min
```

Substitute `--profile baseline` for the full dataset. Expect it to take a long
time on a laptop — that is the reason this run used the reduced profile.

Raw artifacts (predictions, per-recording metrics, RTTMs) are **not in git** —
`artifacts/` is gitignored by design, since it also holds model weights and
datasets. To share a portable bundle of this run:

```bash
python scripts/export_run.py --run-id 2026-07-29_cpu_streaming_subset6min
```

## Relationship to the other tracks

CPU and GPU results are never merged into one leaderboard, and no blended score
exists — see `docs/methodology_streaming.md`. The GPU batch baseline is in
`docs/results.md`; the GPU streaming work is in
`docs/benchmark_summary_batch_streaming.md`. This document covers the CPU
streaming track only.
