# Benchmark methodology

## Architecture

```
prepare_datasets ──> manifest.jsonl + refs
                          │
        ┌─────────────────┼──────────────────┐
        v                 v                  │
  ASR stage         Diarization stage        │   one model loaded at a time,
  (per model,       (per model,              │   outputs cached as JSON/RTTM
   per recording)    per recording)          │
        │                 │                  │
        └────────┬────────┘                  │
                 v                           │
          Fusion stage  ← every valid (ASR × diar) pair from cache
                 │
                 v
          Metrics stage ← references + cached predictions only
                 │
                 v
          Reporting     ← aggregation, leaderboards, charts, summary.md
```

Key rules:

* **No duplicated inference.** Each ASR model and each diarizer runs once
  per recording; pairs are evaluated purely from cached outputs.
* **One model in memory at a time.** Load → process all recordings → unload
  (with CUDA cache release) → next model.
* **Resumable everywhere.** A completed cached output (status
  `completed`) is skipped on rerun unless `--force`. Reuse the run id to
  resume an interrupted run. Files are written atomically
  (temp file + `os.replace`), so partial writes are never mistaken for
  results.
* **Failure isolation.** Exceptions become a `failed` prediction JSON plus a
  structured error record in `errors/` (stage, model, recording, timestamp,
  exception type, message, traceback); the run continues, and reporting
  includes failures explicitly.
* **Incompatible frameworks get separate environments** (`.venv`,
  `.venv-pyannote4`, `.venv-voxtral`, `.venv-nemo`) rather than one forced
  environment. The cache-then-fuse design means results produced from
  different envs land in the same run folder (run the benchmark once per
  env with the same `--run-id`; already-completed models are skipped).

## Canonical schemas

`ASRResult` (text, detected language + probability, segments, word
timestamps + confidences, runtime, load time, peak RAM/VRAM, model metadata,
status) and `DiarizationResult` (speaker turns with confidences, speaker
count, same runtime/resource fields) — `src/speech_benchmark/schemas.py`.
Diarization additionally exports RTTM.

## Text normalization (before every text metric)

Unicode NFC → lowercase → strip punctuation/symbols (Unicode P*/S*) →
collapse whitespace. Arabic: remove diacritics + tatweel, unify alef
variants, ta marbuta → ha, alef maqsura → ya. Chinese: scored at
**character level** (`token_unit=char`); other languages at word level.
Identical normalization for reference and hypothesis.

## Metrics

**ASR:** WER (token-level per language rules above), CER, sub/del/ins
counts, timestamp availability, language-detection accuracy (models run
with auto-detect; `force_language: true` disables this to isolate pure
transcription quality).

**Diarization:** DER via `pyannote.metrics` with **collar 0.5 s total
(±0.25 s)**, overlap scored (`skip_overlap: false`) — both recorded in every
row and configurable in the track config. Components: missed speech, false
alarm, speaker confusion (as fractions of reference speech time), plus
speaker-count error (hyp − ref).

**Combined (speaker-attributed):**
* **cpWER** — per-speaker concatenated token streams; optimal
  hyp↔ref speaker assignment (Hungarian, cost = token errors; unmatched
  streams cost deletions/insertions); total errors / total ref tokens.
  Hypothesis streams use **word-level** speaker assignments.
* **word attribution accuracy** — % of hypothesis words (with a reference
  speaker active at the word midpoint) whose time-overlap-mapped speaker is
  correct.
* **sentence attribution accuracy** — same at sentence level
  (majority-overlap reference speaker).

**Performance:** model load time, runtime, RTF, peak RAM (process tree RSS,
0.2 s sampling), peak VRAM (NVML), weight disk size (model configs),
plus the streaming metrics below. CPU-track and GPU-track numbers are
never mixed in one leaderboard, and **MacBook numbers are never treated as
predictive of Linux numbers** — the manifest records the host of every run.

## Fusion (sentence-level output)

Word-level ASR output (interpolated timestamps when a model lacks word
timing — recorded via `has_word_timestamps`) → each word gets the maximal-
overlap diarization turn (nearest-turn fallback within 2 s) → sentence
splits on final punctuation, silence gaps > 0.8 s, diarized speaker changes,
or 30 s max length → sentence speaker = duration-weighted majority of its
words. This mirrors the target product output (sentence-level,
speaker-labeled, Scribe-style).

## Streaming simulation

Batch models are re-run over a moving window (default 20 s window, 5 s hop)
under a simulated real-time clock (audio arrives in real time; inference
latency accumulates if slower than the hop). Measured: time to first text,
mean sentence-finalization delay (text unchanged for N consecutive steps),
revision count, chunked RTF. Results are labeled `simulated_chunked` and
must never be compared 1:1 with a true streaming model's native latency
(Voxtral Realtime would be assessed with its own interface). This answers
"can this candidate support sentence-level near-real-time later", not
"what is production latency".

## Run organization

`artifacts/runs/<run_id>/` contains the full manifest (environment, GPU
inventory, driver/CUDA versions, dependency freeze, seeds, exact config
copies, CLI args, statuses), predictions (`asr/`, `diarization/` incl.
RTTM, `combined/`), metrics at per-recording / per-language / per-model /
per-stack levels (JSON + CSV), tables, charts (PNG), structured errors,
logs per stage/model, and `reports/` (summary.md, leaderboards, failures).
`artifacts/index.csv` lists all runs; `artifacts/latest` points at the most
recent. `scripts/export_run.py` builds the portable bundle.

## What this stage deliberately does not do

Production streaming server, packaging/deployment (K8s, APIs, licenses
tooling), noisy/overlap/accents stress testing, model fine-tuning. The
shortlist was filtered for *future* packageability only.
