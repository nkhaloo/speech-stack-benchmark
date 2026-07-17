# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A research benchmark (not a product) for selecting two private, self-hosted speech stacks: a GPU enterprise stack (Linux + NVIDIA) and a CPU desktop stack. It compares open-weight ASR models and speaker-diarization systems — individually and as fused sentence-level, speaker-attributed transcripts — across en, es, fr, ar, zh. Development happens on a MacBook; the real benchmark runs on a remote Linux GPU machine. The owner handles all git/GitHub operations themselves — never init repos, commit, or push.

## Commands

```bash
source .venv/bin/activate                    # env created by setup script

# Setup (creates .venv, installs .[dev], runs tests + smoke test)
./scripts/setup_macos.sh                     # Mac; --with-cpu-models adds runtimes
./scripts/setup_linux_gpu.sh                 # Linux lab; --with-pyannote4/--with-voxtral/--with-nemo add extra venvs

# Tests
python -m pytest                             # full suite, no downloads needed
python -m pytest tests/test_fusion.py -k test_split_on_gap   # single test

# End-to-end smoke (dummy data + dummy adapters, zero downloads)
./scripts/run_smoke_test.sh

# Real benchmark workflow (lab machine)
python scripts/prepare_datasets.py --profile baseline        # --source dummy for offline
python scripts/download_models.py --track gpu                # weights -> artifacts/models/ + HF cache
./scripts/run_gpu_benchmark.sh                               # or run_cpu_benchmark.sh
python scripts/generate_report.py --run-id <id>              # rebuild reports, no inference
python scripts/export_run.py --run-id <id>                   # portable results bundle
```

There is no linter configured. Scripts in `scripts/` import `_bootstrap` to add `src/` to `sys.path`, so they work without installation.

## Architecture: cache-then-fuse

The core design (docs/methodology.md) avoids re-running every ASR×diarizer pair:

1. **ASR stage** — each ASR model runs once per recording; normalized `ASRResult` JSON cached.
2. **Diarization stage** — each diarizer runs once per recording; `DiarizationResult` JSON + RTTM cached.
3. **Fusion stage** — every valid pair is combined *from cache only* (`fusion/assign.py`): word→speaker by max temporal overlap, then sentence splits on punctuation / >0.8s gaps / speaker changes.
4. **Metrics stage** — computed from cached predictions + references only.
5. **Reporting** — pure post-processing of saved metrics; must never load models.

Everything hinges on `src/speech_benchmark/schemas.py` (canonical `ASRResult`/`DiarizationResult`/`Reference`/`Recording` dataclasses with JSON round-trip). Adapters normalize into these; fusion/metrics/reporting consume only these. All writes go through `atomic_write_json` (temp + rename) so interrupted files never look complete.

**Adapter contract** (`asr/base.py`, `diarization/base.py`): `load()` / `transcribe(recording, language)` or `diarize(recording)` / `unload()`. The runner (`benchmark/runner.py`) guarantees one model in memory at a time, wraps calls with `ResourceMonitor` (peak RAM via psutil thread, VRAM via NVML), and converts exceptions into `failed` prediction JSON + structured records in `errors/` — a failing model must never abort the run. `AdapterUnavailable` at load time marks all that model's recordings `skipped` and continues. New adapters are registered in the `_REGISTRY` dict of `asr/__init__.py` / `diarization/__init__.py` and referenced by `runtime:` in a `configs/models/*.yaml` card.

**Config chain**: track configs (`configs/{cpu,gpu,smoke}.yaml`) list model-card paths; `config.load_track_config()` inlines them and drops `enabled: false` entries. Model cards carry both runtime params and license/download metadata — `download_models.py` and the report's license-flag section read the same cards.

**Resumability**: predictions live under `artifacts/runs/<run_id>/predictions/`; before inference the runner skips outputs with `status: completed` unless `--force`. Re-running with the same `--run-id` resumes. This also enables the multi-environment pattern: heavyweight/conflicting frameworks (pyannote.audio 4, vLLM, NeMo) live in separate venvs (`.venv-*`) and contribute to the *same run* by re-invoking with the same run id.

**Dummy adapters** (`asr/dummy.py`, `diarization/dummy.py`) read the reference and corrupt it with seeded error rates — this is how tests and the smoke test exercise the full pipeline with known-magnitude metrics and zero downloads. Reporting excludes `family: dummy` from leaderboards (unless the run is dummy-only, then it's flagged).

**Datasets** (`datasets/`): conversations are *constructed* (real Common Voice speech concatenated into multi-speaker dialogues with exact ground truth) — not TTS. Deterministic via seed + stable hashing; clip selection saved to `selection.json`. The `dummy` source generates tone audio for tests.

## Non-obvious conventions

- Chinese is scored at character level (`metrics/text_norm.py` `tokens()`); WER values are not comparable across languages — compare within a language, use macro/consistency columns across.
- cpWER (`metrics/combined_metrics.py`) builds hypothesis streams from **word-level** speaker labels, not sentence labels (sentence-level lumping inflated cpWER ~2.5x; fixed by also splitting sentences on speaker change).
- DER settings (collar 0.5s total, overlap scored) come from the track config `metrics:` block and are recorded in every metrics row.
- CPU and GPU results are never merged into one leaderboard, and no single blended score exists by design.
- License constraints are load-bearing: NeMo Sortformer weights are CC-BY-NC (non-commercial) — its config stays `enabled: false` and its adapter refuses to load without `acknowledge_non_commercial: true`. pyannote pipelines are gated (need `HF_TOKEN` + one-time terms acceptance) but offline after download. See docs/licensing.md before adding candidates.
- `artifacts/` (weights, datasets, runs, exports) is entirely gitignored; only code/configs/docs are committed.
- `docs/results.md` is a template to be filled from `artifacts/runs/<id>/reports/` after the Linux baseline run; docs/model_shortlist.md holds the pre-registered configuration-ladder hypotheses.
