# speech-stack-benchmark

Reproducible research benchmark for choosing two private, self-hosted
speech-processing stacks:

1. **GPU enterprise stack** (Linux + NVIDIA, offline) — primary.
2. **CPU personal-computer stack** (macOS / Windows desktop).

It compares open-weight **ASR** models and **speaker-diarization** systems —
individually and as fused, sentence-level speaker-attributed transcripts —
across **English, Spanish, French, Arabic, Chinese**.

Research stage only: no production packaging, APIs, or deployment infra.

## Documentation

| Doc | Contents |
|---|---|
| [docs/model_shortlist.md](docs/model_shortlist.md) | Candidate models, inclusion/rejection rationale, config ladder |
| [docs/licensing.md](docs/licensing.md) | Verified library + weights licenses (incl. non-commercial exclusions) |
| [docs/datasets.md](docs/datasets.md) | Dataset design (synthetic Common Voice conversations + real anchors) |
| [docs/methodology.md](docs/methodology.md) | Pipeline, metrics definitions, scoring settings, run layout |
| [docs/results.md](docs/results.md) | Results & recommendations (filled after the Linux run) |

## Repository layout

```
configs/            track configs (cpu/gpu/smoke), model cards, dataset profiles
src/speech_benchmark/
  asr/              adapters: faster-whisper, whisper.cpp, Voxtral(vLLM), dummy
  diarization/      adapters: pyannote, sherpa-onnx, NeMo Sortformer(flagged), dummy
  datasets/         deterministic synthetic-conversation builder (+ Common Voice)
  fusion/           word→speaker assignment, sentence segmentation
  metrics/          WER/CER, DER, cpWER, speaker attribution
  benchmark/        runner (cache/resume/manifest), resources, streaming sim
  reporting/        aggregation, leaderboards, charts, summary.md, export
scripts/            setup + workflow entry points
tests/              unit + end-to-end tests (dummy adapters, no downloads)
artifacts/          models/datasets/runs/exports — NOT committed (.gitignore)
```

## MacBook workflow (development)

```bash
cd ~/Desktop/speech-stack-benchmark
./scripts/setup_macos.sh          # env + deps + unit tests + dummy smoke test
```

That's it — the smoke test exercises the entire pipeline (dataset → ASR →
diarization → fusion → metrics → report) with dummy adapters and **zero
model downloads**. Inspect `artifacts/latest/reports/summary.md`.

Optional real CPU-model testing on the Mac:

```bash
./scripts/setup_macos.sh --with-cpu-models
brew install whisper-cpp                       # whisper-cli binary
python scripts/download_models.py --track cpu  # + export HF_TOKEN for pyannote
python scripts/prepare_datasets.py --profile smoke   # real Common Voice data
./scripts/run_cpu_benchmark.sh smoke
```

MacBook timings are development data only — they do not predict Linux GPU
performance, and reports keep the tracks separate.

## Linux lab workflow (the real benchmark)

After you commit and push this repo yourself, on the lab machine:

```bash
git clone <repository-url>
cd speech-stack-benchmark
./scripts/setup_linux_gpu.sh                 # verifies driver/GPUs, creates .venv, GPU validation
source .venv/bin/activate
export HF_TOKEN=...                          # gated pyannote models
export MDC_API_KEY=...                       # Common Voice downloads

python scripts/prepare_datasets.py --profile baseline
python scripts/download_models.py --track gpu
./scripts/run_gpu_benchmark.sh               # baseline profile; resumable
python scripts/generate_report.py --track gpu
```

Optional extra environments (kept separate on purpose):
`./scripts/setup_linux_gpu.sh --with-pyannote4 --with-voxtral --with-nemo`
(pyannote community-1 / Voxtral-vLLM / NeMo Sortformer — the last is
CC-BY-NC, reference only). Run the same `--run-id` from each env; cached
completed models are skipped automatically.

Dataset preparation uses Common Voice Scripted Speech 26.0 from Mozilla Data
Collective. Before running it, accept the conditions on each of the five locale
pages listed in `docs/datasets.md` and create an API key under MDC Profile → API.
The official client resumes interrupted archive downloads; archives and
extracted files remain under `artifacts/datasets/mdc/` and are gitignored.

### 1. Where results live

Every run gets a readable id, e.g. `2026-07-17_gpu_baseline_v1`:

```
artifacts/runs/<run_id>/
  run_manifest.json    # env, GPUs, versions, seeds, configs, statuses
  predictions/         # cached ASR / diarization (JSON+RTTM) / combined
  metrics/             # per_recording / per_language / per_model / per_stack
  reports/             # summary.md, leaderboard_gpu.csv, failures.csv, ...
  charts/  logs/  errors/  config/  environment/
artifacts/latest       # symlink to newest run
artifacts/index.csv    # one row per run
```

### 2. Resuming

Interrupted? Re-run the same command with the same `--run-id` (the default
id is date-based, so same-day reruns resume automatically). Completed
outputs are skipped; use `--force` to recompute.

### 3. Regenerating reports (no inference)

```bash
python scripts/generate_report.py --run-id 2026-07-17_gpu_baseline_v1
```

### 4. Exporting a compact results bundle

```bash
python scripts/export_run.py --run-id 2026-07-17_gpu_baseline_v1 \
  --output artifacts/exports/2026-07-17_gpu_baseline_v1
# add --include-predictions for compressed raw predictions
```

The bundle contains manifests, configs, metrics, leaderboards, charts,
reports, and failure records — never model weights or datasets.

### 5. Copying results back to the MacBook

```bash
rsync -av <lab-host>:~/speech-stack-benchmark/artifacts/exports/<run_id>/ \
  ~/Desktop/speech-stack-benchmark/artifacts/exports/<run_id>/
open artifacts/exports/<run_id>/reports/summary.md
```

## Working rules baked into the design

* Everything runs offline after downloads; no hosted inference APIs
  (ElevenLabs Scribe is an output-format reference only).
* Each ASR/diarization model runs **once** per recording; every valid pair
  is fused from cache. One model in memory at a time.
* CPU and GPU results are reported separately; no single blended score —
  leaderboards show quality, consistency-across-languages, and cost columns
  side by side.
* Non-commercial weights (NeMo Sortformer) are flagged and disabled by
  default; see docs/licensing.md before changing that.
