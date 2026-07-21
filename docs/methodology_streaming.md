# Streaming benchmark — methodology (pre-registration)

> **Status: proposal / pre-registration.** This document specifies a *second,
> separate* benchmark protocol for **streaming-capable** speech systems. It does
> not modify the existing batch benchmark (`methodology.md`). It is written
> before any streaming code exists so the protocol — candidates, contract,
> metrics — is fixed in advance, the same way `model_shortlist.md` pre-registers
> the batch hypotheses. Numbers marked *(default, to confirm)* are starting
> values to be locked once the harness runs on the lab machine.

## 1. Why a separate benchmark

The existing benchmark is **cache-then-fuse**: each ASR model and each diarizer
runs *once* over a whole recording, its output is cached, and every ASR×diarizer
pair is scored purely from cache (`methodology.md`). That design assumes the
whole file is available up front and that ASR and diarization are independent
until fusion.

Streaming breaks both assumptions. A streaming system consumes audio
incrementally under a real-time clock, emits partial hypotheses that may be
revised, and — for diarization — must maintain speaker identity *over time*
rather than clustering a finished file. You cannot run-once-and-cache a
streaming system, so this is a parallel protocol with its own runner, its own
metrics, and its own report. **Batch and streaming results are never merged**,
exactly as CPU and GPU tracks are never merged.

Output artifact: **`results_streaming.md`** (separate from `results.md`), built
from a separate run directory.

## 2. Candidate constraints (unchanged from the batch shortlist)

Every streaming candidate must:

* be **open-source / open-weight** and downloadable,
* run **fully offline** after download,
* be at least **plausibly commercial-compatible** in licensing (non-commercial
  weights are *reference-only*, flagged, and disabled by default — same rule as
  NeMo Sortformer in the batch track; see `licensing.md`),
* cover all five target languages **en, es, fr, ar, zh** (the binding filter),
* run **incrementally** — natively streaming, or a batch model driven by a
  windowed wrapper (both are eligible; the wrapper is labeled as such).

## 3. Candidate set (pre-registered)

The open-weight + 5-language + commercial filter makes the field small. That is
expected and acceptable — the benchmark's job is to measure the few survivors
honestly, not to be large.

### 3.1 Streaming ASR

| id | System | Kind | License | Notes |
|---|---|---|---|---|
| `whisperlive` | faster-whisper via WhisperLive (VAD + LocalAgreement stabilization) | **native streaming** | MIT | Streaming Whisper done right — stable, timestamped incremental output; covers all 5 languages (planned) |
| `window-fw-large-v3-turbo` | faster-whisper large-v3-turbo + windowing wrapper | batch-in-window (**baseline**) | MIT | The batch winner, run incrementally, as the reference to beat |
| `window-fw-medium` | faster-whisper medium + windowing wrapper | batch-in-window (baseline) | MIT | Cheaper baseline |

### 3.2 Streaming diarization

| id | System | Kind | License | Notes |
|---|---|---|---|---|
| `stream-diart` | diart online diarization (over pyannote `segmentation` + `embedding`) | **native streaming** | Apache-2.0 code + gated pyannote weights | Incremental clustering; models differ from batch pyannote-3.1 — see §7 |
| `window-pyannote-31` | pyannote-3.1 + growing/sliding-buffer wrapper | batch-in-window (**baseline**) | MIT (gated) | The batch diarizer, run incrementally |
| `ref-streaming-sortformer` | NVIDIA Streaming Sortformer | native streaming | **CC-BY-NC — reference-only, disabled** | Non-commercial; included only as a quality yardstick, off by default |

License and language coverage for every candidate must be re-verified against
its Hugging Face card at pre-registration time and recorded in `licensing.md`
before it is enabled — no candidate is trusted from memory.

## 4. Dataset — reuse the existing corpus (no new data required to start)

Streaming evaluation needs, per recording: the audio, and a time-stamped,
speaker-attributed reference. The existing corpus already provides both — a
`Reference` is a list of `ReferenceTurn`s carrying **speaker + start + end +
text** (`schemas.py`). That is exactly what streaming metrics score against
(*what* was said, *by whom*, *when*). **The same audio and the same references
are reused unchanged**; only the way audio is *fed* (incrementally) and the
output is *measured* is new.

**Known limitation.** The corpus is clean, non-overlapping, read speech — a
best-case floor (as `results.md` states). This is *desirable* for a first
streaming pass: it isolates the streaming penalty from acoustic difficulty. But
it under-tests the cases streaming is hardest at — overlapping speech, natural
turn-taking, backchannels. A more natural/overlapping **v2 dataset** would make
the numbers more field-realistic and is recorded here as an explicit follow-up,
**not** a prerequisite.

## 5. Streaming contract (the pre-registered regime)

Ambiguity here makes "streaming WER/DER" meaningless, so the regime is fixed up
front and recorded in every metrics row:

* **Feed model.** Audio is delivered to the system in order under a **simulated
  real-time clock** (wall-time paced to audio time). Inference latency
  accumulates against that clock; a system slower than real time falls behind
  and is scored on when it *actually* emits.
* **Chunk / step size.** *(default, to confirm)* 0.5 s audio frames delivered to
  the adapter; the adapter decides its own internal window.
* **Latency budget.** *(default, to confirm)* a candidate targets sentence
  finalization within **≤ 2.0 s** of the reference turn's end. This is reported,
  not enforced — systems that exceed it are not disqualified, they score worse on
  latency.
* **Relabeling policy.** Retroactive revision of already-emitted text/speaker
  labels **is allowed** but **counted** (see revision metrics). A system that
  never revises and a system that revises constantly are both valid; the metrics
  expose the trade-off rather than forbidding either.
* **Emission log.** Every candidate produces a time-ordered log of emissions
  `(wall_time, sentence_text, speaker_label, is_final)` — the raw material for
  all streaming metrics. This log is cached per recording (analogous to the batch
  predictions cache) so metrics and reporting never re-run inference.

## 6. Streaming adapter interface

Batch adapters implement `transcribe(recording, language)` /
`diarize(recording)` (`asr/base.py`, `diarization/base.py`). Streaming adapters
implement a **push interface** instead:

```
load() / unload()                     # unchanged lifecycle
reset()                               # clear per-recording streaming state
push(audio_chunk, t) -> [Emission]    # feed a frame, get zero+ (revisable) emissions
flush() -> [Emission]                 # end-of-stream: final emissions
```

* **Native streaming** systems (WhisperLive, diart) implement `push` directly.
* **Batch** models (Whisper, pyannote-3.1) are wrapped by a generic
  **windowing driver** that buffers audio, re-invokes the batch adapter on a
  window (growing buffer for bounded sessions, or sliding window + speaker
  stitching), diffs against the last emission, and emits revisions. The wrapper
  is generic, so any existing batch adapter becomes a streaming baseline for free.

Failure isolation, resumability, atomic writes, and one-model-in-memory carry
over unchanged from the batch runner.

## 7. Metrics

Streaming quality is **not** a single number. Each candidate is scored on three
axes, all computed from the cached emission log + the existing references.

**A. Final accuracy** (does the end result match?) — the emitted transcript
(final emissions only) is scored with the *existing* metrics stack and text
normalization: WER/CER (per-language rules; Chinese at character level), DER
(collar 0.5 s, overlap scored), and **cpWER** built from word-level speaker
labels. This is directly comparable to the batch numbers as the *quality cost of
going incremental*.

**B. Latency** (how soon is it right?) —
* **time-to-first-token** per recording;
* **finalization delay** = emission wall-time of a sentence minus the reference
  turn's end time, summarized as median / p90.

**C. Stability** (does it stop changing?) —
* **revision rate** = revised emissions / total emissions;
* **speaker-label churn** = fraction of already-emitted sentences whose speaker
  label later changes (the metric that captures the retroactive-relabeling
  problem of growing-buffer diarization);
* **token flicker** = mean edits to already-shown text before it finalizes.

Every row records the streaming contract (§5) it was produced under. The three
axes are reported side by side; **no blended streaming score is produced**, by
design — a fast-but-churny system and a slow-but-stable one are left visibly
different for the reader to judge against the product's latency budget.

## 8. Run organization & reporting

A streaming run mirrors the batch layout under its own run id:
`artifacts/runs/<id>/` with the manifest (host, GPU, driver/CUDA, dep freeze,
seeds, config copies, **the streaming contract**), the per-recording emission
logs (the streaming prediction cache), metrics at per-recording / per-language /
per-candidate levels, tables, charts, structured errors, and
`reports/results_streaming.md`. Reporting is pure post-processing of the cached
emission logs and references and **must never load a model** — identical rule to
the batch reporter. MacBook numbers are never treated as predictive of Linux
numbers; the manifest records the host.

## 9. Pre-registered hypotheses (to confirm or overturn)

* **H1.** Windowed large-v3-turbo remains competitive on final WER but pays a
  materially worse latency/finalization-delay than a native streaming ASR
  (WhisperLive).
* **H2.** Growing-buffer windowed pyannote-3.1 has *low* final DER but *high*
  speaker-label churn; diart has higher final DER but far lower churn — i.e. the
  choice is accuracy-vs-stability, not a clean win.
* **H3.** French remains the weak language in streaming too (it is in batch), and
  the streaming penalty (Δ vs batch) is largest where diarization is hardest.
* **H4.** No single candidate wins all three axes; the recommendation will be
  contract-dependent (which latency budget the product actually needs).

## 10. Running it (lab machine)

**Baseline windowed stacks — no extra setup** (uses the batch-track models):
```bash
python scripts/prepare_datasets.py --profile baseline
python scripts/download_models.py --track gpu
python scripts/run_streaming_benchmark.py --config configs/streaming.yaml --profile baseline
# -> reports/results_streaming.md
```

**Focused `diart + WhisperLive large-v3-turbo` test** — separate server and
diart client environments + gated weights:
```bash
./scripts/setup_diart.sh                       # creates .venv-diart (pinned torch)
./scripts/setup_whisperlive.sh                 # creates .venv-whisperlive
# accept terms: huggingface.co/pyannote/segmentation and /pyannote/embedding
export HF_TOKEN=<token>
.venv-diart/bin/python scripts/check_streaming_env.py     # readiness preflight
./scripts/run_diart_whisperlive.sh baseline
```

The run script starts a localhost WhisperLive server with the CTranslate2
`deepdml/faster-whisper-large-v3-turbo-ct2` model, feeds the already-prepared baseline manifest through the
WhisperLive WebSocket in 0.5 s frames, fuses each segment snapshot with diart's
online speaker turns, then stops the server. Set `WHISPERLIVE_PORT` if port 9090
is occupied.

The WhisperLive setup deliberately omits microphone-only PyAudio and installs
the CUDA 12 cuBLAS/cuDNN runtime libraries required by CTranslate2 inside the
venv, so neither PortAudio headers nor root access are required.

Diart uses its published AMI hyperparameters (`tau=0.507`, `rho=0.006`,
`delta=1.057`) and caps online clustering at four speakers. The corpus contains
at most three speakers, so the cap leaves headroom while preventing runaway
cluster creation; it does not use each recording's reference speaker count.
The promoted runtime uses 2 s diart latency: the smoke tuning ladder reduced
cpWER and fused-timeline DER while cutting speaker-label churn substantially,
with no material RTF cost. The prior 0.5 s baseline remains preserved under
its original run id.

To compare pyannote's newer segmentation model without changing the promoted
stack, run the isolated `segmentation-3.0` variant:

```bash
STREAMING_CONFIG=configs/streaming_diart_whisperlive_segmentation3.yaml \
STREAMING_TAG=diart-whisperlive-segmentation3-latency2-v1 \
./scripts/run_diart_whisperlive.sh baseline
```

This changes only the segmentation weights to
`pyannote/segmentation-3.0`; the `pyannote/embedding` model, diart AMI
thresholds, 2 s latency, and WhisperLive settings remain unchanged. Treat its
thresholds as untuned until the comparison is complete.

**Parameter-tuning smoke ladder** — runs the unchanged control plus six
predeclared variants on the smoke corpus in one resumable comparison run:

```bash
./scripts/run_streaming_tuning_smoke.sh
```

The single-variable variants test finalization at 2 s, WhisperLive stability
threshold 3, diart latency at 1 s and 2 s, and `delta_new=0.95`. A combined
candidate is included as a hypothesis only and must be interpreted after the
single-variable rows. The baseline card and completed baseline artifacts are
not modified.

The `--run-id` re-invocation is the multi-env pattern (§1): each stack contributes
to one shared run from whatever env it needs; already-completed stacks are skipped.

## 11. What this benchmark deliberately does not do

Production streaming server, packaging/deployment, noisy/overlap/accent stress
testing (until the v2 dataset), model fine-tuning, and any merge with the batch
leaderboard. It answers one question: *among open-weight, multilingual,
streaming-capable systems, what is the accuracy / latency / stability trade-off
for sentence-level speaker-attributed transcription?*
