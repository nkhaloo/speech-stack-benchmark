# Dataset shortlist and design

Requirement: audio + reference transcripts + speaker turns + timestamps +
language labels, in **en, es, fr, ar, zh**, reasonably clean/conversational,
openly licensed.

## The core problem

There is **no single open corpus** that provides speaker-labeled,
transcribed, conversational audio in all five languages. Real conversational
corpora with full annotation are either English/Chinese only (AMI,
AISHELL-4), behind LDC paywalls (CALLHOME, DIHARD), or research-agreement
gated (MGB-2 for Arabic). This is the decisive constraint, so the benchmark
uses a two-part design:

### 1. Primary: deterministic synthetic conversations (all 5 languages)

`scripts/prepare_datasets.py` builds multi-speaker "conversations" by
concatenating single-speaker, human-read utterances from **Mozilla Common
Voice Scripted Speech 26.0** (CC0; per-clip speaker `client_id`; validated
transcripts; all five languages, `zh` = `zh-CN`), with randomized speaker
order and 0.4–1.2 s silence gaps, no overlap. Mozilla Data Collective hosts
the archives; the retired Hugging Face Common Voice repositories are no
longer used.

Why this is the right baseline:

* **Perfect ground truth** — exact transcript, speaker, start/end for every
  turn, in every language, under one permissive license (CC0).
* **Uniform difficulty across languages** — differences between models are
  attributable to the model, not to corpus mismatch.
* **Reproducible** — fixed seed; speaker/clip selection is a stable hash
  order; the exact clip list is saved to `selection.json` per profile.
* Matches the requested baseline conditions: clean audio, no heavy noise,
  no overlapping speech, general vocabulary.

Known limitations (accept consciously, revisit in the extended phase):

* Read speech, not spontaneous conversation — WER will be optimistic vs.
  real meetings; relative model ranking is still informative.
* No overlapped speech and clean turn gaps — DER will be optimistic and
  clustering-friendly; diarizer differences compress but ordering holds.
* Common Voice clips have variable mic quality — mild realistic noise.

### 2. Real-audio anchors (planned extension; loaders not yet implemented)

To sanity-check that synthetic rankings transfer to real conversations:

| Language | Corpus | License | Provides | Use |
|---|---|---|---|---|
| en | **AMI** (headset mix, scenario meetings) | CC-BY-4.0 | audio, words, speakers | DER + full-stack anchor; RTTMs available from the pyannote `AMI-diarization-setup` repo (MIT) |
| zh | **AISHELL-4** (real Mandarin meetings) | CC-BY-SA-4.0 | audio, transcripts, speaker VAD | DER + CER anchor; share-alike applies to redistributed derivatives only |

Rejected for this stage: CALLHOME/CallFriend/DIHARD (LDC, paid),
MGB-2 (Arabic TV, research agreement), VoxConverse (annotations CC-BY but
audio must be re-fetched from YouTube — availability rot), AliMeeting
(license terms unclear for commercial-adjacent use), FLEURS (CC-BY-4.0 but
no stable per-speaker IDs → cannot build multi-turn same-speaker
conversations), VoxPopuli (en/es/fr only, oratory).

For Arabic, no fully open real conversational corpus with transcripts +
speaker labels was identified; candidates to investigate later: SADA
(SDAIA, TV broadcast — verify license), QASR (Al Jazeera, research terms).
Until then, Arabic conclusions rest on the synthetic set — flagged in
reporting.

## Profiles

Configured in `configs/datasets/synthetic.yaml` (seed `20260717`):

| Profile | Per language | Recording length | Speakers | Purpose |
|---|---|---|---|---|
| `smoke` | ~6 min | 3 min | 2–3 | adapter/metric debugging |
| `baseline` | ~45 min | 5 min | 2–4 | the main comparison |
| `extended` | ~120 min | 6 min | 2–5 | optional later testing |

A `dummy` source (`--source dummy`) generates tone-based pseudo-speech with
codeword transcripts for zero-download pipeline testing (used by unit tests
and `run_smoke_test.sh`).

## Access notes

* Common Voice requires a Mozilla Data Collective account, one-time acceptance
  of each locale's conditions, and `MDC_API_KEY` from Profile → API. Dataset
  ids are pinned in `configs/datasets/synthetic.yaml`: English (`en`), Spanish
  (`es`), French (`fr`), Arabic (`ar`), and Chinese (China, `zh-CN`).
* The archives are large because MDC provides complete per-locale releases.
  Downloads are resumable and cached in `artifacts/datasets/mdc/`; make sure
  the lab machine has sufficient disk space before preparing the baseline.
* Preparation is fully independent of the benchmark run:
  `python scripts/prepare_datasets.py --profile baseline` on the lab machine.
* Outputs land in `artifacts/datasets/<dataset>/<profile>/`: per-language
  WAVs (16 kHz mono), `<id>.ref.json` references, `manifest.jsonl`,
  `selection.json`. Nothing under `artifacts/` is committed.
