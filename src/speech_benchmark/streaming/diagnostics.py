"""Cache-only diagnostics for a completed streaming benchmark run."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..schemas import (StreamingResult, atomic_write_json, atomic_write_text,
                       load_json)


def _finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _mean(values) -> float | None:
    vals = [_finite(v) for v in values]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _percentile(values, q: float) -> float | None:
    vals = [_finite(v) for v in values]
    vals = [v for v in vals if v is not None]
    return float(np.percentile(vals, q)) if vals else None


def _fmt(value, digits: int = 3) -> str:
    value = _finite(value)
    return "—" if value is None else f"{value:.{digits}f}"


def _emission_details(run_dir: Path) -> dict[tuple[str, str], dict]:
    details = {}
    root = run_dir / "predictions" / "streaming"
    for path in root.glob("*/*.json") if root.exists() else []:
        sr = StreamingResult.from_dict(load_json(path))
        if sr.status != "completed":
            continue
        finals = sr.final_emissions()
        delays = [max(0.0, e.wall_time - e.end) for e in finals
                  if e.is_final and e.end is not None]
        ids = {e.sentence_id for e in sr.emissions}
        final_ids = {e.sentence_id for e in sr.emissions if e.is_final}
        revisions = {}
        for e in sr.emissions:
            revisions[e.sentence_id] = max(revisions.get(e.sentence_id, 0),
                                            e.revision or 0)
        details[(sr.model_id, sr.recording_id)] = {
            "finalization_delays": delays,
            "latency_budget_passes": None,
            "num_emissions": len(sr.emissions),
            "num_sentence_ids": len(ids),
            "finalized_id_fraction": len(final_ids) / len(ids) if ids else None,
            "max_revisions": max(revisions.values()) if revisions else 0,
            "mean_revisions": _mean(revisions.values()),
            "final_speakers": len({e.speaker for e in finals if e.speaker}),
        }
    return details


def generate_streaming_diagnostics(run_dir: str | Path) -> Path:
    """Generate detailed JSON and Markdown using cached metrics/emissions only."""
    import pandas as pd

    run_dir = Path(run_dir)
    manifest = load_json(run_dir / "run_manifest.json")
    rows = load_json(run_dir / "metrics/per_recording/streaming_rows.json")
    completed = pd.DataFrame([r for r in rows if r.get("status") == "completed"])
    if completed.empty:
        raise ValueError("No completed streaming rows found")

    budget = float(manifest.get("streaming_contract", {}).get(
        "latency_budget_sec", 2.0))
    details = _emission_details(run_dir)
    all_delays: list[float] = []
    all_budget_hits: list[bool] = []
    detail_rows = []
    for row in completed.to_dict("records"):
        d = details.get((row.get("stack"), row.get("recording_id")), {})
        delays = d.pop("finalization_delays", [])
        hits = [v <= budget for v in delays]
        all_delays.extend(delays)
        all_budget_hits.extend(hits)
        d["latency_budget_pass_rate"] = _mean(hits)
        detail_rows.append({**row, **d})
    df = pd.DataFrame(detail_rows)

    accuracy_cols = ["wer", "cpwer", "der", "missed_speech",
                     "false_alarm_speech", "speaker_confusion"]

    def summary(frame) -> dict:
        out = {c: _mean(frame[c]) if c in frame else None for c in accuracy_cols}
        count_error = frame.get("speaker_count_error", pd.Series(dtype=float))
        out.update({
            "recordings": int(len(frame)),
            "audio_hours": _mean([frame.get("audio_duration_sec", pd.Series()).sum()
                                   / 3600.0]),
            "speaker_count_exact_rate": _mean(count_error == 0),
            "speaker_count_mae": _mean(count_error.abs()),
            "speaker_under_count_rate": _mean(count_error < 0),
            "speaker_over_count_rate": _mean(count_error > 0),
            "ttft_median_sec": _percentile(frame.get("time_to_first_token_sec", []), 50),
            "ttft_p90_sec": _percentile(frame.get("time_to_first_token_sec", []), 90),
            "revision_rate": _mean(frame.get("revision_rate", [])),
            "speaker_label_churn": _mean(frame.get("speaker_label_churn", [])),
            "token_flicker": _mean(frame.get("token_flicker", [])),
            "streaming_rtf": _mean(frame.get("streaming_rtf", [])),
            "latency_budget_pass_rate": _mean(frame.get("latency_budget_pass_rate", [])),
            "finalized_id_fraction": _mean(frame.get("finalized_id_fraction", [])),
            "mean_revisions": _mean(frame.get("mean_revisions", [])),
            "max_revisions": _finite(frame.get("max_revisions", pd.Series([0])).max()),
        })
        return out

    overall = summary(df)
    overall.update({
        "finalization_delay_p50_sec": _percentile(all_delays, 50),
        "finalization_delay_p90_sec": _percentile(all_delays, 90),
        "finalization_delay_p95_sec": _percentile(all_delays, 95),
        "finalization_delay_max_sec": max(all_delays) if all_delays else None,
        "latency_budget_pass_rate": _mean(all_budget_hits),
    })
    per_language = {lang: summary(group) for lang, group in df.groupby("language")}

    def worst(column: str, n: int = 10) -> list[dict]:
        if column not in df:
            return []
        cols = list(dict.fromkeys(
            c for c in ("recording_id", "language", column, "wer", "cpwer",
                        "der", "missed_speech", "speaker_confusion",
                        "speaker_count_error", "finalization_delay_p90_sec")
            if c in df
        ))
        return df.sort_values(column, ascending=False)[cols].head(n).to_dict("records")

    payload = {
        "run_id": manifest.get("run_id", run_dir.name),
        "source": "cached streaming rows and emission logs; no inference",
        "latency_budget_sec": budget,
        "overall": overall,
        "per_language": per_language,
        "worst_by_cpwer": worst("cpwer"),
        "worst_by_fused_timeline_der": worst("der"),
    }
    atomic_write_json(run_dir / "metrics/streaming_diagnostics.json", payload)
    out = run_dir / "reports/streaming_diagnostics.md"
    atomic_write_text(out, _markdown(payload))
    return out


def _markdown(data: dict) -> str:
    o = data["overall"]
    lines = [
        "# Streaming benchmark — cached diagnostics\n",
        f"**Run:** `{data['run_id']}` · cache-only analysis; no inference rerun.\n",
        "## Accuracy and diarization components\n",
        "The DER below is **fused-timeline DER** reconstructed from finalized "
        "speaker-attributed sentences; it is not raw-diart DER.\n",
        "| WER | cpWER | Fused DER | Missed speech | False alarm | Speaker confusion |",
        "|--:|--:|--:|--:|--:|--:|",
        f"| {_fmt(o['wer'])} | {_fmt(o['cpwer'])} | {_fmt(o['der'])} | "
        f"{_fmt(o['missed_speech'])} | {_fmt(o['false_alarm_speech'])} | "
        f"{_fmt(o['speaker_confusion'])} |",
        "\n## Latency distribution\n",
        "| TTFT median | TTFT p90 | Final p50 | Final p90 | Final p95 | Final max | Within budget |",
        "|--:|--:|--:|--:|--:|--:|--:|",
        f"| {_fmt(o['ttft_median_sec'], 2)}s | {_fmt(o['ttft_p90_sec'], 2)}s | "
        f"{_fmt(o['finalization_delay_p50_sec'], 2)}s | "
        f"{_fmt(o['finalization_delay_p90_sec'], 2)}s | "
        f"{_fmt(o['finalization_delay_p95_sec'], 2)}s | "
        f"{_fmt(o['finalization_delay_max_sec'], 2)}s | "
        f"{_fmt(o['latency_budget_pass_rate'] * 100 if o['latency_budget_pass_rate'] is not None else None, 1)}% |",
        "\n## Speaker counting and stability\n",
        "| Exact speaker count | Count MAE | Under-count | Over-count | Revision rate | Speaker churn | Token flicker | RTF |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|",
        f"| {_fmt(o['speaker_count_exact_rate'] * 100, 1)}% | "
        f"{_fmt(o['speaker_count_mae'])} | {_fmt(o['speaker_under_count_rate'] * 100, 1)}% | "
        f"{_fmt(o['speaker_over_count_rate'] * 100, 1)}% | {_fmt(o['revision_rate'])} | "
        f"{_fmt(o['speaker_label_churn'])} | {_fmt(o['token_flicker'])} | {_fmt(o['streaming_rtf'])} |",
        "\n## Per-language diagnostics\n",
        "| Language | N | WER | cpWER | Fused DER | Miss | Confusion | Exact speakers | Within budget |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for lang, s in sorted(data["per_language"].items()):
        lines.append(
            f"| {lang} | {s['recordings']} | {_fmt(s['wer'])} | {_fmt(s['cpwer'])} | "
            f"{_fmt(s['der'])} | {_fmt(s['missed_speech'])} | "
            f"{_fmt(s['speaker_confusion'])} | {_fmt(s['speaker_count_exact_rate'] * 100, 1)}% | "
            f"{_fmt(s['latency_budget_pass_rate'] * 100 if s['latency_budget_pass_rate'] is not None else None, 1)}% |"
        )
    for title, key, metric in (("Worst recordings by cpWER", "worst_by_cpwer", "cpwer"),
                               ("Worst recordings by fused-timeline DER",
                                "worst_by_fused_timeline_der", "der")):
        lines.extend([f"\n## {title}\n", "| Recording | Lang | Value | WER | cpWER | DER | Miss | Confusion |",
                      "|:--|:--|--:|--:|--:|--:|--:|--:|"])
        for r in data[key]:
            lines.append(f"| `{r['recording_id']}` | {r['language']} | {_fmt(r.get(metric))} | "
                         f"{_fmt(r.get('wer'))} | {_fmt(r.get('cpwer'))} | {_fmt(r.get('der'))} | "
                         f"{_fmt(r.get('missed_speech'))} | {_fmt(r.get('speaker_confusion'))} |")
    lines.append("\nRaw-diart DER and alternative model settings require an audio replay; they are not inferred here.\n")
    return "\n".join(lines)


__all__ = ["generate_streaming_diagnostics"]
