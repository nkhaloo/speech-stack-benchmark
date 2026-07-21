"""Streaming report — pure post-processing of cached streaming metrics into
`reports/results_streaming.md`. Never loads a model (same rule as the batch
reporter). The three axes are reported side by side; no blended score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..schemas import atomic_write_text, load_json


def _fmt(v, nd: int = 3) -> str:
    if v is None:
        return "—"
    try:
        if v != v:  # NaN
            return "—"
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _macro(df, stack: str, col: str):
    """Macro = mean over per-language means (so languages are weighted equally)."""
    import numpy as np
    sub = df[df["stack"] == stack]
    if col not in sub.columns:
        return None, None
    per_lang = sub.groupby("language")[col].mean()
    per_lang = per_lang.dropna()
    if per_lang.empty:
        return None, None
    worst = per_lang.max()  # for error metrics higher = worse
    return float(per_lang.mean()), float(worst)


def generate_streaming_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    manifest = load_json(run_dir / "run_manifest.json") if (
        run_dir / "run_manifest.json").exists() else {}
    rows_path = run_dir / "metrics" / "per_recording" / "streaming_rows.json"
    rows = load_json(rows_path) if rows_path.exists() else []

    import pandas as pd
    df = pd.DataFrame(rows)
    out = run_dir / "reports" / "results_streaming.md"

    lines: list[str] = []
    contract = manifest.get("streaming_contract", {})
    lines.append("# Streaming speech-stack benchmark — results\n")
    lines.append(f"**Run:** `{manifest.get('run_id', run_dir.name)}`  ·  "
                 f"**Track:** {manifest.get('track', 'streaming')}  ·  "
                 f"**Profile:** {manifest.get('profile', '?')}  ·  "
                 f"**Status:** {manifest.get('status', '?')}\n")
    lines.append(f"*Contract:* frame {contract.get('frame_sec', '?')} s · "
                 f"latency budget {contract.get('latency_budget_sec', '?')} s · "
                 "relabeling allowed (counted). See docs/methodology_streaming.md.\n")

    if df.empty:
        lines.append("\n_No streaming rows found._\n")
        atomic_write_text(out, "\n".join(lines))
        return out

    completed = df[df.get("status") == "completed"].copy()
    families = set(completed.get("family", pd.Series(dtype=str)).dropna())
    dummy_only = families and families <= {"dummy"}
    if not dummy_only and "family" in completed.columns:
        completed = completed[completed["family"] != "dummy"]

    if dummy_only:
        lines.append("\n> ⚠️ **Dummy-only run.** All stacks are `family: dummy` "
                     "(reference-derived, seeded). Numbers validate the pipeline, "
                     "not real systems.\n")

    stacks = list(dict.fromkeys(completed["stack"])) if "stack" in completed else []
    if not stacks:
        lines.append("\n_No completed non-dummy stacks to report._\n")
        _append_status(lines, df)
        atomic_write_text(out, "\n".join(lines))
        return out

    # A. Final accuracy
    lines.append("\n## A. Final accuracy (macro across languages, lower is better)\n")
    lines.append("| Stack | Macro WER ↓ | Macro cpWER ↓ | Macro DER ↓ | Worst-lang cpWER |")
    lines.append("|:--|--:|--:|--:|--:|")
    for st in stacks:
        wer, _ = _macro(completed, st, "wer")
        cp, cp_worst = _macro(completed, st, "cpwer")
        der, _ = _macro(completed, st, "der")
        lines.append(f"| `{st}` | {_fmt(wer)} | {_fmt(cp)} | {_fmt(der)} | {_fmt(cp_worst)} |")

    # B. Latency
    lines.append("\n## B. Latency (seconds, lower is better)\n")
    lines.append("| Stack | Time-to-first-token (median) | Finalization delay (median) | Finalization delay (p90) |")
    lines.append("|:--|--:|--:|--:|")
    for st in stacks:
        ttft, _ = _macro(completed, st, "time_to_first_token_sec")
        fd, _ = _macro(completed, st, "finalization_delay_median_sec")
        fd90, _ = _macro(completed, st, "finalization_delay_p90_sec")
        lines.append(f"| `{st}` | {_fmt(ttft, 2)} | {_fmt(fd, 2)} | {_fmt(fd90, 2)} |")

    # C. Stability
    lines.append("\n## C. Stability (lower is better)\n")
    lines.append("| Stack | Revision rate | Speaker-label churn | Token flicker | Streaming RTF |")
    lines.append("|:--|--:|--:|--:|--:|")
    for st in stacks:
        rr, _ = _macro(completed, st, "revision_rate")
        ch, _ = _macro(completed, st, "speaker_label_churn")
        tf, _ = _macro(completed, st, "token_flicker")
        rtf, _ = _macro(completed, st, "streaming_rtf")
        lines.append(f"| `{st}` | {_fmt(rr)} | {_fmt(ch)} | {_fmt(tf)} | {_fmt(rtf)} |")

    # Per-language cpWER
    if "cpwer" in completed.columns:
        langs = sorted(set(completed["language"]))
        lines.append("\n## Per-language cpWER ↓\n")
        lines.append("| Stack | " + " | ".join(langs) + " |")
        lines.append("|:--|" + "|".join(["--:"] * len(langs)) + "|")
        for st in stacks:
            sub = completed[completed["stack"] == st]
            cells = []
            for lg in langs:
                v = sub[sub["language"] == lg]["cpwer"].mean()
                cells.append(_fmt(v))
            lines.append(f"| `{st}` | " + " | ".join(cells) + " |")

    lines.append("\n---\n")
    lines.append("The three axes are intentionally separate — a fast-but-churny stack "
                 "and a slow-but-stable one are left visibly different. Choose against "
                 "the product's latency budget. Batch and streaming leaderboards are "
                 "never merged (see docs/methodology_streaming.md).\n")
    _append_status(lines, df)
    atomic_write_text(out, "\n".join(lines))
    return out


def _append_status(lines: list[str], df) -> None:
    import pandas as pd
    if "status" not in df.columns:
        return
    counts = df["status"].value_counts().to_dict()
    if set(counts) - {"completed"}:
        lines.append("\n### Run status counts\n")
        for k, v in counts.items():
            lines.append(f"- {k}: {v}")
        lines.append("")


__all__ = ["generate_streaming_report"]
