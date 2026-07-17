"""Markdown summary report, generated purely from saved tables.

Deliberately avoids any single blended score: the summary shows the primary
quality metric per axis (WER / DER / cpWER), consistency across the five
languages, and resource cost side by side, then leaves the tradeoff explicit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..schemas import atomic_write_text, load_json


def _md(df: pd.DataFrame, cols: list[str], n: int | None = None,
        sort: str | None = None, ascending: bool = True) -> str:
    if df.empty:
        return "_no data_\n"
    d = df.copy()
    if sort and sort in d.columns:
        d = d.sort_values(sort, ascending=ascending, na_position="last")
    if n:
        d = d.head(n)
    cols = [c for c in cols if c in d.columns]
    d = d[cols]
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(4)
    return d.to_markdown(index=False) + "\n"


def _best_by_language(df: pd.DataFrame, model_cols: list[str],
                      metric: str, ascending: bool = True) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    d = df.dropna(subset=[metric])
    if d.empty:
        return pd.DataFrame()
    idx = d.groupby("language")[metric].idxmin() if ascending \
        else d.groupby("language")[metric].idxmax()
    return d.loc[idx, ["language"] + model_cols + [metric]].sort_values("language")


def generate_summary_md(run_dir: str | Path, tables: dict[str, pd.DataFrame]) -> str:
    run_dir = Path(run_dir)
    manifest = load_json(run_dir / "run_manifest.json")
    model_cfgs = manifest.get("model_configs", {})
    track = manifest.get("track", "?")

    def is_dummy(mid: str) -> bool:
        return (model_cfgs.get(mid, {}) or {}).get("family") == "dummy"

    dummy_shown = False

    def strip_dummy(df: pd.DataFrame) -> pd.DataFrame:
        # Drop dummy adapters from ranked sections — unless the run contains
        # nothing else (pure smoke run), in which case show them with a flag.
        nonlocal dummy_shown
        if df.empty:
            return df
        stripped = df
        for col in ("asr_model", "diar_model"):
            if col in stripped.columns:
                stripped = stripped[~stripped[col].map(is_dummy)]
        if stripped.empty and not df.empty:
            dummy_shown = True
            return df
        return stripped

    asr_lang = strip_dummy(tables.get("asr_by_language", pd.DataFrame()))
    diar_lang = strip_dummy(tables.get("diar_by_language", pd.DataFrame()))
    pair_lang = strip_dummy(tables.get("pair_by_language", pd.DataFrame()))
    asr_model = strip_dummy(tables.get("asr_by_model", pd.DataFrame()))
    diar_model = strip_dummy(tables.get("diar_by_model", pd.DataFrame()))
    stack = strip_dummy(tables.get("pair_by_stack", pd.DataFrame()))
    failures = tables.get("failures", pd.DataFrame())

    lines: list[str] = []
    a = lines.append
    a(f"# Benchmark summary — `{manifest.get('run_id', run_dir.name)}`\n")
    a(f"- **Track:** {track}")
    a(f"- **Profile:** {manifest.get('profile')}")
    a(f"- **Languages:** {', '.join(manifest.get('languages', []))}")
    a(f"- **Recordings:** {len(manifest.get('recordings', []))}")
    a(f"- **ASR models:** {', '.join(manifest.get('asr_models', []))}")
    a(f"- **Diarization models:** {', '.join(manifest.get('diarization_models', []))}")
    a(f"- **Status:** {manifest.get('status')}")
    env = manifest.get("environment", {})
    gpus = env.get("gpus") or []
    if gpus:
        a(f"- **GPUs:** " + "; ".join(
            f"{g['name']} ({g['vram_total_mb']/1000:.0f} GB)" for g in gpus))
    a(f"- **Host:** {env.get('os')} / Python {env.get('python_version')}\n")
    a("> Chinese scores use character-level error (token_unit=char); WER "
      "values are not directly comparable across languages — compare models "
      "*within* a language, and use the macro/consistency columns across.\n")
    if any(is_dummy(m) for m in manifest.get("asr_models", [])
           + manifest.get("diarization_models", [])) and \
            all(is_dummy(m) for m in manifest.get("asr_models", [])
                + manifest.get("diarization_models", [])):
        a("> ⚠️ **Dummy-adapter run** — results below validate the pipeline "
          "only and say nothing about real models.\n")

    a("## Best ASR by language (WER ↓)\n")
    a(_md(_best_by_language(asr_lang, ["asr_model"], "wer"),
          ["language", "asr_model", "wer"]))
    a("## Best diarizer by language (DER ↓)\n")
    a(_md(_best_by_language(diar_lang, ["diar_model"], "der"),
          ["language", "diar_model", "der"]))
    a("## Best combined stack by language (cpWER ↓)\n")
    a(_md(_best_by_language(pair_lang, ["asr_model", "diar_model"], "cpwer"),
          ["language", "asr_model", "diar_model", "cpwer"]))

    a(f"## {track.upper()} stack leaderboard (macro across languages)\n")
    a("Sorted by macro cpWER; check the worst-language and std columns for "
      "cross-language consistency before picking a winner.\n")
    a(_md(stack, ["asr_model", "diar_model", "cpwer", "cpwer_worst_language",
                  "cpwer_std_across_languages", "word_attribution_accuracy",
                  "sentence_attribution_accuracy", "real_time_factor",
                  "peak_ram_mb", "peak_vram_mb", "n_languages"],
          sort="cpwer"))

    a("## ASR models (macro across languages)\n")
    a(_md(asr_model, ["asr_model", "wer", "wer_worst_language",
                      "wer_std_across_languages", "cer", "real_time_factor",
                      "model_load_time_sec", "peak_ram_mb", "peak_vram_mb",
                      "time_to_first_text_sec",
                      "sentence_finalization_delay_sec"], sort="wer"))
    a("## Diarization models (macro across languages)\n")
    a(_md(diar_model, ["diar_model", "der", "der_worst_language",
                       "der_std_across_languages", "missed_speech",
                       "false_alarm_speech", "speaker_confusion",
                       "speaker_count_error", "real_time_factor",
                       "peak_ram_mb", "peak_vram_mb"], sort="der"))

    lang_det = tables.get("language_detection", pd.DataFrame())
    if not lang_det.empty:
        a("## Language detection accuracy\n")
        a(_md(lang_det, ["asr_model", "language", "detection_accuracy"]))

    a("## License / deployment flags\n")
    flags = []
    for mid, cfg in model_cfgs.items():
        cfg = cfg or {}
        notes = []
        wl = str(cfg.get("weights_license", "") or "")
        if "NC" in wl.upper():
            notes.append("**NON-COMMERCIAL weights — excluded from recommendations**")
        if cfg.get("gated"):
            notes.append("gated download (accept terms + HF token)")
        if "BY" in wl.upper():
            notes.append("attribution required")
        if cfg.get("deployment_notes"):
            notes.append(str(cfg["deployment_notes"]))
        if notes:
            flags.append(f"- `{mid}` ({wl or cfg.get('license', '?')}): "
                         + "; ".join(notes))
    a("\n".join(flags) + "\n" if flags else "_none_\n")

    a("## Failed / incomplete experiments\n")
    if failures.empty:
        a("_none_\n")
    else:
        a(_md(failures, ["row_type", "recording_id", "asr_model", "diar_model",
                         "language", "status", "error"], n=50))
        a(f"\nTotal non-completed rows: {len(failures)}. "
          "Structured error records: `errors/`.\n")

    a("---\n_Not a single blended score by design — see docs/methodology.md. "
      "Generated from cached results only._\n")
    text = "\n".join(lines)
    atomic_write_text(run_dir / "reports" / "summary.md", text)
    # summary.csv: the stack leaderboard as machine-readable summary
    if not stack.empty:
        stack.to_csv(run_dir / "reports" / "summary.csv", index=False)
    return text
