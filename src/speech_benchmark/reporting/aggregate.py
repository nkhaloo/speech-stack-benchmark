"""Aggregation of per-recording metrics into per-language / per-model /
per-stack tables, leaderboards, and a failures table.

Operates purely on saved metrics JSON — no models are ever loaded here.
Dummy adapters (family == "dummy") are kept in the raw tables but excluded
from leaderboards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..schemas import load_json


def load_rows(run_dir: Path) -> dict[str, pd.DataFrame]:
    base = run_dir / "metrics" / "per_recording"
    out = {}
    for name in ("asr_rows", "diar_rows", "pair_rows"):
        p = base / f"{name}.json"
        out[name] = pd.DataFrame(load_json(p)) if p.exists() else pd.DataFrame()
    return out


def _completed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "status" not in df:
        return df
    return df[df["status"] == "completed"].copy()


def _agg(df: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    present = [m for m in metrics if m in df.columns]
    if df.empty or not present:
        return pd.DataFrame()
    agg = df.groupby(keys, dropna=False)[present].mean(numeric_only=True).reset_index()
    counts = df.groupby(keys, dropna=False).size().rename("n_recordings").reset_index()
    return agg.merge(counts, on=keys)


ASR_METRICS = ["wer", "cer", "real_time_factor", "runtime_sec", "model_load_time_sec",
               "peak_ram_mb", "peak_vram_mb", "time_to_first_text_sec",
               "sentence_finalization_delay_sec", "chunked_rtf"]
DIAR_METRICS = ["der", "missed_speech", "false_alarm_speech", "speaker_confusion",
                "speaker_count_error", "real_time_factor", "runtime_sec",
                "model_load_time_sec", "peak_ram_mb", "peak_vram_mb"]
PAIR_METRICS = ["cpwer", "wer", "word_attribution_accuracy",
                "sentence_attribution_accuracy", "real_time_factor",
                "runtime_sec", "peak_ram_mb", "peak_vram_mb"]


def _macro_over_languages(per_lang: pd.DataFrame, model_keys: list[str],
                          primary: str) -> pd.DataFrame:
    """Macro-average across languages + consistency (worst language, std)."""
    if per_lang.empty:
        return pd.DataFrame()
    num_cols = [c for c in per_lang.columns
                if c not in model_keys + ["language"] and
                pd.api.types.is_numeric_dtype(per_lang[c])]
    g = per_lang.groupby(model_keys, dropna=False)
    out = g[num_cols].mean().reset_index()
    if primary in per_lang.columns:
        extra = g[primary].agg(["max", "std", "count"]).reset_index().rename(columns={
            "max": f"{primary}_worst_language",
            "std": f"{primary}_std_across_languages",
            "count": "n_languages"})
        out = out.merge(extra, on=model_keys)
    return out


def aggregate_run(run_dir: str | Path) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    rows = load_rows(run_dir)
    manifest = load_json(run_dir / "run_manifest.json")
    model_cfgs = manifest.get("model_configs", {})

    def is_dummy(model_id) -> bool:
        return (model_cfgs.get(model_id, {}) or {}).get("family") == "dummy" or \
            (isinstance(model_id, str) and model_id.startswith("dummy"))

    tables: dict[str, pd.DataFrame] = {}

    asr = _completed(rows["asr_rows"])
    diar = _completed(rows["diar_rows"])
    pair = _completed(rows["pair_rows"])

    tables["asr_by_language"] = _agg(asr, ["asr_model", "language"], ASR_METRICS)
    tables["diar_by_language"] = _agg(diar, ["diar_model", "language"], DIAR_METRICS)
    tables["pair_by_language"] = _agg(pair, ["asr_model", "diar_model", "language"],
                                      PAIR_METRICS)

    tables["asr_by_model"] = _macro_over_languages(
        tables["asr_by_language"], ["asr_model"], "wer")
    tables["diar_by_model"] = _macro_over_languages(
        tables["diar_by_language"], ["diar_model"], "der")
    tables["pair_by_stack"] = _macro_over_languages(
        tables["pair_by_language"], ["asr_model", "diar_model"], "cpwer")

    # Leaderboards: real models only, sorted by primary quality metric.
    lb = tables["pair_by_stack"]
    if not lb.empty:
        mask = ~(lb["asr_model"].map(is_dummy) | lb["diar_model"].map(is_dummy))
        lb = lb[mask]
        tables["leaderboard"] = lb.sort_values("cpwer", na_position="last") \
            if "cpwer" in lb.columns else lb
    else:
        tables["leaderboard"] = lb

    # Failures: everything not completed, all three row types.
    fails = []
    for name, df in rows.items():
        if df.empty or "status" not in df:
            continue
        f = df[df["status"] != "completed"].copy()
        if not f.empty:
            f["row_type"] = name
            fails.append(f)
    tables["failures"] = pd.concat(fails, ignore_index=True) if fails else pd.DataFrame()

    # Language-detection accuracy per ASR model, when available.
    if not asr.empty and "language_detection_correct" in asr.columns:
        det = asr.dropna(subset=["language_detection_correct"])
        if not det.empty:
            det = det.copy()
            det["language_detection_correct"] = det["language_detection_correct"].astype(float)
            tables["language_detection"] = (
                det.groupby(["asr_model", "language"])["language_detection_correct"]
                .mean().reset_index()
                .rename(columns={"language_detection_correct": "detection_accuracy"}))

    return tables


def write_tables(run_dir: str | Path, tables: dict[str, pd.DataFrame]) -> None:
    run_dir = Path(run_dir)
    manifest = load_json(run_dir / "run_manifest.json")
    track = manifest.get("track", "cpu")

    tdir = run_dir / "tables"
    mdir = run_dir / "metrics"
    rdir = run_dir / "reports"
    for d in (tdir, rdir):
        d.mkdir(parents=True, exist_ok=True)

    mapping = {
        "asr_by_language": mdir / "per_language" / "asr_by_language.csv",
        "diar_by_language": mdir / "per_language" / "diar_by_language.csv",
        "pair_by_language": mdir / "per_language" / "pair_by_language.csv",
        "asr_by_model": mdir / "per_model" / "asr_by_model.csv",
        "diar_by_model": mdir / "per_model" / "diar_by_model.csv",
        "pair_by_stack": mdir / "per_stack" / "pair_by_stack.csv",
        "language_detection": mdir / "per_model" / "language_detection.csv",
        "failures": rdir / "failures.csv",
        "leaderboard": rdir / f"leaderboard_{track}.csv",
    }
    for name, df in tables.items():
        path = mapping.get(name, tdir / f"{name}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        # convenience copies in reports/ for the export bundle
        if name in ("asr_by_language", "diar_by_language", "pair_by_language"):
            df.to_csv(rdir / "results_by_language.csv" if name == "pair_by_language"
                      else rdir / f"{name}.csv", index=False)
        if name == "asr_by_model":
            df.to_csv(rdir / "results_by_model.csv", index=False)
        if name == "pair_by_stack":
            df.to_csv(rdir / "results_by_stack.csv", index=False)
