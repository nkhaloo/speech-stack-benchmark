"""Charts (PNG) for the run report. Skipped gracefully if matplotlib is
missing. One chart per metric, models on the x-axis grouped by language."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _grouped_bar(df: pd.DataFrame, model_cols: list[str], metric: str,
                 title: str, out_path: Path, ascending: bool = True) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty or metric not in df.columns:
        return
    d = df.dropna(subset=[metric]).copy()
    if d.empty:
        return
    d["model"] = d[model_cols].astype(str).agg(" + ".join, axis=1)
    pivot = d.pivot_table(index="model", columns="language", values=metric,
                          aggfunc="mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=ascending).index]

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(pivot)), 4.5))
    pivot.plot.bar(ax=ax, width=0.8)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xlabel("")
    ax.legend(title="language", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_charts(run_dir: str | Path, tables: dict[str, pd.DataFrame]) -> list[str]:
    run_dir = Path(run_dir)
    charts_dir = run_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    try:
        specs = [
            ("asr_by_language", ["asr_model"], "wer", "WER by ASR model and language"),
            ("asr_by_language", ["asr_model"], "real_time_factor",
             "ASR real-time factor (lower = faster)"),
            ("asr_by_language", ["asr_model"], "peak_ram_mb", "ASR peak RAM (MB)"),
            ("asr_by_language", ["asr_model"], "peak_vram_mb", "ASR peak VRAM (MB)"),
            ("diar_by_language", ["diar_model"], "der", "DER by diarizer and language"),
            ("diar_by_language", ["diar_model"], "peak_vram_mb", "Diarizer peak VRAM (MB)"),
            ("pair_by_language", ["asr_model", "diar_model"], "cpwer",
             "Speaker-attributed WER (cpWER) by stack and language"),
            ("pair_by_language", ["asr_model", "diar_model"], "runtime_sec",
             "Total stack runtime (s)"),
        ]
        for table, model_cols, metric, title in specs:
            out = charts_dir / f"{table}_{metric}.png"
            _grouped_bar(tables.get(table, pd.DataFrame()), model_cols, metric,
                         title, out)
            if out.exists():
                made.append(str(out))
    except ImportError:
        pass  # matplotlib not installed; tables/CSVs still cover everything
    return made
