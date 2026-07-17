"""Export a compact, portable results bundle for a run (no model weights,
no datasets; raw predictions optionally compressed)."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

from ..schemas import atomic_write_json, utc_now_iso


def export_run(run_dir: str | Path, output_dir: str | Path,
               include_predictions: bool = False) -> Path:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    if not (run_dir / "run_manifest.json").exists():
        raise FileNotFoundError(f"not a run folder: {run_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for item in ("run_manifest.json", "config", "environment", "metrics",
                 "tables", "charts", "reports", "errors"):
        src = run_dir / item
        dst = output_dir / item
        if not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        copied.append(item)

    if include_predictions and (run_dir / "predictions").exists():
        archive = output_dir / "predictions.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(run_dir / "predictions", arcname="predictions")
        copied.append("predictions.tar.gz")

    atomic_write_json(output_dir / "export_info.json", {
        "run_id": run_dir.name,
        "exported_at": utc_now_iso(),
        "contents": copied,
        "includes_predictions": include_predictions,
        "note": "Model weights and datasets are never exported.",
    })
    return output_dir
