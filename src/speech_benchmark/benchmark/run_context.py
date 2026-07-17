"""Run folder layout, run manifest, run index, and structured error records.

Layout (see README / spec):
    artifacts/runs/<run_id>/
        run_manifest.json
        config/            copies of the exact configs used
        environment/       environment.json, pip_freeze.txt
        logs/              one log per stage/model
        predictions/{asr,diarization,combined}/
        metrics/{per_recording,per_language,per_model,per_stack}/
        tables/  charts/  errors/  reports/
    artifacts/latest -> runs/<run_id>   (symlink)
    artifacts/index.csv                 (one row per run)
"""

from __future__ import annotations

import csv
import logging
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path

from ..schemas import atomic_write_json, load_json, utc_now_iso

STAGES = ("asr", "diarization", "fusion", "metrics", "report")


def make_run_id(track: str, profile: str, tag: str = "v1") -> str:
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{date}_{track}_{profile}_{tag}"


class RunContext:
    def __init__(self, artifacts_dir: str | Path, run_id: str):
        self.run_id = run_id
        self.artifacts_dir = Path(artifacts_dir)
        self.run_dir = self.artifacts_dir / "runs" / run_id
        for sub in ("config", "environment", "logs",
                    "predictions/asr", "predictions/diarization", "predictions/combined",
                    "metrics/per_recording", "metrics/per_language",
                    "metrics/per_model", "metrics/per_stack",
                    "tables", "charts", "errors", "reports"):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        self._loggers: dict[str, logging.Logger] = {}

    # -- manifest -----------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    def read_manifest(self) -> dict:
        if self.manifest_path.exists():
            return load_json(self.manifest_path)
        return {}

    def update_manifest(self, **fields) -> dict:
        m = self.read_manifest()
        m.update(fields)
        m.setdefault("run_id", self.run_id)
        atomic_write_json(self.manifest_path, m)
        return m

    def copy_config(self, path: str | Path) -> None:
        p = Path(path)
        if p.exists():
            shutil.copy2(p, self.run_dir / "config" / p.name)

    def register_in_index(self, **row) -> None:
        index = self.artifacts_dir / "index.csv"
        fields = ["run_id", "started_at", "completed_at", "track", "profile",
                  "status", "num_recordings", "num_asr_models", "num_diar_models"]
        rows: list[dict] = []
        if index.exists():
            with open(index, newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if r.get("run_id") != self.run_id]
        rows.append({k: row.get(k, "") for k in fields})
        with open(index, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        latest = self.artifacts_dir / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(self.run_dir.relative_to(self.artifacts_dir))
        except OSError:
            pass  # symlinks may fail on some filesystems; index.csv still points there

    # -- predictions --------------------------------------------------------
    def asr_path(self, model_id: str, recording_id: str) -> Path:
        return self.run_dir / "predictions" / "asr" / _safe(model_id) / f"{_safe(recording_id)}.json"

    def diar_path(self, model_id: str, recording_id: str) -> Path:
        return self.run_dir / "predictions" / "diarization" / _safe(model_id) / f"{_safe(recording_id)}.json"

    def rttm_path(self, model_id: str, recording_id: str) -> Path:
        return self.run_dir / "predictions" / "diarization" / _safe(model_id) / f"{_safe(recording_id)}.rttm"

    def combined_path(self, asr_id: str, diar_id: str, recording_id: str) -> Path:
        return (self.run_dir / "predictions" / "combined"
                / f"{_safe(asr_id)}__{_safe(diar_id)}" / f"{_safe(recording_id)}.json")

    # -- logging ------------------------------------------------------------
    def logger(self, name: str) -> logging.Logger:
        if name in self._loggers:
            return self._loggers[name]
        logger = logging.getLogger(f"speech_benchmark.{self.run_id}.{name}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh = logging.FileHandler(self.run_dir / "logs" / f"{_safe(name)}.log",
                                 encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        self._loggers[name] = logger
        return logger

    # -- structured errors --------------------------------------------------
    def record_error(self, stage: str, model_id: str | None,
                     recording_id: str | None, exc: BaseException,
                     gpu_state: dict | None = None) -> None:
        rec = {
            "stage": stage,
            "model": model_id,
            "recording": recording_id,
            "timestamp": utc_now_iso(),
            "exception_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "gpu_state": gpu_state,
        }
        name = f"{_safe(stage)}__{_safe(model_id or 'na')}__{_safe(recording_id or 'na')}.json"
        atomic_write_json(self.run_dir / "errors" / name, rec)


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
