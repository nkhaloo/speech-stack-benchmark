"""Streaming benchmark orchestrator (see docs/methodology_streaming.md).

Parallel to the batch ``BenchmarkRunner`` but for streaming stacks:

  1. Each streaming stack is driven once per recording under a simulated
     real-time clock; its ordered emission log is cached as a ``StreamingResult``.
  2. Metrics (final accuracy + latency + stability) are computed from the cached
     emission logs and references only.

One stack loaded at a time; resumable (completed emission logs are skipped
unless ``force``); a failing stack/recording becomes a ``failed`` record and
never aborts the run; an unavailable stack (missing deps/weights) marks all its
recordings ``skipped`` and continues.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..audio import load_audio
from ..benchmark.environment import collect_environment, pip_freeze
from ..benchmark.resources import ResourceMonitor
from ..benchmark.run_context import RunContext
from ..schemas import (Recording, StreamingResult, atomic_write_json,
                       atomic_write_text, load_json, utc_now_iso)
from . import create_streaming_adapter
from .base import AdapterUnavailable
from .metrics import streaming_metrics

SAMPLE_RATE = 16000


class StreamingRunner:
    def __init__(self, track_config: dict, ctx: RunContext,
                 recordings: list[Recording], force: bool = False,
                 config_paths: Optional[list[str]] = None,
                 cli_args: Optional[list[str]] = None):
        self.cfg = track_config
        self.ctx = ctx
        self.force = force
        self.track = track_config.get("track", "streaming")
        languages = track_config.get("languages")
        self.recordings = [r for r in recordings
                           if not languages or r.language in languages]
        self.stacks = track_config.get("streaming_stacks", [])
        self.config_paths = config_paths or []
        self.cli_args = cli_args or []
        s = track_config.get("streaming", {}) or {}
        self.frame_sec = float(s.get("frame_sec", 0.5))
        self.latency_budget_sec = float(s.get("latency_budget_sec", 2.0))

    # ------------------------------------------------------------------ setup
    def _init_manifest(self) -> None:
        env = collect_environment()
        atomic_write_json(self.ctx.run_dir / "environment" / "environment.json", env)
        atomic_write_text(self.ctx.run_dir / "environment" / "pip_freeze.txt",
                          "\n".join(pip_freeze()) + "\n")
        for p in self.config_paths:
            self.ctx.copy_config(p)
        self.ctx.update_manifest(
            run_id=self.ctx.run_id, started_at=utc_now_iso(), status="running",
            track=self.track, profile=self.cfg.get("profile"),
            languages=sorted({r.language for r in self.recordings}),
            recordings=[r.recording_id for r in self.recordings],
            streaming_stacks=[s["id"] for s in self.stacks],
            model_configs={s["id"]: s for s in self.stacks},
            datasets=sorted({r.dataset for r in self.recordings}),
            cli_args=self.cli_args,
            config_files=[str(p) for p in self.config_paths],
            streaming_contract={"frame_sec": self.frame_sec,
                                "latency_budget_sec": self.latency_budget_sec},
            metric_settings=self.cfg.get("metrics", {}),
            environment=env,
        )
        self.ctx.register_in_index(
            run_id=self.ctx.run_id, started_at=utc_now_iso(), track=self.track,
            profile=self.cfg.get("profile"), status="running",
            num_recordings=len(self.recordings),
            num_asr_models=len(self.stacks), num_diar_models=0)

    # -------------------------------------------------------------- streaming
    def run_streaming_stage(self) -> None:
        for scfg in self.stacks:
            stack_id = scfg["id"]
            log = self.ctx.logger(f"streaming_{stack_id}")
            adapter = create_streaming_adapter(scfg)
            load_time: Optional[float] = None
            loaded = False
            for rec in self.recordings:
                out_path = self.ctx.streaming_path(stack_id, rec.recording_id)
                if not self.force and out_path.exists():
                    try:
                        if load_json(out_path).get("status") == "completed":
                            log.info("skip cached %s", rec.recording_id)
                            continue
                    except Exception:
                        pass
                if not loaded:
                    try:
                        t0 = time.perf_counter()
                        adapter.load()
                        load_time = time.perf_counter() - t0
                        loaded = True
                        log.info("loaded %s in %.1fs", stack_id, load_time)
                    except AdapterUnavailable as e:
                        log.warning("stack unavailable, skipping: %s", e)
                        self.ctx.record_error("streaming", stack_id, None, e)
                        self._mark_skipped(scfg, str(e))
                        break
                log.info("streaming %s (%.0fs audio)", rec.recording_id,
                         rec.duration_sec or 0)
                result = self._run_one(adapter, scfg, rec, load_time)
                atomic_write_json(out_path, result.to_dict())
                log.info("-> %s (%d emissions)", result.status, len(result.emissions))
            if loaded:
                adapter.unload()
                log.info("unloaded %s", stack_id)

    def _run_one(self, adapter, scfg: dict, rec: Recording,
                 load_time) -> StreamingResult:
        gpu_index = int(self.cfg.get("gpu_index", 0))
        meta = dict(self.cfg.get("streaming", {}) or {})
        meta.update(adapter.contract_meta())
        try:
            audio, sr = load_audio(rec.audio_path, SAMPLE_RATE)
            total_sec = len(audio) / sr
            emissions = []
            compute_total = 0.0
            with ResourceMonitor(gpu_index=gpu_index) as mon:
                adapter.reset(rec)
                frame = int(self.frame_sec * sr)
                n = max(1, -(-len(audio) // frame))
                sim_clock = 0.0
                for k in range(n):
                    a0, a1 = k * frame, min(len(audio), (k + 1) * frame)
                    audio_time_end = min(total_sec, a1 / sr)
                    t0 = time.perf_counter()
                    ems = adapter.push(audio[a0:a1], audio_time_end)
                    dt = time.perf_counter() - t0
                    compute_total += dt
                    frame_clock = max(sim_clock, audio_time_end)
                    for e in ems:
                        offset = (e.processing_offset_sec
                                  if e.processing_offset_sec is not None else dt)
                        e.wall_time = frame_clock + min(max(0.0, offset), dt)
                    sim_clock = frame_clock + dt
                    emissions.extend(ems)
                t0 = time.perf_counter()
                ems = adapter.flush()
                dt = time.perf_counter() - t0
                compute_total += dt
                sim_clock = max(sim_clock, total_sec) + dt
                for e in ems:
                    e.wall_time = sim_clock
                emissions.extend(ems)
            return StreamingResult(
                recording_id=rec.recording_id, model_id=scfg["id"],
                emissions=emissions, audio_duration_sec=total_sec,
                runtime_sec=compute_total, load_time_sec=load_time,
                resources=mon.stats, model_meta=adapter.model_meta(),
                streaming_meta=meta, status="completed")
        except Exception as e:
            self.ctx.record_error("streaming", scfg["id"], rec.recording_id, e)
            return StreamingResult(
                recording_id=rec.recording_id, model_id=scfg["id"],
                audio_duration_sec=rec.duration_sec, load_time_sec=load_time,
                streaming_meta=meta, status="failed",
                error=f"{type(e).__name__}: {e}")

    def _mark_skipped(self, scfg: dict, reason: str) -> None:
        for rec in self.recordings:
            out_path = self.ctx.streaming_path(scfg["id"], rec.recording_id)
            if not out_path.exists():
                atomic_write_json(out_path, StreamingResult(
                    recording_id=rec.recording_id, model_id=scfg["id"],
                    status="skipped", error=reason).to_dict())

    # ---------------------------------------------------------------- metrics
    def run_metrics_stage(self) -> None:
        log = self.ctx.logger("streaming_metrics")
        mset = self.cfg.get("metrics", {})
        collar = float(mset.get("der_collar", 0.5))
        skip_overlap = bool(mset.get("der_skip_overlap", False))
        rows = []
        for rec in self.recordings:
            ref = rec.load_reference()
            base = {"run_id": self.ctx.run_id, "track": self.track,
                    "dataset": rec.dataset, "recording_id": rec.recording_id,
                    "language": rec.language, "audio_duration_sec": rec.duration_sec}
            for scfg in self.stacks:
                row = dict(base, stack=scfg["id"], family=scfg.get("family"),
                           native=bool(scfg.get("native", False)),
                           latency_budget_sec=self.latency_budget_sec)
                path = self.ctx.streaming_path(scfg["id"], rec.recording_id)
                if not path.exists():
                    rows.append(dict(row, status="missing"))
                    continue
                sr = StreamingResult.from_dict(load_json(path))
                row.update(status=sr.status, error=sr.error,
                           runtime_sec=sr.runtime_sec,
                           streaming_rtf=sr.real_time_factor,
                           model_load_time_sec=sr.load_time_sec,
                           peak_ram_mb=sr.resources.peak_ram_mb if sr.resources else None,
                           peak_vram_mb=sr.resources.peak_vram_mb if sr.resources else None)
                if sr.status == "completed":
                    try:
                        row.update(streaming_metrics(sr, ref, rec.language,
                                                     collar=collar,
                                                     skip_overlap=skip_overlap))
                    except Exception as e:
                        self.ctx.record_error("streaming_metrics", scfg["id"],
                                              rec.recording_id, e)
                        row.update(status="failed", error=f"metrics: {e}")
                rows.append(row)
        out_dir = self.ctx.run_dir / "metrics" / "per_recording"
        atomic_write_json(out_dir / "streaming_rows.json", rows)
        _rows_to_csv(out_dir / "streaming_rows.csv", rows)
        log.info("streaming metrics written: %d rows", len(rows))

    # ------------------------------------------------------------------- run
    def run(self) -> None:
        self._init_manifest()
        status = "completed"
        try:
            self.run_streaming_stage()
            self.run_metrics_stage()
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        except Exception as e:
            status = "failed"
            self.ctx.record_error("streaming_runner", None, None, e)
            raise
        finally:
            self.ctx.update_manifest(completed_at=utc_now_iso(), status=status)
            self.ctx.register_in_index(
                run_id=self.ctx.run_id,
                started_at=self.ctx.read_manifest().get("started_at"),
                completed_at=utc_now_iso(), track=self.track,
                profile=self.cfg.get("profile"), status=status,
                num_recordings=len(self.recordings),
                num_asr_models=len(self.stacks), num_diar_models=0)


def _rows_to_csv(path, rows: list[dict]) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
