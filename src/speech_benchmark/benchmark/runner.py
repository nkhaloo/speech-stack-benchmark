"""Benchmark orchestrator.

Design (see docs/methodology.md):
  1. Each ASR model runs once per recording; normalized output is cached.
  2. Each diarization model runs once per recording; output is cached.
  3. Cached outputs are fused into every valid (ASR × diarization) pair.
  4. Metrics are computed from cached predictions only.

Only one model is loaded at a time. Every stage is resumable: existing
completed outputs are skipped unless ``force`` is set. A failed model or
recording never blocks the rest — failures become structured error records
and explicit failed rows in the metrics tables.
"""

from __future__ import annotations

import time
from typing import Optional

from ..asr import AdapterUnavailable, create_asr_adapter
from ..diarization import create_diarization_adapter
from ..fusion import CombinedResult, fuse
from ..metrics.asr_metrics import asr_metrics
from ..metrics.combined_metrics import combined_metrics
from ..metrics.diar_metrics import diarization_metrics
from ..schemas import (ASRResult, DiarizationResult, Recording,
                       atomic_write_json, atomic_write_text, load_json,
                       utc_now_iso)
from .environment import collect_environment, pip_freeze
from .resources import ResourceMonitor
from .run_context import RunContext
from .streaming import ChunkedConfig, run_chunked_simulation


class BenchmarkRunner:
    def __init__(self, track_config: dict, ctx: RunContext,
                 recordings: list[Recording], force: bool = False,
                 config_paths: Optional[list[str]] = None,
                 cli_args: Optional[list[str]] = None):
        self.cfg = track_config
        self.ctx = ctx
        self.force = force
        self.track = track_config.get("track", "cpu")
        languages = track_config.get("languages")
        self.recordings = [r for r in recordings
                           if not languages or r.language in languages]
        self.config_paths = config_paths or []
        self.cli_args = cli_args or []

    # ------------------------------------------------------------------ setup
    def _init_manifest(self) -> None:
        env = collect_environment()
        atomic_write_json(self.ctx.run_dir / "environment" / "environment.json", env)
        atomic_write_text(self.ctx.run_dir / "environment" / "pip_freeze.txt",
                          "\n".join(pip_freeze()) + "\n")
        for p in self.config_paths:
            self.ctx.copy_config(p)
        self.ctx.update_manifest(
            run_id=self.ctx.run_id,
            started_at=utc_now_iso(),
            status="running",
            track=self.track,
            profile=self.cfg.get("profile"),
            languages=sorted({r.language for r in self.recordings}),
            recordings=[r.recording_id for r in self.recordings],
            asr_models=[m["id"] for m in self.cfg.get("asr_models", [])],
            diarization_models=[m["id"] for m in self.cfg.get("diarization_models", [])],
            model_configs={m["id"]: m for m in
                           self.cfg.get("asr_models", []) + self.cfg.get("diarization_models", [])},
            datasets=sorted({r.dataset for r in self.recordings}),
            cli_args=self.cli_args,
            config_files=[str(p) for p in self.config_paths],
            seeds={"dataset_seed": self.cfg.get("dataset_seed")},
            metric_settings=self.cfg.get("metrics", {}),
            environment=env,
        )
        self.ctx.register_in_index(
            run_id=self.ctx.run_id, started_at=utc_now_iso(), track=self.track,
            profile=self.cfg.get("profile"), status="running",
            num_recordings=len(self.recordings),
            num_asr_models=len(self.cfg.get("asr_models", [])),
            num_diar_models=len(self.cfg.get("diarization_models", [])),
        )

    # ------------------------------------------------------------------- ASR
    def run_asr_stage(self) -> None:
        force_language = bool(self.cfg.get("force_language", False))
        for mcfg in self.cfg.get("asr_models", []):
            model_id = mcfg["id"]
            log = self.ctx.logger(f"asr_{model_id}")
            adapter = create_asr_adapter(mcfg)
            load_time: Optional[float] = None
            loaded = False
            for rec in self.recordings:
                out_path = self.ctx.asr_path(model_id, rec.recording_id)
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
                        log.info("loaded %s in %.1fs", model_id, load_time)
                    except AdapterUnavailable as e:
                        log.warning("adapter unavailable, skipping model: %s", e)
                        self.ctx.record_error("asr", model_id, None, e)
                        self._mark_skipped_asr(mcfg, str(e))
                        break
                language = rec.language if force_language else None
                log.info("transcribing %s (%.0fs audio)", rec.recording_id,
                         rec.duration_sec or 0)
                result = self._run_one_asr(adapter, mcfg, rec, language, load_time)
                result.load_time_sec = load_time
                atomic_write_json(out_path, result.to_dict())
                log.info("-> %s (%s)", result.status, out_path.name)
            if loaded:
                # streaming simulation on the first recording per language
                if mcfg.get("chunked_eval", False):
                    self._run_streaming_sim(adapter, mcfg, log)
                adapter.unload()
                log.info("unloaded %s", model_id)

    def _run_one_asr(self, adapter, mcfg: dict, rec: Recording,
                     language: Optional[str], load_time) -> ASRResult:
        gpu_index = int(self.cfg.get("gpu_index", 0))
        try:
            with ResourceMonitor(gpu_index=gpu_index) as mon:
                t0 = time.perf_counter()
                result = adapter.transcribe(rec, language=language)
                runtime = time.perf_counter() - t0
            result.runtime_sec = runtime
            result.audio_duration_sec = rec.duration_sec
            result.resources = mon.stats
            result.status = "completed"
            return result
        except Exception as e:
            self.ctx.record_error("asr", mcfg["id"], rec.recording_id, e)
            return ASRResult(
                recording_id=rec.recording_id, model_id=mcfg["id"],
                language_requested=language, audio_duration_sec=rec.duration_sec,
                status="failed", error=f"{type(e).__name__}: {e}",
            )

    def _mark_skipped_asr(self, mcfg: dict, reason: str) -> None:
        for rec in self.recordings:
            out_path = self.ctx.asr_path(mcfg["id"], rec.recording_id)
            if not out_path.exists():
                atomic_write_json(out_path, ASRResult(
                    recording_id=rec.recording_id, model_id=mcfg["id"],
                    status="skipped", error=reason,
                ).to_dict())

    def _run_streaming_sim(self, adapter, mcfg: dict, log) -> None:
        chunk_cfg = ChunkedConfig(**(self.cfg.get("chunked", {}) or {}))
        done_langs: set[str] = set()
        for rec in self.recordings:
            if rec.language in done_langs:
                continue
            done_langs.add(rec.language)
            out = self.ctx.asr_path(mcfg["id"], rec.recording_id)
            stream_path = out.with_suffix(".streaming.json")
            if not self.force and stream_path.exists():
                continue
            log.info("chunked simulation on %s", rec.recording_id)
            try:
                meta = run_chunked_simulation(adapter, rec, chunk_cfg)
                meta["recording_id"] = rec.recording_id
                meta["model_id"] = mcfg["id"]
                atomic_write_json(stream_path, meta)
            except Exception as e:
                self.ctx.record_error("asr_streaming", mcfg["id"], rec.recording_id, e)

    # ----------------------------------------------------------- diarization
    def run_diarization_stage(self) -> None:
        for mcfg in self.cfg.get("diarization_models", []):
            model_id = mcfg["id"]
            log = self.ctx.logger(f"diar_{model_id}")
            adapter = create_diarization_adapter(mcfg)
            load_time: Optional[float] = None
            loaded = False
            for rec in self.recordings:
                out_path = self.ctx.diar_path(model_id, rec.recording_id)
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
                        log.info("loaded %s in %.1fs", model_id, load_time)
                    except AdapterUnavailable as e:
                        log.warning("adapter unavailable, skipping model: %s", e)
                        self.ctx.record_error("diarization", model_id, None, e)
                        self._mark_skipped_diar(mcfg, str(e))
                        break
                log.info("diarizing %s", rec.recording_id)
                result = self._run_one_diar(adapter, mcfg, rec)
                result.load_time_sec = load_time
                atomic_write_json(out_path, result.to_dict())
                if result.status == "completed":
                    atomic_write_text(self.ctx.rttm_path(model_id, rec.recording_id),
                                      result.to_rttm())
                log.info("-> %s", result.status)
            if loaded:
                adapter.unload()
                log.info("unloaded %s", model_id)

    def _run_one_diar(self, adapter, mcfg: dict, rec: Recording) -> DiarizationResult:
        gpu_index = int(self.cfg.get("gpu_index", 0))
        try:
            with ResourceMonitor(gpu_index=gpu_index) as mon:
                t0 = time.perf_counter()
                result = adapter.diarize(rec)
                runtime = time.perf_counter() - t0
            result.runtime_sec = runtime
            result.audio_duration_sec = rec.duration_sec
            result.resources = mon.stats
            result.status = "completed"
            return result
        except Exception as e:
            self.ctx.record_error("diarization", mcfg["id"], rec.recording_id, e)
            return DiarizationResult(
                recording_id=rec.recording_id, model_id=mcfg["id"],
                audio_duration_sec=rec.duration_sec,
                status="failed", error=f"{type(e).__name__}: {e}",
            )

    def _mark_skipped_diar(self, mcfg: dict, reason: str) -> None:
        for rec in self.recordings:
            out_path = self.ctx.diar_path(mcfg["id"], rec.recording_id)
            if not out_path.exists():
                atomic_write_json(out_path, DiarizationResult(
                    recording_id=rec.recording_id, model_id=mcfg["id"],
                    status="skipped", error=reason,
                ).to_dict())

    # ---------------------------------------------------------------- fusion
    def run_fusion_stage(self) -> None:
        log = self.ctx.logger("fusion")
        gap = float(self.cfg.get("fusion", {}).get("sentence_gap_sec", 0.8))
        for asr_cfg in self.cfg.get("asr_models", []):
            for diar_cfg in self.cfg.get("diarization_models", []):
                for rec in self.recordings:
                    out_path = self.ctx.combined_path(
                        asr_cfg["id"], diar_cfg["id"], rec.recording_id)
                    if not self.force and out_path.exists():
                        continue
                    try:
                        asr = ASRResult.from_dict(load_json(
                            self.ctx.asr_path(asr_cfg["id"], rec.recording_id)))
                        diar = DiarizationResult.from_dict(load_json(
                            self.ctx.diar_path(diar_cfg["id"], rec.recording_id)))
                    except FileNotFoundError:
                        continue
                    if asr.status != "completed" or diar.status != "completed":
                        combined = CombinedResult(
                            recording_id=rec.recording_id,
                            asr_model_id=asr_cfg["id"],
                            diarization_model_id=diar_cfg["id"],
                            status="skipped",
                            error=f"asr={asr.status}, diar={diar.status}",
                        )
                    else:
                        try:
                            combined = fuse(asr, diar, gap_threshold_sec=gap)
                        except Exception as e:
                            self.ctx.record_error("fusion",
                                                  f"{asr_cfg['id']}+{diar_cfg['id']}",
                                                  rec.recording_id, e)
                            combined = CombinedResult(
                                recording_id=rec.recording_id,
                                asr_model_id=asr_cfg["id"],
                                diarization_model_id=diar_cfg["id"],
                                status="failed", error=str(e),
                            )
                    atomic_write_json(out_path, combined.to_dict())
        log.info("fusion complete")

    # --------------------------------------------------------------- metrics
    def run_metrics_stage(self) -> None:
        log = self.ctx.logger("metrics")
        mset = self.cfg.get("metrics", {})
        collar = float(mset.get("der_collar", 0.5))
        skip_overlap = bool(mset.get("der_skip_overlap", False))

        asr_rows, diar_rows, pair_rows = [], [], []
        for rec in self.recordings:
            ref = rec.load_reference()
            base = {
                "run_id": self.ctx.run_id, "track": self.track,
                "dataset": rec.dataset, "recording_id": rec.recording_id,
                "language": rec.language, "audio_duration_sec": rec.duration_sec,
            }
            asr_cache: dict[str, ASRResult] = {}
            diar_cache: dict[str, DiarizationResult] = {}

            for mcfg in self.cfg.get("asr_models", []):
                row = dict(base, asr_model=mcfg["id"], family=mcfg.get("family"))
                try:
                    asr = ASRResult.from_dict(load_json(
                        self.ctx.asr_path(mcfg["id"], rec.recording_id)))
                    asr_cache[mcfg["id"]] = asr
                except FileNotFoundError:
                    asr_rows.append(dict(row, status="missing"))
                    continue
                row.update(status=asr.status, error=asr.error,
                           runtime_sec=asr.runtime_sec,
                           real_time_factor=asr.real_time_factor,
                           model_load_time_sec=asr.load_time_sec,
                           peak_ram_mb=asr.resources.peak_ram_mb if asr.resources else None,
                           peak_vram_mb=asr.resources.peak_vram_mb if asr.resources else None,
                           language_detected=asr.language_detected,
                           language_detection_correct=(
                               None if asr.language_detected is None else
                               _lang_match(asr.language_detected, rec.language)),
                           has_word_timestamps=any(
                               s.words for s in asr.segments) or None)
                if asr.status == "completed" and ref is not None:
                    row.update(asr_metrics(ref.text, asr.text, rec.language))
                stream_path = self.ctx.asr_path(
                    mcfg["id"], rec.recording_id).with_suffix(".streaming.json")
                if stream_path.exists():
                    s = load_json(stream_path)
                    row.update(
                        time_to_first_text_sec=s.get("time_to_first_text_sec"),
                        sentence_finalization_delay_sec=s.get("sentence_finalization_delay_sec"),
                        streaming_revisions=s.get("revisions"),
                        chunked_rtf=s.get("chunked_rtf"))
                asr_rows.append(row)

            for mcfg in self.cfg.get("diarization_models", []):
                row = dict(base, diar_model=mcfg["id"], family=mcfg.get("family"))
                try:
                    diar = DiarizationResult.from_dict(load_json(
                        self.ctx.diar_path(mcfg["id"], rec.recording_id)))
                    diar_cache[mcfg["id"]] = diar
                except FileNotFoundError:
                    diar_rows.append(dict(row, status="missing"))
                    continue
                row.update(status=diar.status, error=diar.error,
                           runtime_sec=diar.runtime_sec,
                           real_time_factor=diar.real_time_factor,
                           model_load_time_sec=diar.load_time_sec,
                           peak_ram_mb=diar.resources.peak_ram_mb if diar.resources else None,
                           peak_vram_mb=diar.resources.peak_vram_mb if diar.resources else None)
                if diar.status == "completed" and ref is not None:
                    try:
                        row.update(diarization_metrics(
                            ref.turns, diar.turns, collar=collar,
                            skip_overlap=skip_overlap))
                    except Exception as e:
                        self.ctx.record_error("metrics", mcfg["id"],
                                              rec.recording_id, e)
                        row.update(status="failed", error=f"metrics: {e}")
                diar_rows.append(row)

            for asr_cfg in self.cfg.get("asr_models", []):
                for diar_cfg in self.cfg.get("diarization_models", []):
                    row = dict(base, asr_model=asr_cfg["id"],
                               diar_model=diar_cfg["id"])
                    path = self.ctx.combined_path(
                        asr_cfg["id"], diar_cfg["id"], rec.recording_id)
                    if not path.exists():
                        pair_rows.append(dict(row, status="missing"))
                        continue
                    combined = CombinedResult.from_dict(load_json(path))
                    row["status"] = combined.status
                    row["error"] = combined.error
                    asr = asr_cache.get(asr_cfg["id"])
                    diar = diar_cache.get(diar_cfg["id"])
                    if asr and diar and asr.runtime_sec and diar.runtime_sec:
                        total = asr.runtime_sec + diar.runtime_sec
                        row["runtime_sec"] = total
                        if rec.duration_sec:
                            row["real_time_factor"] = total / rec.duration_sec
                        row["peak_ram_mb"] = max(filter(None, [
                            asr.resources.peak_ram_mb if asr.resources else None,
                            diar.resources.peak_ram_mb if diar.resources else None]),
                            default=None)
                        row["peak_vram_mb"] = max(filter(None, [
                            asr.resources.peak_vram_mb if asr.resources else None,
                            diar.resources.peak_vram_mb if diar.resources else None]),
                            default=None)
                    if combined.status == "completed" and ref is not None and diar:
                        try:
                            row.update(combined_metrics(
                                ref, combined, diar.turns, rec.language))
                            if asr is not None:
                                row["wer"] = asr_metrics(
                                    ref.text, asr.text, rec.language)["wer"]
                        except Exception as e:
                            self.ctx.record_error(
                                "metrics", f"{asr_cfg['id']}+{diar_cfg['id']}",
                                rec.recording_id, e)
                            row.update(status="failed", error=f"metrics: {e}")
                    pair_rows.append(row)

        out_dir = self.ctx.run_dir / "metrics" / "per_recording"
        atomic_write_json(out_dir / "asr_rows.json", asr_rows)
        atomic_write_json(out_dir / "diar_rows.json", diar_rows)
        atomic_write_json(out_dir / "pair_rows.json", pair_rows)
        _rows_to_csv(out_dir / "asr_rows.csv", asr_rows)
        _rows_to_csv(out_dir / "diar_rows.csv", diar_rows)
        _rows_to_csv(out_dir / "pair_rows.csv", pair_rows)
        log.info("metrics written: %d asr rows, %d diar rows, %d pair rows",
                 len(asr_rows), len(diar_rows), len(pair_rows))

    # ------------------------------------------------------------------- run
    def run(self) -> None:
        self._init_manifest()
        status = "completed"
        try:
            self.run_asr_stage()
            self.run_diarization_stage()
            self.run_fusion_stage()
            self.run_metrics_stage()
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        except Exception as e:
            status = "failed"
            self.ctx.record_error("runner", None, None, e)
            raise
        finally:
            self.ctx.update_manifest(completed_at=utc_now_iso(), status=status)
            self.ctx.register_in_index(
                run_id=self.ctx.run_id, started_at=self.ctx.read_manifest().get("started_at"),
                completed_at=utc_now_iso(), track=self.track,
                profile=self.cfg.get("profile"), status=status,
                num_recordings=len(self.recordings),
                num_asr_models=len(self.cfg.get("asr_models", [])),
                num_diar_models=len(self.cfg.get("diarization_models", [])),
            )


def _lang_match(detected: str, expected: str) -> bool:
    d = detected.lower().split("-")[0].split("_")[0]
    e = expected.lower().split("-")[0].split("_")[0]
    aliases = {"cmn": "zh", "yue": "zh", "arb": "ar"}
    return aliases.get(d, d) == aliases.get(e, e)


def _rows_to_csv(path, rows: list[dict]) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
