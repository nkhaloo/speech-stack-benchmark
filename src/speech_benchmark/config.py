"""YAML config loading for tracks (cpu/gpu), models, and datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(p: str | Path, root: Path | None = None) -> Path:
    """Resolve a possibly repo-relative path against the project root."""
    p = Path(p)
    if p.is_absolute():
        return p
    return (root or project_root()) / p


def load_track_config(path: str | Path) -> dict:
    """Load a track config (configs/cpu.yaml or configs/gpu.yaml) and inline
    the referenced model config files."""
    path = Path(path)
    cfg = load_yaml(path)
    root = project_root()
    for key in ("asr_models", "diarization_models", "streaming_stacks"):
        resolved: list[dict] = []
        for entry in cfg.get(key, []):
            if isinstance(entry, str):
                mc = load_yaml(resolve_path(entry, root))
            elif isinstance(entry, dict) and "card" in entry:
                mc = load_yaml(resolve_path(entry["card"], root))
                mc = _deep_merge(mc, entry.get("overrides", {}))
            else:
                mc = dict(entry)
            if not mc.get("enabled", True):
                continue
            resolved.append(mc)
        cfg[key] = resolved
    return cfg


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge config overrides without mutating either input."""
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
