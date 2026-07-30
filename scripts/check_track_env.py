#!/usr/bin/env python3
"""Preflight for any track: reports exactly which runtimes, tokens, gated-repo
acceptances, and weights are in place, so a failed run is never a surprise.
Read-only; downloads nothing.

The gated-repo list is derived from the track config itself (including the
`extra_repos` a pipeline needs but does not name in its own repo), so this stays
correct as cards change.

  python scripts/check_track_env.py --track cpu_streaming
  python scripts/check_track_env.py --track cpu
"""

import argparse
import os
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from download_models import collect_cards
from speech_benchmark.config import load_yaml, project_root, resolve_path

ROOT = project_root()
OK, BAD, WARN = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[33m!\033[0m"

RUNTIME_IMPORTS = {
    "faster_whisper": "faster_whisper",
    "sherpa_onnx": "sherpa_onnx",
    "sherpa_streaming": "sherpa_onnx",   # same runtime, streaming transducers
    "vosk": "vosk",
    "pyannote": "pyannote.audio",
    "whisper_cpp": None,          # external binary, checked separately
}


def mark(state) -> str:
    return OK if state is True else (WARN if state is None else BAD)


def check(label: str, fn):
    try:
        state, detail = fn()
    except Exception as e:  # noqa: BLE001
        state, detail = False, f"{type(e).__name__}: {e}"
    print(f"  {mark(state)} {label}: {detail}")
    return state


def _token() -> str | None:
    """Env var first, then the CLI's stored credential (`hf auth login`).

    huggingface_hub falls back to the stored token on its own when passed None,
    but the project's own code paths read the env vars explicitly, so both
    sources are reported separately below."""
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:  # noqa: BLE001
        return None


def _token_source() -> tuple[object, str]:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True, "HF_TOKEN env var"
    if _token():
        return True, "stored CLI login (~/.cache/huggingface/token)"
    return False, "none — run `hf auth login`, or export HF_TOKEN"


def _check_import(mod: str):
    def _fn():
        import importlib.util
        if importlib.util.find_spec(mod) is None:
            return False, f"{mod} not installed"
        return True, f"{mod} importable"
    return _fn


def _check_binary(name: str):
    def _fn():
        from shutil import which
        p = which(name)
        return (True, p) if p else (False, f"{name} not on PATH")
    return _fn


def _check_gated(repo: str):
    """Real download-permission test, not a metadata read.

    ``model_info`` succeeds on a gated repo for anyone — HF serves the metadata
    publicly and only blocks the files — so checking it reports "accessible"
    for a repo whose weights will 401 at download time. ``auth_check`` asks the
    question that actually matters: may *this* token fetch the files."""
    def _fn():
        from huggingface_hub import HfApi, auth_check
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
        tok = _token()
        gated = getattr(HfApi().model_info(repo, token=tok), "gated", False)
        try:
            auth_check(repo, token=tok)
        except GatedRepoError:
            return False, "GATED — accept its conditions at "\
                          f"https://huggingface.co/{repo}"
        except RepositoryNotFoundError:
            return False, "not found, or token lacks access"
        return True, "downloadable" + (f" (gated={gated}, terms accepted)" if gated else "")
    return _fn


def _weights_present(card: dict) -> tuple[object, str]:
    """Best-effort: has this card's download already landed?"""
    dl = card.get("download") or {}
    kind = dl.get("kind")
    if kind == "faster_whisper":
        d = ROOT / "artifacts" / "models" / "faster-whisper"
        hits = list(d.glob(f"models--*{dl['name']}*")) if d.exists() else []
        return (True, str(hits[0].name)) if hits else (False, "not downloaded")
    if kind == "hf_file":
        p = resolve_path(dl["dest"], ROOT)
        return (True, f"{p.stat().st_size/1e6:.0f} MB") if p.exists() else (False, "not downloaded")
    if kind == "sherpa_bundle":
        # Every sherpa bundle extracts into the SAME dest directory, so "dest
        # is non-empty" would report a missing ASR model as present the moment
        # the diarization bundle landed. Check this card's own targets.
        from download_models import sherpa_bundle_targets
        targets = sherpa_bundle_targets(dl)
        missing = [t.name for t in targets if not t.exists()]
        if not targets:
            return None, "no urls in card"
        return (False, f"not downloaded ({', '.join(missing)})") if missing \
            else (True, "present")
    if kind == "hf_snapshot":
        # snapshot_download() caches to the HF hub cache, not artifacts/models
        cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
        slug = "models--" + dl["repo"].replace("/", "--")
        return (True, "cached") if (cache / slug).exists() else (False, "not cached")
    return None, f"no check for kind={kind}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", required=True, help="track config basename in configs/")
    args = ap.parse_args()

    cfg_path = ROOT / "configs" / f"{args.track}.yaml"
    if not cfg_path.exists():
        sys.exit(f"No such track config: {cfg_path}")
    cards = collect_cards(load_yaml(cfg_path))
    weighted = [c for c in cards if c.get("download")]

    print(f"Track: {args.track}  ({len(weighted)} models with weights)\n")

    print("Runtimes:")
    runtimes = sorted({c.get("runtime") for c in cards if c.get("runtime")}
                      & set(RUNTIME_IMPORTS))
    rt_ok = []
    for rt in runtimes:
        mod = RUNTIME_IMPORTS[rt]
        rt_ok.append(check(rt, _check_import(mod) if mod
                           else _check_binary("whisper-cli")))

    print("\nHugging Face access:")
    tok = _token()
    check("credential", _token_source)
    gated_repos: list[str] = []
    for c in weighted:
        dl = c["download"]
        if c.get("gated") or dl.get("kind") == "hf_snapshot":
            gated_repos += [dl["repo"], *(dl.get("extra_repos") or [])]
    gate_ok = [check(r, _check_gated(r)) for r in sorted(set(gated_repos))] or [True]

    print("\nWeights on disk:")
    w_ok = []
    for c in weighted:
        state, detail = _weights_present(c)
        print(f"  {mark(state)} {c['id']}: {detail}")
        w_ok.append(state is not False)

    ready = all(rt_ok) and all(x is not False for x in gate_ok) and all(w_ok)
    print(f"\nSummary: {'READY' if ready else 'NOT READY'}")
    if not ready:
        if not tok:
            print("  → `hf auth login` (or export HF_TOKEN) — gated weights need it")
        if not all(w_ok):
            print(f"  → python scripts/download_models.py --track {args.track}")
    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
