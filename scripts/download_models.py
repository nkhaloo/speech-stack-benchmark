#!/usr/bin/env python3
"""Download model weights for a track into artifacts/models/.

Reads the `download:` section of every enabled model config in the track.
Gated pyannote pipelines need a one-time terms acceptance on huggingface.co
and HF_TOKEN in the environment. After downloading, inference is offline
(HF_HUB_OFFLINE=1 works).

  python scripts/download_models.py --track gpu
  python scripts/download_models.py --track cpu
  python scripts/download_models.py --track gpu --models fw-large-v3
  python scripts/download_models.py --track gpu --include-disabled
"""

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401
from speech_benchmark.config import load_yaml, project_root, resolve_path

ROOT = project_root()


def download_faster_whisper(name: str) -> None:
    from faster_whisper.utils import download_model

    dest = ROOT / "artifacts" / "models" / "faster-whisper"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  faster-whisper: {name} -> {dest}")
    download_model(name, cache_dir=str(dest))


def download_hf_file(repo: str, file: str, dest: str) -> None:
    from huggingface_hub import hf_hub_download

    dest_path = resolve_path(dest, ROOT)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  hf file: {repo}/{file} -> {dest_path}")
    got = hf_hub_download(repo_id=repo, filename=file,
                          local_dir=str(dest_path.parent))
    got = Path(got)
    if got != dest_path and got.exists():
        got.replace(dest_path)


def download_hf_snapshot(repo: str, extra_repos: list[str] | None = None) -> None:
    import os

    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    for r in [repo] + (extra_repos or []):
        print(f"  hf snapshot: {r}")
        try:
            snapshot_download(repo_id=r, token=token)
        except Exception as e:
            print(f"  !! {r}: {e}", file=sys.stderr)
            if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
                print(f"     Gated repo — accept the terms at "
                      f"https://huggingface.co/{r} and set HF_TOKEN.",
                      file=sys.stderr)


def download_sherpa_bundle(segmentation_url: str, embedding_url: str, dest: str) -> None:
    dest_dir = resolve_path(dest, ROOT)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for url in (segmentation_url, embedding_url):
        name = url.rsplit("/", 1)[-1]
        target = dest_dir / name
        if target.exists() or (dest_dir / name.replace(".tar.bz2", "")).exists():
            print(f"  exists: {name}")
        else:
            print(f"  fetching {url}")
            urllib.request.urlretrieve(url, target)
        if name.endswith(".tar.bz2") and target.exists():
            print(f"  extracting {name}")
            with tarfile.open(target, "r:bz2") as tar:
                tar.extractall(dest_dir)
            target.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", required=True, choices=["cpu", "gpu", "smoke"])
    ap.add_argument("--models", nargs="*", default=None,
                    help="limit to these model ids")
    ap.add_argument("--include-disabled", action="store_true")
    args = ap.parse_args()

    track_cfg = load_yaml(ROOT / "configs" / f"{args.track}.yaml")
    entries = track_cfg.get("asr_models", []) + track_cfg.get("diarization_models", [])
    failures = 0
    for entry in entries:
        mc = load_yaml(resolve_path(entry, ROOT)) if isinstance(entry, str) else entry
        if not mc.get("enabled", True) and not args.include_disabled:
            print(f"skip (disabled): {mc.get('id')}")
            continue
        if args.models and mc.get("id") not in args.models:
            continue
        dl = mc.get("download")
        if not dl:
            print(f"skip (no download spec): {mc.get('id')}")
            continue
        print(f"== {mc['id']} ==")
        try:
            kind = dl.get("kind")
            if kind == "faster_whisper":
                download_faster_whisper(dl["name"])
            elif kind == "hf_file":
                download_hf_file(dl["repo"], dl["file"], dl["dest"])
            elif kind == "hf_snapshot":
                download_hf_snapshot(dl["repo"], dl.get("extra_repos"))
            elif kind == "sherpa_bundle":
                download_sherpa_bundle(dl["segmentation_url"],
                                       dl["embedding_url"], dl["dest"])
            else:
                print(f"  unknown download kind: {kind}", file=sys.stderr)
        except Exception as e:
            failures += 1
            print(f"  FAILED: {e}", file=sys.stderr)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
