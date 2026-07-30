#!/usr/bin/env python3
"""Re-point a dataset manifest copied from another machine at its local files.

`prepare_datasets.py` writes **absolute** `audio_path`/`reference_path` values
into manifest.jsonl, so a dataset rsync'd from the lab machine arrives with a
manifest pointing at /home/... paths that do not exist here. The audio and
references themselves are fully portable; only the manifest needs fixing.

Layout is `<dataset_dir>/<language>/<recording>.wav`, so each entry is
re-pointed by its last two path components. Idempotent — running it on an
already-local manifest is a no-op. Every referenced file is checked to exist,
so a partial rsync is caught here rather than mid-run.

  python scripts/import_dataset.py artifacts/datasets/synthetic_commonvoice_mdc/smoke
  python scripts/import_dataset.py <dir> --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from speech_benchmark.schemas import atomic_write_text

FIELDS = ("audio_path", "reference_path")


def repoint(value: str, dataset_dir: Path) -> str:
    """<anything>/<lang>/<file>  ->  <dataset_dir>/<lang>/<file>"""
    parts = Path(value).parts
    if len(parts) < 2:
        return value
    return str((dataset_dir / parts[-2] / parts[-1]).resolve())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_dir", help="copied dataset dir (holds manifest.jsonl)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change; write nothing")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    manifest = dataset_dir / "manifest.jsonl"
    if not manifest.exists():
        sys.exit(f"No manifest.jsonl in {dataset_dir}")

    rows, changed, missing = [], 0, []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for f in FIELDS:
            old = rec.get(f)
            if not old:
                continue
            new = repoint(old, dataset_dir)
            if new != old:
                rec[f] = new
                changed += 1
            if not Path(rec[f]).exists():
                missing.append(rec[f])
        rows.append(rec)

    total_min = sum(r.get("duration_sec") or 0 for r in rows) / 60
    langs = sorted({r.get("language") for r in rows if r.get("language")})
    print(f"{len(rows)} recordings · {total_min:.1f} min audio · languages: {', '.join(langs)}")
    print(f"paths re-pointed: {changed}")

    if missing:
        print(f"\n{len(missing)} referenced file(s) MISSING — the copy is incomplete:",
              file=sys.stderr)
        for p in missing[:10]:
            print(f"  {p}", file=sys.stderr)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("\n(dry run — manifest not written)")
        return

    atomic_write_text(manifest,
                      "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"\nmanifest rewritten: {manifest}")
    print("All referenced files present. Ready to run.")


if __name__ == "__main__":
    main()
