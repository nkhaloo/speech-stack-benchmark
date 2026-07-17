#!/usr/bin/env python3
"""Check MDC Terms & Conditions access for each configured dataset.

Reports, per language, whether the dataset is:
  * ALREADY DOWNLOADED  — extracted locally with a .complete marker
  * ACCESS OK           — terms accepted; ready to download
  * TERMS NOT ACCEPTED  — must accept T&C on the MDC website first
  * ERROR               — some other failure (printed for inspection)

Runs the same download-plan API call the real downloader makes, but stops
before transferring any data, so it is instant and downloads nothing.

    python scripts/check_mdc_access.py
"""

import inspect
import os
from pathlib import Path

import _bootstrap  # noqa: F401
from speech_benchmark.config import load_yaml, project_root
from speech_benchmark.datasets.sources import CV_LANG, _find_cv_tsv

ROOT = project_root()


def _probe_access(dataset_id: str) -> tuple[str, str]:
    """Return (status, detail) for a dataset id without downloading it."""
    try:
        from datacollective.download import _get_download_plan
    except Exception as e:  # pragma: no cover - depends on SDK internals
        return "ERROR", f"cannot import download-plan helper: {e!r}"

    # Fill the plan function's parameters by name from what we know, so this
    # keeps working if the SDK signature shifts slightly.
    api_key = os.environ.get("MDC_API_KEY")
    known = {
        "dataset_id": dataset_id, "dataset": dataset_id, "id": dataset_id,
        "api_key": api_key, "token": api_key, "key": api_key,
    }
    sig = inspect.signature(_get_download_plan)
    kwargs = {}
    missing = []
    for name, param in sig.parameters.items():
        if name in known and known[name] is not None:
            kwargs[name] = known[name]
        elif param.default is inspect.Parameter.empty:
            missing.append(name)
    if missing:
        return "ERROR", (f"cannot auto-fill required plan args {missing}; "
                         f"signature is {sig}")

    try:
        _get_download_plan(**kwargs)
        return "ACCESS OK", "terms accepted; ready to download"
    except PermissionError as e:
        return "TERMS NOT ACCEPTED", str(e).splitlines()[0]
    except Exception as e:  # pragma: no cover - network/other
        return "ERROR", f"{type(e).__name__}: {e}"


def main() -> None:
    cfg = load_yaml(ROOT / "configs/datasets/synthetic.yaml")
    sk = cfg.get("source_kwargs", {})
    dataset_ids = sk.get("dataset_ids", {})
    download_dir = ROOT / sk.get("download_dir", "artifacts/datasets/mdc")
    split = sk.get("split", "validated")

    if not os.environ.get("MDC_API_KEY"):
        print("!! MDC_API_KEY is not set in this shell — the probe will fail.\n")

    width = max(len(l) for l in dataset_ids) if dataset_ids else 4
    for lang, dataset_id in dataset_ids.items():
        extract_dir = download_dir / dataset_id
        if (extract_dir / ".complete").exists() and \
                _find_cv_tsv(extract_dir, split) is not None:
            status, detail = "ALREADY DOWNLOADED", str(extract_dir)
        else:
            status, detail = _probe_access(dataset_id)
        cv_lang = CV_LANG.get(lang, lang)
        print(f"{lang:<{width}} ({cv_lang:<5}) {dataset_id}  ->  {status}")
        print(f"{'':<{width}}         {detail}")
        if status == "TERMS NOT ACCEPTED":
            print(f"{'':<{width}}         accept: "
                  f"https://mozilladatacollective.com/datasets/{dataset_id}")


if __name__ == "__main__":
    main()
