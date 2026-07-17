import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speech_benchmark.datasets.synthetic import prepare_synthetic_dataset  # noqa: E402


@pytest.fixture(scope="session")
def dummy_dataset(tmp_path_factory):
    """Tiny two-language dummy dataset shared across tests."""
    out = tmp_path_factory.mktemp("data")
    recordings = prepare_synthetic_dataset(
        out_dir=out,
        source_name="dummy",
        profile="unittest",
        languages=["en", "zh"],
        minutes_per_language=1.0,
        recording_minutes=0.5,
        speakers_range=(2, 2),
        seed=1234,
    )
    return out, recordings
