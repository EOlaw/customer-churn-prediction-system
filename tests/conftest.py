from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churn.demo import make_demo_data


@pytest.fixture
def demo_frame(tmp_path) -> pd.DataFrame:
    """Valid synthetic rows as strings, so tests can corrupt individual cells."""
    path = make_demo_data(tmp_path / "base.csv", rows=100)
    return pd.read_csv(path, dtype=str)


@pytest.fixture
def write_csv(tmp_path):
    def write(frame: pd.DataFrame, name: str = "data.csv") -> Path:
        path = tmp_path / name
        frame.to_csv(path, index=False)
        return path
    return write


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """One synthetic end-to-end training run shared by read-only tests (fast bootstrap)."""
    from churn.ml.train import train

    root = tmp_path_factory.mktemp("trained")
    csv_path = make_demo_data(root / "demo.csv")
    report = train(csv_path, root / "artifacts", "synthetic", resamples=100)
    return csv_path, root / "artifacts", report
