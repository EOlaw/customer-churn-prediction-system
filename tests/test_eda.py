from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from churn.demo import make_demo_data
from churn.ml.data import DataQualityError
from churn.ml.eda import SMALL_SEGMENT, build_eda


@pytest.fixture
def demo_csv(tmp_path):
    return make_demo_data(tmp_path / "demo.csv", rows=300)


def test_eda_is_byte_reproducible(demo_csv, tmp_path):
    first = build_eda(demo_csv, "synthetic", tmp_path / "a")
    second = build_eda(demo_csv, "synthetic", tmp_path / "b")
    names = sorted(p.name for p in first.iterdir())
    assert names == sorted(p.name for p in second.iterdir())
    assert "manifest.json" in names
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    manifest = json.loads((first / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == digest


def test_every_artifact_is_labelled_with_its_source(demo_csv, tmp_path):
    synthetic = build_eda(demo_csv, "synthetic", tmp_path / "s")
    external = build_eda(demo_csv, "external", tmp_path / "e")
    assert synthetic.name.startswith("eda-synthetic-") and external.name.startswith("eda-external-")
    for out, kind in ((synthetic, "synthetic"), (external, "external")):
        for csv_file in out.glob("*.csv"):
            assert set(pd.read_csv(csv_file)["source_kind"]) == {kind}, csv_file.name
        assert kind.upper() in (out / "README.md").read_text()
        assert json.loads((out / "manifest.json").read_text())["source_kind"] == kind
        assert json.loads((out / "quality_report.json").read_text())["source_kind"] == kind
    assert "not real bank data" in (synthetic / "README.md").read_text()
    assert "not real bank data" not in (external / "README.md").read_text()


def test_tables_match_independent_calculations(demo_csv, tmp_path):
    out = build_eda(demo_csv, "synthetic", tmp_path)
    raw = pd.read_csv(demo_csv)
    balance = pd.read_csv(out / "class_balance.csv")
    assert balance["rows"].sum() == 300
    assert balance.loc[balance["class"] == 1, "rows"].iloc[0] == raw["Exited"].sum()
    segments = pd.read_csv(out / "segment_exit_rates.csv")
    germany = segments[(segments.feature == "Geography") & (segments.segment == "Germany")].iloc[0]
    expected = raw[raw.Geography == "Germany"]["Exited"]
    assert germany["rows"] == len(expected) and germany["exits"] == expected.sum()
    assert germany["exit_rate_ci95_low"] <= germany["exit_rate"] <= germany["exit_rate_ci95_high"]
    for feature, group in segments.groupby("feature"):
        assert group["rows"].sum() == 300, feature  # every row lands in exactly one segment
    assert (segments["small_segment"] == (segments["rows"] < SMALL_SEGMENT)).all()
    correlations = pd.read_csv(out / "target_correlations.csv")
    assert correlations["pearson_with_exited"].abs().le(1).all()
    assert set(pd.read_csv(out / "missing_values.csv")["missing"]) == {0}
    categorical = pd.read_csv(out / "categorical_distribution.csv")
    assert categorical.groupby("column")["rows"].sum().eq(300).all()


def test_outliers_are_located_by_row_number_not_identifier(tmp_path):
    frame = pd.read_csv(make_demo_data(tmp_path / "base.csv", rows=200))
    frame.loc[41, "Balance"] = 9_000_000.0
    path = tmp_path / "outlier.csv"
    frame.to_csv(path, index=False)
    out = build_eda(path, "synthetic", tmp_path / "out")
    rows = pd.read_csv(out / "outlier_rows.csv")
    hit = rows[(rows.column == "Balance") & (rows.data_row == 42)]
    assert len(hit) == 1 and hit.iloc[0]["side"] == "above" and hit.iloc[0]["value"] == 9_000_000.0
    assert "CustomerId" not in rows.columns
    summary = pd.read_csv(out / "outlier_summary_iqr.csv").set_index("column")
    assert summary.loc["Balance", "above_fence"] >= 1


def test_eda_refuses_invalid_files_without_writing(tmp_path):
    frame = pd.read_csv(make_demo_data(tmp_path / "base.csv", rows=100))
    frame.loc[0, "Age"] = 3
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(DataQualityError, match="outside"):
        build_eda(path, "synthetic", tmp_path / "out")
    assert not (tmp_path / "out").exists()
