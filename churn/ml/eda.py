"""Reproducible exploratory-analysis artifacts for a validated banking CSV.

Only files that pass the contract are profiled. Output is a pure function of the input bytes:
no timestamps, no randomness, rounded floats, sorted keys. Every table carries a
``source_kind`` column and the manifest/README repeat the SYNTHETIC or EXTERNAL label, so an
artifact cannot be read without knowing what it describes. All rates are in-sample
descriptions of one cross-sectional snapshot, not forecasts or causal effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import BANKING
from churn.ml.contract import SOURCE_KINDS
from churn.ml.data import SOURCE_LABELS, ValidatedDataset, validate_csv

SMALL_SEGMENT = 30
HISTOGRAM_BINS = 10
EXTREME_ROWS_PER_COLUMN = 20
CONTINUOUS = ("CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary")
SEGMENT_BANDS = {
    "Age": [18, 30, 40, 50, 60, 101],
    "CreditScore": [300, 500, 600, 700, 851],
}
SEGMENTED_AS_IS = ("Geography", "Gender", "HasCrCard", "IsActiveMember", "NumOfProducts", "Tenure")


def _wilson(exits: int, rows: int, z: float = 1.96) -> tuple[float, float]:
    p = exits / rows
    denom = 1 + z * z / rows
    centre = (p + z * z / (2 * rows)) / denom
    half = z * np.sqrt(p * (1 - p) / rows + z * z / (4 * rows * rows)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _segment_rows(frame: pd.DataFrame, feature: str, labels: pd.Series) -> list[dict]:
    target = frame[BANKING.target_column]
    rows = []
    for label, group in target.groupby(labels, observed=True, sort=True):
        n, exits = len(group), int(group.sum())
        low, high = _wilson(exits, n)
        rows.append({"feature": feature, "segment": str(label), "rows": n, "exits": exits,
                     "exit_rate": exits / n, "exit_rate_ci95_low": low, "exit_rate_ci95_high": high,
                     "share_of_rows": n / len(frame), "small_segment": n < SMALL_SEGMENT})
    return rows


def class_balance(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame[BANKING.target_column].value_counts().sort_index()
    return pd.DataFrame({"class": counts.index.astype(int), "rows": counts.values,
                         "share": counts.values / len(frame)})


def missing_values(dataset: ValidatedDataset) -> pd.DataFrame:
    # A validated dataset has no missing required values; the counts come from the raw-file
    # quality report so a reader can see the check was made, not assumed.
    columns = dataset.report["columns"]
    return pd.DataFrame({"column": list(columns), "missing": [c["missing"] for c in columns.values()],
                         "share": [c["missing"] / dataset.rows for c in columns.values()]})


def numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in BANKING.numeric_columns:
        s = frame[column]
        q = s.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        rows.append({"column": column, "rows": len(s), "mean": s.mean(), "std": s.std(ddof=1),
                     "min": s.min(), "p01": q[0.01], "p25": q[0.25], "median": q[0.5],
                     "p75": q[0.75], "p99": q[0.99], "max": s.max(), "skew": s.skew(),
                     "zeros": int((s == 0).sum()), "distinct": int(s.nunique())})
    return pd.DataFrame(rows)


def numeric_histograms(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in BANKING.numeric_columns:
        s = frame[column]
        if s.nunique() <= HISTOGRAM_BINS:  # discrete: exact value counts
            for value, count in s.value_counts().sort_index().items():
                rows.append({"column": column, "bin_low": value, "bin_high": value, "rows": int(count)})
            continue
        counts, edges = np.histogram(s, bins=HISTOGRAM_BINS)
        rows += [{"column": column, "bin_low": edges[i], "bin_high": edges[i + 1], "rows": int(counts[i])}
                 for i in range(HISTOGRAM_BINS)]
    return pd.DataFrame(rows)


def categorical_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in BANKING.categorical_columns:
        for value, count in frame[column].value_counts().sort_index().items():
            rows.append({"column": column, "value": value, "rows": int(count),
                         "share": count / len(frame)})
    return pd.DataFrame(rows)


def segment_exit_rates(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for feature in SEGMENTED_AS_IS:
        rows += _segment_rows(frame, feature, frame[feature])
    for feature, edges in SEGMENT_BANDS.items():
        bands = pd.cut(frame[feature], bins=edges, right=False)
        labels = bands.map(lambda b: f"[{int(b.left)}, {int(b.right)})")
        rows += _segment_rows(frame, f"{feature} band", labels)
    balance = np.where(frame["Balance"] == 0, "zero balance", "positive balance")
    rows += _segment_rows(frame, "Balance zero/positive", pd.Series(balance, index=frame.index))
    return pd.DataFrame(rows)


def target_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    target = frame[BANKING.target_column]
    rows = [{"feature": c, "pearson_with_exited": frame[c].corr(target),
             "spearman_with_exited": frame[c].corr(target, method="spearman")}
            for c in BANKING.numeric_columns]
    return pd.DataFrame(rows)


def feature_correlation_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    matrix = frame[list(BANKING.numeric_columns)].corr(method="spearman")
    return matrix.rename_axis("feature").reset_index()


def outliers(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, extremes = [], []
    for column in CONTINUOUS:
        s = frame[column]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        below, above = s < low, s > high
        summary.append({"column": column, "q1": q1, "q3": q3, "lower_fence": low, "upper_fence": high,
                        "below_fence": int(below.sum()), "above_fence": int(above.sum()),
                        "share_outside": float((below | above).mean()), "min": s.min(), "max": s.max()})
        flagged = s[below | above]
        distance = (flagged - flagged.clip(low, high)).abs().sort_values(ascending=False, kind="stable")
        for index in distance.index[:EXTREME_ROWS_PER_COLUMN]:
            # Row numbers (1-based, header excluded) instead of customer identifiers.
            extremes.append({"column": column, "data_row": int(index) + 1, "value": s[index],
                             "side": "below" if s[index] < low else "above"})
    extreme_frame = pd.DataFrame(extremes, columns=["column", "data_row", "value", "side"])
    return pd.DataFrame(summary), extreme_frame


def _write_csv(frame: pd.DataFrame, path: Path, source_kind: str) -> None:
    out = frame.copy()
    out.insert(0, "source_kind", source_kind)
    out.round(6).to_csv(path, index=False, lineterminator="\n")


def _readme(dataset: ValidatedDataset, files: list[str]) -> str:
    report = dataset.report
    balance = class_balance(dataset.frame)
    prevalence = balance.loc[balance["class"] == 1, "share"].iloc[0]
    warnings = [i for i in report["issues"] if i["severity"] == "warning"]
    lines = [
        f"# Banking EDA: {dataset.source_kind.upper()}", "",
        f"> **{SOURCE_LABELS[dataset.source_kind]}**", "",
        f"- Dataset SHA-256: `{dataset.sha256}`",
        f"- Contract: `{dataset.contract_version}`; quality status: `{report['status']}`",
        f"- Rows: {dataset.rows:,}; exit prevalence: {prevalence:.2%}", "",
        "## Reading these tables", "",
        "- One cross-sectional snapshot: no observation dates, feature cutoff, or prediction horizon.",
        ("- Exit rates and correlations describe this file only. They are not forecasts, causal "
         "effects, or evidence of retention impact."),
        (f"- Segments with fewer than {SMALL_SEGMENT} rows are flagged `small_segment`; "
         "read their intervals, not their rates."),
        "- Outlier rows are identified by data-row number (header excluded), never by customer ID.",
        "- Outliers are inspected, not removed: values inside the contract's ranges are kept.", ""]
    if warnings:
        lines += ["## Quality warnings", ""] + [f"- `{w['code']}`: {w['message']}" for w in warnings] + [""]
    lines += ["## Files", ""] + [f"- `{name}`" for name in files] + [""]
    return "\n".join(lines)


def build_eda(csv_path: Path, source_kind: str, output_root: Path) -> Path:
    dataset = validate_csv(csv_path, source_kind)
    frame = dataset.frame
    out = output_root / f"eda-{source_kind}-{dataset.sha256[:12]}"
    out.mkdir(parents=True, exist_ok=True)
    summary, extremes = outliers(frame)
    tables = {
        "class_balance.csv": class_balance(frame),
        "missing_values.csv": missing_values(dataset),
        "numeric_summary.csv": numeric_summary(frame),
        "numeric_histograms.csv": numeric_histograms(frame),
        "categorical_distribution.csv": categorical_distribution(frame),
        "segment_exit_rates.csv": segment_exit_rates(frame),
        "target_correlations.csv": target_correlations(frame),
        "feature_correlations_spearman.csv": feature_correlation_matrix(frame),
        "outlier_summary_iqr.csv": summary,
        "outlier_rows.csv": extremes,
    }
    for name, table in tables.items():
        _write_csv(table, out / name, source_kind)
    (out / "quality_report.json").write_text(json.dumps(dataset.report, indent=2, sort_keys=True) + "\n")
    files = sorted([*tables, "quality_report.json", "README.md"])
    (out / "README.md").write_text(_readme(dataset, files))
    manifest = {
        "source_kind": source_kind, "source_label": SOURCE_LABELS[source_kind],
        "dataset_sha256": dataset.sha256, "rows": dataset.rows,
        "contract_version": dataset.contract_version,
        "quality_report_sha256": dataset.report_sha256,
        "files": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                  for name in files},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Write reproducible EDA artifacts for a banking CSV")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--source-kind", choices=SOURCE_KINDS, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    print(build_eda(args.csv, args.source_kind, args.output))


if __name__ == "__main__":
    main()
