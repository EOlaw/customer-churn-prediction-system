"""Banking CSV validation and data-quality reporting.

``inspect_csv`` never raises on bad data: it returns a complete quality report listing every
problem found (with data-row numbers, never record values beyond category labels), and a
typed frame only when the report has no errors. ``validate_csv`` is the strict gate used by
import and training. Rows are never dropped, repaired, or imputed here.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import BANKING, ChurnDefinition
from churn.ml.contract import (
    CONTRACT_VERSION,
    IGNORED_COLUMNS,
    LEAKAGE_SUSPECT_COLUMNS,
    MAX_FILE_BYTES,
    MAX_ID_LENGTH,
    MIN_CLASS_EXAMPLES,
    MIN_ROWS,
    SOURCE_KINDS,
    SPEC_BY_NAME,
    ColumnSpec,
)

EXAMPLE_ROWS = 5
SOURCE_LABELS = {
    "synthetic": "SYNTHETIC: generated test data, not real bank data",
    "external": "EXTERNAL: user-supplied file; provenance and permitted use are the importer's responsibility",
}


@dataclass(frozen=True)
class ValidatedDataset:
    frame: pd.DataFrame
    sha256: str
    rows: int
    source_kind: str
    contract_version: str
    report: dict
    report_sha256: str


@dataclass(frozen=True)
class Inspection:
    report: dict
    frame: pd.DataFrame | None


class DataQualityError(ValueError):
    def __init__(self, report: dict):
        self.report = report
        errors = [issue for issue in report["issues"] if issue["severity"] == "error"]
        detail = "; ".join(issue["message"] for issue in errors[:5])
        more = f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""
        super().__init__(f"Dataset failed {report['contract_version']}: {detail}{more}")


def report_fingerprint(report: dict) -> str:
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class _Issues:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, code: str, severity: str, message: str, column: str | None = None,
            count: int = 0, rows: list[int] | None = None, **extra: object) -> None:
        item = {"code": code, "severity": severity, "column": column, "count": int(count),
                "message": message, "example_rows": (rows or [])[:EXAMPLE_ROWS]}
        item.update(extra)
        self.items.append(item)

    @property
    def has_errors(self) -> bool:
        return any(item["severity"] == "error" for item in self.items)


def _rows(mask: pd.Series) -> list[int]:
    """1-based data-row numbers (header excluded) for the True positions of ``mask``."""
    return [int(i) + 1 for i in np.flatnonzero(mask.to_numpy())]


def _numeric_issues(series: pd.Series, spec: ColumnSpec, issues: _Issues) -> pd.Series:
    present = series.notna()
    numeric = pd.to_numeric(series, errors="coerce")
    unparsable = present & numeric.isna()
    if unparsable.any():
        issues.add("invalid_type", "error",
                   f"{spec.name}: {int(unparsable.sum())} non-numeric value(s)",
                   spec.name, int(unparsable.sum()), _rows(unparsable))
    nonfinite = numeric.notna() & ~np.isfinite(numeric.astype("float64"))
    if nonfinite.any():
        issues.add("non_finite", "error", f"{spec.name}: {int(nonfinite.sum())} non-finite value(s)",
                   spec.name, int(nonfinite.sum()), _rows(nonfinite))
    finite = numeric.where(~nonfinite)
    if spec.kind == "integer":
        fractional = finite.notna() & (finite != finite.round())
        if fractional.any():
            issues.add("not_integer", "error",
                       f"{spec.name}: {int(fractional.sum())} non-integer value(s)",
                       spec.name, int(fractional.sum()), _rows(fractional))
    out = finite.notna() & ((finite < spec.minimum) | (finite > spec.maximum))
    if out.any():
        issues.add("out_of_range", "error",
                   f"{spec.name}: {int(out.sum())} value(s) outside [{spec.minimum}, {spec.maximum}]",
                   spec.name, int(out.sum()), _rows(out))
    return finite


def _blank_report(source_kind: str, definition: ChurnDefinition) -> dict:
    return {"contract_version": CONTRACT_VERSION, "definition_key": definition.key,
            "source_kind": source_kind, "source_label": SOURCE_LABELS[source_kind],
            "file": {"sha256": None, "size_bytes": None}, "rows": 0, "columns_present": [],
            "status": "failed", "issues": [], "columns": {}, "target": None,
            "duplicates": {"customer_id_rows": 0, "feature_target_rows": 0}}


def inspect_csv(path: Path, source_kind: str, definition: ChurnDefinition = BANKING) -> Inspection:
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"source_kind must be one of {SOURCE_KINDS}")
    report = _blank_report(source_kind, definition)
    issues = _Issues()

    def finish(frame: pd.DataFrame | None = None) -> Inspection:
        report["issues"] = issues.items
        if issues.has_errors:
            report["status"] = "failed"
            frame = None
        else:
            report["status"] = "passed_with_warnings" if issues.items and any(
                i["severity"] == "warning" for i in issues.items) else "passed"
        return Inspection(report, frame)

    size = path.stat().st_size
    report["file"]["size_bytes"] = size
    if size > MAX_FILE_BYTES:
        issues.add("file_too_large", "error", f"File is {size} bytes; limit is {MAX_FILE_BYTES}")
        return finish()
    raw = path.read_bytes()  # hash and parse the same bytes
    report["file"]["sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        issues.add("not_utf8", "error", "File is not valid UTF-8")
        return finish()
    if not text.strip():
        issues.add("empty_file", "error", "File is empty")
        return finish()

    records = [row for row in csv.reader(io.StringIO(text)) if row]
    header = records[0]
    repeated = sorted({name for name in header if header.count(name) > 1})
    if repeated:
        issues.add("duplicate_columns", "error", f"Repeated column names: {repeated}")
    # pandas would silently shift columns into an implicit index when a row has extra fields,
    # so field counts are checked explicitly before parsing.
    ragged = pd.Series([len(row) != len(header) for row in records[1:]], dtype=bool)
    if ragged.any():
        issues.add("malformed_csv", "error",
                   f"{int(ragged.sum())} row(s) do not have {len(header)} fields",
                   count=int(ragged.sum()), rows=_rows(ragged))
        return finish()
    try:
        raw_frame = pd.read_csv(io.StringIO(text), dtype=str, on_bad_lines="error", index_col=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        issues.add("malformed_csv", "error", f"CSV could not be parsed: {exc}")
        return finish()
    report["columns_present"] = list(raw_frame.columns)
    report["rows"] = len(raw_frame)
    if raw_frame.empty:
        issues.add("no_rows", "error", "File has a header but no records")

    required = [definition.customer_id_column, *definition.feature_columns, definition.target_column]
    missing_columns = [name for name in required if name not in raw_frame.columns]
    if missing_columns:
        issues.add("missing_columns", "error", f"Missing required columns: {missing_columns}",
                   count=len(missing_columns))
    for name in raw_frame.columns:
        if name in required:
            continue
        if name in LEAKAGE_SUSPECT_COLUMNS:
            issues.add("leakage_suspect_column", "error",
                       f"Column {name!r} is outcome-derived or unavailable before exit in related "
                       "datasets; remove it from the file after confirming it is not a valid feature",
                       name, 1)
        elif name in IGNORED_COLUMNS:
            issues.add("ignored_column", "info",
                       f"Column {name!r} is ignored and never stored or used as a feature", name, 1)
        else:
            issues.add("unexpected_column", "warning",
                       f"Column {name!r} is not in the contract; it is excluded from storage and features",
                       name, 1)
    if issues.has_errors and (raw_frame.empty or missing_columns):
        return finish()

    # Whitespace-only cells count as missing; surrounding whitespace is stripped.
    cells = raw_frame[[c for c in required if c in raw_frame.columns]].apply(
        lambda col: col.str.strip().replace("", pd.NA))
    typed: dict[str, pd.Series] = {}
    for name in cells.columns:
        spec = SPEC_BY_NAME[name]
        series = cells[name]
        missing = series.isna()
        profile: dict = {"kind": spec.kind, "missing": int(missing.sum())}
        if missing.any():
            issues.add("missing_values", "error",
                       f"{name}: {int(missing.sum())} missing value(s); no imputation is applied at import",
                       name, int(missing.sum()), _rows(missing))
        if spec.kind == "id":
            long_ids = series.str.len() > MAX_ID_LENGTH
            if long_ids.any():
                issues.add("id_too_long", "error", f"{name}: identifiers longer than {MAX_ID_LENGTH}",
                           name, int(long_ids.sum()), _rows(long_ids))
            padded = (raw_frame[name] != raw_frame[name].str.strip()) & ~missing
            if padded.any():
                issues.add("id_whitespace_normalized", "warning",
                           f"{name}: surrounding whitespace stripped from {int(padded.sum())} identifier(s)",
                           name, int(padded.sum()), _rows(padded))
            repeated_id = series.notna() & series.duplicated(keep=False)
            report["duplicates"]["customer_id_rows"] = int(repeated_id.sum())
            if repeated_id.any():
                issues.add("duplicate_customer_id", "error",
                           f"{name}: {int(repeated_id.sum())} rows share "
                           f"{int(series[repeated_id].nunique())} duplicated identifier(s)",
                           name, int(repeated_id.sum()), _rows(repeated_id))
            profile["distinct"] = int(series.nunique())
            typed[name] = series
        elif spec.kind == "category":
            bad = series.notna() & ~series.isin(spec.categories)
            if bad.any():
                labels = sorted({str(v)[:40] for v in series[bad].unique()})[:10]
                issues.add("unknown_category", "error",
                           f"{name}: {int(bad.sum())} value(s) outside {list(spec.categories)}; "
                           f"seen {labels}", name, int(bad.sum()), _rows(bad))
            profile["categories"] = {k: int(v) for k, v in
                                     series.value_counts().sort_index().items()}
            typed[name] = series
        else:
            numeric = _numeric_issues(series, spec, issues)
            valid = numeric.dropna()
            profile.update(minimum=float(valid.min()) if len(valid) else None,
                           maximum=float(valid.max()) if len(valid) else None,
                           contract_minimum=spec.minimum, contract_maximum=spec.maximum)
            typed[name] = numeric
        report["columns"][name] = profile

    target_name = definition.target_column
    target = typed[target_name]
    counts = target.value_counts().sort_index()
    report["target"] = {"counts": {str(int(k)): int(v) for k, v in counts.items()
                                   if pd.notna(k)}}
    valid_target = target.dropna()
    if len(valid_target):
        report["target"]["prevalence"] = float(valid_target.mean())
    positives = int((valid_target == 1).sum())
    negatives = int((valid_target == 0).sum())
    if len(raw_frame) < MIN_ROWS:
        issues.add("insufficient_rows", "error",
                   f"{len(raw_frame)} rows; at least {MIN_ROWS} are required", count=len(raw_frame))
    if min(positives, negatives) < MIN_CLASS_EXAMPLES:
        issues.add("insufficient_class_examples", "error",
                   f"Each target class needs at least {MIN_CLASS_EXAMPLES} examples "
                   f"(found {negatives} of 0 and {positives} of 1)", target_name, min(positives, negatives))

    features = list(definition.feature_columns)
    repeated_content = pd.concat([typed[c] for c in [*features, target_name]], axis=1).duplicated()
    report["duplicates"]["feature_target_rows"] = int(repeated_content.sum())
    if repeated_content.any():
        issues.add("duplicate_feature_rows", "warning",
                   f"{int(repeated_content.sum())} row(s) repeat an earlier row's features and target "
                   "under a different identifier; a random split can place copies on both sides",
                   count=int(repeated_content.sum()), rows=_rows(repeated_content))

    if issues.has_errors:
        return finish()
    frame = pd.DataFrame({definition.customer_id_column: typed[definition.customer_id_column]})
    for name in [*features, target_name]:
        spec = SPEC_BY_NAME[name]
        frame[name] = typed[name].astype("int64" if spec.kind == "integer" else
                                         "float64" if spec.kind == "float" else "object")
    return finish(frame)


def validate_csv(path: Path, source_kind: str, definition: ChurnDefinition = BANKING) -> ValidatedDataset:
    inspection = inspect_csv(path, source_kind, definition)
    if inspection.frame is None:
        raise DataQualityError(inspection.report)
    report = inspection.report
    return ValidatedDataset(frame=inspection.frame, sha256=report["file"]["sha256"],
                            rows=report["rows"], source_kind=source_kind,
                            contract_version=CONTRACT_VERSION, report=report,
                            report_sha256=report_fingerprint(report))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate a banking CSV and print the quality report")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--source-kind", choices=SOURCE_KINDS, required=True)
    parser.add_argument("--output", type=Path, help="write the report JSON here")
    args = parser.parse_args()
    report = inspect_csv(args.csv, args.source_kind).report
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if report["status"] != "failed" else 1)


if __name__ == "__main__":
    main()
