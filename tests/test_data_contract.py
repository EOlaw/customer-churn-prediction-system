from __future__ import annotations

import json
import sys

import pytest

from churn.config import BANKING
from churn.ml import contract
from churn.ml import data as data_module
from churn.ml.contract import CONTRACT_VERSION, IGNORED_COLUMNS, SPEC_BY_NAME
from churn.ml.data import DataQualityError, inspect_csv, report_fingerprint, validate_csv


def codes(report, severity="error"):
    return {i["code"] for i in report["issues"] if i["severity"] == severity}


def issue(report, code):
    return next(i for i in report["issues"] if i["code"] == code)


def inspect(path):
    return inspect_csv(path, "synthetic").report


def test_contract_matches_churn_definition():
    definition_columns = {BANKING.customer_id_column, BANKING.target_column, *BANKING.feature_columns}
    assert set(SPEC_BY_NAME) == definition_columns
    for column in BANKING.categorical_columns:
        assert SPEC_BY_NAME[column].kind == "category"
    for column in BANKING.numeric_columns:
        assert SPEC_BY_NAME[column].kind in {"integer", "float"}


def test_valid_file_is_typed_and_labelled(write_csv, demo_frame):
    result = inspect_csv(write_csv(demo_frame), "synthetic")
    assert result.report["status"] == "passed"
    assert result.report["contract_version"] == CONTRACT_VERSION
    assert "SYNTHETIC" in result.report["source_label"]
    assert result.frame is not None and len(result.frame) == 100
    assert str(result.frame["Age"].dtype) == "int64"
    assert str(result.frame["Balance"].dtype) == "float64"
    assert result.report["columns"]["Age"]["missing"] == 0


def test_invalid_source_kind_is_rejected(write_csv, demo_frame):
    with pytest.raises(ValueError, match="source_kind"):
        inspect_csv(write_csv(demo_frame), "real")


def test_empty_and_header_only_files(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    assert "empty_file" in codes(inspect(empty))
    header = tmp_path / "header.csv"
    header.write_text(",".join(SPEC_BY_NAME) + "\n")
    assert "no_rows" in codes(inspect(header))


def test_non_utf8_and_malformed_files(tmp_path):
    bad = tmp_path / "latin1.csv"
    bad.write_bytes("CustomerId,Geography\n1,Fran\xe7e\n".encode("latin-1"))
    assert "not_utf8" in codes(inspect(bad))
    ragged = tmp_path / "ragged.csv"
    ragged.write_text("a,b\n1,2,3\n")
    report = inspect(ragged)
    assert "malformed_csv" in codes(report)
    assert issue(report, "malformed_csv")["example_rows"] == [1]
    short = tmp_path / "short.csv"
    short.write_text("a,b,c\n1,2\n")
    assert "malformed_csv" in codes(inspect(short))


def test_utf8_bom_is_accepted(tmp_path, demo_frame, write_csv):
    path = write_csv(demo_frame)
    bom = tmp_path / "bom.csv"
    bom.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert inspect(bom)["status"] == "passed"


def test_file_size_limit(write_csv, demo_frame, monkeypatch):
    monkeypatch.setattr(data_module, "MAX_FILE_BYTES", 100)
    report = inspect(write_csv(demo_frame))
    assert "file_too_large" in codes(report)
    assert report["file"]["sha256"] is None  # rejected before reading


def test_missing_duplicate_and_unexpected_columns(write_csv, demo_frame):
    report = inspect(write_csv(demo_frame.drop(columns=["Age", "Balance"])))
    assert issue(report, "missing_columns")["message"].endswith("['Age', 'Balance']")
    dup = write_csv(demo_frame, "dup.csv")
    dup.write_text(dup.read_text().replace("Age", "Gender", 1))
    assert "duplicate_columns" in codes(inspect(dup))
    extra = demo_frame.assign(RowNumber="1", Surname="X", Branch="north")
    report = inspect(write_csv(extra, "extra.csv"))
    assert report["status"] == "passed_with_warnings"
    assert {i["column"] for i in report["issues"] if i["code"] == "ignored_column"} == set(IGNORED_COLUMNS)
    assert issue(report, "unexpected_column")["column"] == "Branch"
    assert validate_csv(write_csv(extra, "extra2.csv"), "synthetic").frame.columns.tolist() == [
        "CustomerId", *BANKING.feature_columns, "Exited"]


@pytest.mark.parametrize("column", contract.LEAKAGE_SUSPECT_COLUMNS)
def test_leakage_suspect_columns_are_errors(write_csv, demo_frame, column):
    report = inspect(write_csv(demo_frame.assign(**{column: "1"})))
    assert report["status"] == "failed"
    assert issue(report, "leakage_suspect_column")["column"] == column


EDGES = [
    ("CreditScore", ["300", "850"], ["299", "851"]),
    ("Age", ["18", "100"], ["17", "101"]),
    ("Tenure", ["0", "10"], ["-1", "11"]),
    ("Balance", ["0", "0.0", "10000000"], ["-0.01", "10000000.01"]),
    ("NumOfProducts", ["1", "4"], ["0", "5"]),
    ("HasCrCard", ["0", "1"], ["2", "-1"]),
    ("IsActiveMember", ["0", "1"], ["2"]),
    ("EstimatedSalary", ["0", "199999.99"], ["-1"]),
]


@pytest.mark.parametrize(("column", "valid", "invalid"), EDGES)
def test_numeric_boundaries(write_csv, demo_frame, column, valid, invalid):
    for value in valid:
        frame = demo_frame.copy()
        frame.loc[5, column] = value
        assert inspect(write_csv(frame))["status"] == "passed", (column, value)
    for value in invalid:
        frame = demo_frame.copy()
        frame.loc[5, column] = value
        report = inspect(write_csv(frame))
        assert "out_of_range" in codes(report), (column, value)
        assert issue(report, "out_of_range")["example_rows"] == [6]


@pytest.mark.parametrize(("value", "code"), [("abc", "invalid_type"), ("inf", "non_finite"),
                                              ("-inf", "non_finite"), ("35.5", "not_integer")])
def test_numeric_type_errors(write_csv, demo_frame, value, code):
    frame = demo_frame.copy()
    frame.loc[0, "Age"] = value
    assert code in codes(inspect(write_csv(frame)))


def test_integer_valued_floats_are_accepted(write_csv, demo_frame):
    frame = demo_frame.copy()
    frame.loc[0, "Age"] = "41.0"
    assert inspect_csv(write_csv(frame), "synthetic").frame["Age"].iloc[0] == 41


def test_target_values(write_csv, demo_frame):
    for value in ("2", "0.5", "yes", "-1"):
        frame = demo_frame.copy()
        frame.loc[3, "Exited"] = value
        assert codes(inspect(write_csv(frame))) & {"out_of_range", "invalid_type", "not_integer"}


def test_missing_values_are_reported_not_dropped(write_csv, demo_frame):
    frame = demo_frame.copy()
    frame.loc[1, "Balance"] = ""
    frame.loc[2, "Geography"] = "   "
    frame.loc[4, "Exited"] = None
    report = inspect(write_csv(frame))
    assert report["rows"] == 100
    assert {i["column"] for i in report["issues"] if i["code"] == "missing_values"} == {
        "Balance", "Geography", "Exited"}
    assert issue(report, "missing_values")["example_rows"] != []
    assert report["columns"]["Balance"]["missing"] == 1
    with pytest.raises(DataQualityError, match="missing value"):
        validate_csv(write_csv(frame, "again.csv"), "synthetic")


def test_unknown_categories(write_csv, demo_frame):
    frame = demo_frame.copy()
    frame.loc[0, "Geography"] = "france"
    frame.loc[1, "Geography"] = "Italy"
    frame.loc[2, "Gender"] = "Other"
    report = inspect(write_csv(frame))
    bad = [i for i in report["issues"] if i["code"] == "unknown_category"]
    assert {i["column"] for i in bad} == {"Geography", "Gender"}
    assert next(i for i in bad if i["column"] == "Geography")["count"] == 2


def test_duplicate_ids_including_whitespace_variants(write_csv, demo_frame):
    frame = demo_frame.copy()
    frame.loc[10, "CustomerId"] = frame.loc[3, "CustomerId"]
    frame.loc[20, "CustomerId"] = " " + frame.loc[7, "CustomerId"]
    report = inspect(write_csv(frame))
    duplicate = issue(report, "duplicate_customer_id")
    assert duplicate["count"] == 4
    assert duplicate["example_rows"] == [4, 8, 11, 21]
    assert report["duplicates"]["customer_id_rows"] == 4
    with pytest.raises(DataQualityError, match="share"):
        validate_csv(write_csv(frame, "dup.csv"), "synthetic")


def test_id_length_limit(write_csv, demo_frame):
    frame = demo_frame.copy()
    frame.loc[0, "CustomerId"] = "x" * 121
    assert "id_too_long" in codes(inspect(write_csv(frame)))
    frame.loc[0, "CustomerId"] = "x" * 120
    assert inspect(write_csv(frame, "ok.csv"))["status"] == "passed"


def test_all_problems_are_reported_together(write_csv, demo_frame):
    frame = demo_frame.copy()
    frame.loc[0, "Age"] = "5"
    frame.loc[1, "Geography"] = "Italy"
    frame.loc[2, "Balance"] = ""
    frame.loc[4, "CustomerId"] = frame.loc[3, "CustomerId"]
    assert codes(inspect(write_csv(frame))) >= {
        "out_of_range", "unknown_category", "missing_values", "duplicate_customer_id"}


def test_sample_size_boundaries(write_csv, demo_frame):
    def labelled(rows: int, positives: int):
        frame = demo_frame.head(rows).copy()
        frame["Exited"] = ["1"] * positives + ["0"] * (rows - positives)
        return frame

    assert inspect(write_csv(labelled(50, 10), "a.csv"))["status"] == "passed"
    assert "insufficient_rows" in codes(inspect(write_csv(labelled(49, 10), "b.csv")))
    assert "insufficient_class_examples" in codes(inspect(write_csv(labelled(60, 9), "c.csv")))
    assert "insufficient_class_examples" in codes(inspect(write_csv(labelled(60, 51), "d.csv")))
    assert "insufficient_class_examples" in codes(inspect(write_csv(labelled(60, 0), "e.csv")))


def test_repeated_feature_rows_warn_but_pass(write_csv, demo_frame):
    frame = demo_frame.copy()
    for column in (*BANKING.feature_columns, "Exited"):
        frame.loc[9, column] = frame.loc[8, column]
    report = inspect(write_csv(frame))
    assert report["status"] == "passed_with_warnings"
    assert issue(report, "duplicate_feature_rows")["example_rows"] == [10]


def test_reports_are_reproducible_and_hash_the_parsed_bytes(write_csv, demo_frame):
    path = write_csv(demo_frame)
    first, second = validate_csv(path, "synthetic"), validate_csv(path, "synthetic")
    assert first.report == second.report
    assert first.report_sha256 == second.report_sha256 == report_fingerprint(first.report)
    assert first.sha256 == first.report["file"]["sha256"]
    changed = demo_frame.copy()
    changed.loc[0, "Age"] = "99"
    other = validate_csv(write_csv(changed, "changed.csv"), "synthetic")
    assert other.sha256 != first.sha256 and other.report_sha256 != first.report_sha256
    relabelled = validate_csv(path, "external")
    assert relabelled.sha256 == first.sha256 and relabelled.report_sha256 != first.report_sha256
    json.dumps(first.report, allow_nan=False)


def test_validate_cli_exit_codes(write_csv, demo_frame, monkeypatch, capsys):
    def run(path):
        monkeypatch.setattr(sys, "argv", ["validate", str(path), "--source-kind", "synthetic"])
        with pytest.raises(SystemExit) as exit_info:
            data_module.main()
        return exit_info.value.code

    assert run(write_csv(demo_frame)) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
    assert run(write_csv(demo_frame.drop(columns=["Age"]), "bad.csv")) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
