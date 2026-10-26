from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn.config import BANKING
from churn.ml import metrics as m
from churn.ml import train as train_module
from churn.ml.data import validate_csv
from churn.ml.lineage import file_sha256, id_fingerprint
from churn.ml.model import Calibrator, ScoringModel
from churn.ml.splits import PARTITIONS, assign_partitions, verify_partitions
from churn.ml.train import (
    DEMOGRAPHIC_FEATURES,
    FEATURE_SETS,
    UNCERTAIN_AVAILABILITY_FEATURES,
    train,
)

ID = BANKING.customer_id_column


def artifacts_of(trained):
    _, directory, report = trained
    version = report["version"]
    return (pd.read_csv(directory / f"{version}.splits.csv", dtype={ID: str}),
            pd.read_csv(directory / f"{version}.evaluation.csv", dtype={ID: str}),
            directory / f"{version}.joblib")


# ---- splits ---------------------------------------------------------------------------------

def test_partitions_are_disjoint_complete_stratified(trained):
    csv_path, _, report = trained
    frame = validate_csv(csv_path, "synthetic").frame
    splits, _, _ = artifacts_of(trained)
    assert splits[ID].is_unique and set(splits[ID]) == set(frame[ID])
    assert set(splits["partition"]) == set(PARTITIONS)
    sizes = splits["partition"].value_counts().to_dict()
    assert sizes == {"train": 360, "calibration": 120, "test": 120}
    joined = frame.merge(splits, on=ID)
    rates = joined.groupby("partition")["Exited"].mean()
    assert rates.max() - rates.min() < 0.02  # stratified
    for name in PARTITIONS:
        assert id_fingerprint(splits.loc[splits.partition == name, ID]) == \
            report["split"]["customer_id_sha256"][name]
    assert report["split"]["disjoint"] is True
    assert report["split"]["assignments_sha256"] == file_sha256(
        trained[1] / report["split"]["assignments_file"])


def test_assignments_are_reproducible_and_independent_of_row_order(trained):
    csv_path, _, _ = trained
    frame = validate_csv(csv_path, "synthetic").frame
    first = assign_partitions(frame[ID], frame["Exited"], 42)
    shuffled = frame.sample(frac=1.0, random_state=9)
    second = assign_partitions(shuffled[ID], shuffled["Exited"], 42)
    assert dict(zip(frame[ID], first, strict=True)) == dict(zip(shuffled[ID], second, strict=True))
    other_seed = assign_partitions(frame[ID], frame["Exited"], 43)
    assert (first != other_seed).any()


def test_partition_verification_rejects_bad_assignments():
    with pytest.raises(RuntimeError, match="non-empty"):
        verify_partitions(pd.Series(["train", "test", "train"]))
    with pytest.raises(RuntimeError, match="exactly one known"):
        verify_partitions(pd.Series(["train", "validation", None]))
    with pytest.raises(ValueError, match="unique"):
        assign_partitions(pd.Series(["a", "a"]), pd.Series([0, 1]), 1)


def test_training_is_reproducible(trained, tmp_path):
    csv_path, directory, report = trained
    again = train(csv_path, tmp_path / "again", "synthetic", resamples=100)
    assert again["version"] == report["version"]
    for key in ("split", "selection", "calibration", "threshold", "held_out_test",
                "sensitivity_analyses", "subgroups", "run_config_sha256"):
        assert again[key] == report[key], key
    name = report["version"]
    assert (tmp_path / "again" / f"{name}.splits.csv").read_bytes() == \
        (directory / f"{name}.splits.csv").read_bytes()
    assert (tmp_path / "again" / f"{name}.evaluation.csv").read_bytes() == \
        (directory / f"{name}.evaluation.csv").read_bytes()


def test_changed_configuration_changes_the_version(trained, tmp_path):
    csv_path, _, report = trained
    other = train(csv_path, tmp_path / "cap", "synthetic", review_capacity=0.2, resamples=100)
    assert other["version"] != report["version"]
    assert other["split"]["customer_id_sha256"] == report["split"]["customer_id_sha256"]
    with pytest.raises(ValueError, match="review_capacity"):
        train(csv_path, tmp_path / "bad", "synthetic", review_capacity=1.5)


# ---- isolation of test data -----------------------------------------------------------------

def test_fit_run_cannot_receive_test_data():
    import inspect

    assert list(inspect.signature(train_module.fit_run).parameters) == [
        "train", "calibration", "features", "estimator_name", "params", "seed"]


def test_only_the_right_rows_reach_each_stage(trained, tmp_path, monkeypatch):
    """Spy on every fit/selection call during a fresh run and check which rows it saw."""
    csv_path, _, report = trained
    frame = validate_csv(csv_path, "synthetic").frame
    partition = assign_partitions(frame[ID], frame["Exited"], 42)
    rows = {name: set(frame.index[partition == name]) for name in PARTITIONS}
    seen = {"selection": [], "pipeline_fit": [], "calibrator_fit": [], "threshold": [], "evaluate": []}

    real_select = train_module.select_candidate
    monkeypatch.setattr(train_module, "select_candidate",
                        lambda x, y, seed: (seen["selection"].append(set(x.index)),
                                            real_select(x, y, seed))[1])
    real_pipeline_fit = Pipeline.fit

    def spy_fit(self, x, y=None, **kw):
        seen["pipeline_fit"].append(set(x.index))
        return real_pipeline_fit(self, x, y, **kw)
    monkeypatch.setattr(Pipeline, "fit", spy_fit)
    real_calibrator_fit = Calibrator.fit
    monkeypatch.setattr(Calibrator, "fit", lambda self, raw, y: (
        seen["calibrator_fit"].append(len(raw)), real_calibrator_fit(self, raw, y))[1])
    real_threshold = m.best_f1_threshold
    monkeypatch.setattr(m, "best_f1_threshold", lambda y, p: (
        seen["threshold"].append(len(y)), real_threshold(y, p))[1])
    real_evaluate = train_module.evaluate_run
    monkeypatch.setattr(train_module, "evaluate_run", lambda run, test, *a: (
        seen["evaluate"].append(set(test.index)), real_evaluate(run, test, *a))[1])

    train(csv_path, tmp_path / "spy", "synthetic", resamples=20)

    assert seen["selection"] == [rows["train"]]
    # every fit, including each cross-validation fold, saw only training rows
    assert seen["pipeline_fit"] and all(s <= rows["train"] for s in seen["pipeline_fit"])
    assert sum(s == rows["train"] for s in seen["pipeline_fit"]) == len(FEATURE_SETS)  # final refits
    n_cal = len(rows["calibration"])
    assert seen["calibrator_fit"] and max(seen["calibrator_fit"]) == n_cal  # never more than calibration
    assert seen["threshold"] == [n_cal] * len(FEATURE_SETS)
    assert seen["evaluate"] == [rows["test"]] * len(FEATURE_SETS)  # test read once per variant, last
    assert report["split"]["rows"]["calibration"] == n_cal


def test_calibrator_and_threshold_are_functions_of_calibration_rows_only(trained):
    """Refit the calibrator from the stored calibration rows alone and reproduce the test scores."""
    _, _, report = trained
    _, evaluation, artifact = artifacts_of(trained)
    cal = evaluation[evaluation.partition == "calibration"]
    test = evaluation[evaluation.partition == "test"]
    method = report["calibration"]["selected_method"]
    refit = Calibrator(method).fit(cal.raw_probability.to_numpy(), cal.label.to_numpy())
    assert np.allclose(refit.predict(test.raw_probability.to_numpy()), test.probability, atol=1e-9)
    # the threshold lies on the calibrated scale and is not any test-set score tuned to the labels
    assert report["threshold"]["selected_on"] == "calibration"
    assert report["threshold"]["never_tuned_on_test"] is True
    stored = joblib.load(artifact)
    assert stored.calibrator.method == method


def test_threshold_is_not_optimal_for_the_test_set_by_construction(trained):
    """A threshold tuned on test would equal the test F1-optimum; the stored one is chosen earlier."""
    _, _, report = trained
    _, evaluation, _ = artifacts_of(trained)
    test = evaluation[evaluation.partition == "test"]
    y, p = test.label.to_numpy(), test.probability.to_numpy()
    tuned_on_test = m.best_f1_threshold(y, p)
    assert report["threshold"]["value"] != tuned_on_test
    fixed = m.threshold_metrics(y, p, report["threshold"]["value"])
    assert fixed["f1"] <= m.threshold_metrics(y, p, tuned_on_test)["f1"] + 1e-12


# ---- metrics reproduce from stored predictions --------------------------------------------

def test_reported_heldout_metrics_recompute_from_stored_predictions(trained):
    _, _, report = trained
    _, evaluation, _ = artifacts_of(trained)
    test = evaluation[evaluation.partition == "test"].reset_index(drop=True)
    y, p, raw = test.label.to_numpy(), test.probability.to_numpy(), test.raw_probability.to_numpy()
    held = report["held_out_test"]
    assert held["rows"] == len(test) == 120 and held["positives"] == int(y.sum())
    assert held["prevalence"] == pytest.approx(y.mean())
    assert held["roc_auc"] == pytest.approx(m.auc_score(y, p))
    assert held["pr_auc"] == pytest.approx(m.pr_auc_score(y, p))
    at = m.threshold_metrics(y, p, report["threshold"]["value"])
    for key in ("tp", "fp", "tn", "fn", "precision", "recall", "f1"):
        assert held["at_threshold"][key] == pytest.approx(at[key])
    cap = m.capacity_metrics(y, p, report["review_capacity"]["share"], raw)
    assert held["at_review_capacity"] == cap
    assert held["calibration"]["brier"] == pytest.approx(float(np.mean((p - y) ** 2)))
    assert sum(held["at_threshold"]["confusion_matrix"].values()) == 120
    assert held["confidence_intervals_95"]["roc_auc"][0] < held["roc_auc"] < \
        held["confidence_intervals_95"]["roc_auc"][1]


def test_report_documents_framing_and_separation(trained):
    _, _, report = trained
    assert "not a prospective" in report["framing"]
    assert report["evidence_status"].startswith("SYNTHETIC")
    assert report["selection"]["data"] == "train partition only"
    assert "calibration partition only" in report["calibration"]["data"]
    assert set(report["selection"]["candidates"]) == {"dummy", "logistic_regression", "gradient_boosting"}
    assert report["selection"]["candidates"]["gradient_boosting"]["configurations_searched"] == 8
    assert report["split"]["roles"]["test"].startswith("evaluated once")
    assert report["calibration"]["selected_method"] in {"none", "sigmoid", "isotonic"}
    assert set(report["calibration"]["cross_fitted_brier"]) == {"none", "sigmoid", "isotonic"}
    assert report["held_out_test"]["reference"]["pr_auc"] == pytest.approx(
        report["held_out_test"]["prevalence"])
    assert any("SYNTHETIC" in item for item in report["limitations"])
    json.dumps(report, allow_nan=False)


# ---- sensitivity analyses and subgroups ---------------------------------------------------

def test_sensitivity_variants_exclude_the_documented_features(trained):
    _, _, report = trained
    variants = report["sensitivity_analyses"]["variants"]
    assert set(variants) == {"full", "no_demographics", "no_uncertain_availability"}
    assert set(variants["no_demographics"]["excluded"]) == set(DEMOGRAPHIC_FEATURES)
    assert set(variants["no_uncertain_availability"]["excluded"]) == set(UNCERTAIN_AVAILABILITY_FEATURES)
    assert variants["full"]["excluded"] == []
    for name, variant in variants.items():
        assert not set(variant["features"]) & set(variant["excluded"])
        assert variant["roc_auc_ci95"][0] <= variant["roc_auc"] <= variant["roc_auc_ci95"][1], name
    assert "roc_auc_difference_vs_full" not in variants["full"]
    diff = variants["no_demographics"]["roc_auc_difference_vs_full"]
    assert diff["ci95"][0] <= diff["ci95"][1]
    assert "causal" in report["sensitivity_analyses"]["note"]


def test_uncertain_features_match_the_leakage_review():
    text = Path("docs/banking-leakage-review.md").read_text()
    for feature in UNCERTAIN_AVAILABILITY_FEATURES:
        assert feature in text
    assert set(FEATURE_SETS["no_uncertain_availability"]) == {
        "CreditScore", "Geography", "Gender", "Age", "EstimatedSalary"}


def test_subgroups_state_insufficient_data_honestly(trained):
    _, _, report = trained
    groups = report["subgroups"]["groups"]
    assert set(groups) == {"Geography", "Gender", "Age band"}
    for entries in groups.values():
        assert sum(g["rows"] for g in entries) == 120  # every test customer is in exactly one group
        for g in entries:
            if g["status"] == "insufficient_data":
                assert g["metrics"] is None and g["reason"]
            else:
                assert g["positives"] >= m.MIN_SUBGROUP_CLASS and g["rows"] >= m.MIN_SUBGROUP_ROWS
    # the 120-row synthetic test partition is too small for most slices; that must be visible
    assert any(g["status"] == "insufficient_data" for gs in groups.values() for g in gs)


# ---- artifact ------------------------------------------------------------------------------

def test_artifact_loads_and_matches_stored_test_predictions(trained):
    csv_path, _, report = trained
    _, evaluation, artifact = artifacts_of(trained)
    assert file_sha256(artifact) == report["artifact_sha256"]
    model = joblib.load(artifact)
    assert isinstance(model, ScoringModel) and model.feature_columns == FEATURE_SETS["full"]
    frame = validate_csv(csv_path, "synthetic").frame
    test = evaluation[evaluation.partition == "test"].merge(frame, on=ID)
    scored = model.predict_proba(test)[:, 1]
    assert np.allclose(scored, test.probability, atol=1e-9)
    assert ((scored >= 0) & (scored <= 1)).all()
    assert np.allclose(model.predict_proba(test).sum(axis=1), 1)
    # extra columns and column order do not matter, only the model's own features are used
    reordered = test[list(reversed(test.columns))]
    assert np.allclose(model.predict_proba(reordered)[:, 1], scored)
