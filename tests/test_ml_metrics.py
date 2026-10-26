from __future__ import annotations

import numpy as np
import pytest

from churn.ml import metrics as m
from churn.ml.model import CALIBRATION_METHODS, Calibrator
from churn.ml.train import simplest_within_one_se


def test_threshold_metrics_confusion_matrix():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.7, 0.3, 0.2, 0.1, 0.05])
    r = m.threshold_metrics(y, p, 0.5)
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (2, 1, 4, 1)
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3)
    assert r["flagged_share"] == pytest.approx(3 / 8)
    assert m.threshold_metrics(y, p, 0.95)["precision"] is None  # nothing flagged
    assert m.threshold_metrics(y, p, 0.0)["recall"] == 1.0  # threshold is inclusive


def test_best_f1_threshold_prefers_higher_threshold_on_ties():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.9, 0.1, 0.1])
    assert m.best_f1_threshold(y, p) == 0.9
    assert m.threshold_metrics(y, p, 0.9)["f1"] == 1.0


def test_capacity_metrics_top_k_and_deterministic_ties():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    r = m.capacity_metrics(y, p, 0.2)
    assert r["reviewed"] == 2 and r["true_positives_found"] == 1
    assert r["recall"] == pytest.approx(1 / 3) and r["precision"] == 0.5
    assert r["lift"] == pytest.approx(0.5 / 0.3)
    # capacity rounds up, and a tie at the cut is resolved by the secondary score, not row order
    tie = np.array([0.5] * 10)
    secondary = np.arange(10, dtype=float)  # row 9 (a positive) ranks first
    assert m.capacity_metrics(y, tie, 0.1, secondary)["true_positives_found"] == 1
    assert m.capacity_metrics(y, tie, 0.05)["reviewed"] == 1


def test_reliability_table_partitions_every_row():
    rng = np.random.RandomState(0)
    p = rng.uniform(size=500)
    y = (rng.uniform(size=500) < p).astype(int)
    table = m.reliability_table(y, p)
    assert sum(r["rows"] for r in table) == 500
    assert [r["bin_low"] for r in table] == pytest.approx(np.linspace(0, 0.9, 10))
    assert m.reliability_table(np.array([1]), np.array([1.0]))[-1]["rows"] == 1  # p == 1 in last bin
    for r in table:
        assert r["observed_ci95_low"] <= r["observed_rate"] <= r["observed_ci95_high"]


def test_calibration_summary_calibrated_vs_overconfident():
    rng = np.random.RandomState(1)
    p = rng.uniform(0.02, 0.98, 4000)
    y = (rng.uniform(size=4000) < p).astype(int)
    good = m.calibration_summary(y, p)
    assert good["expected_calibration_error"] < 0.04
    assert good["calibration_slope"] == pytest.approx(1.0, abs=0.15)
    logit = np.log(p / (1 - p))
    sharp = 1 / (1 + np.exp(-3 * logit))  # same ranking, overconfident
    bad = m.calibration_summary(y, sharp)
    assert bad["expected_calibration_error"] > good["expected_calibration_error"] + 0.05
    assert bad["brier"] > good["brier"]
    assert bad["calibration_slope"] < 0.6
    assert good["brier_skill_score"] > 0
    assert m.calibration_summary(y, np.full(4000, y.mean()))["brier_skill_score"] == pytest.approx(0, abs=1e-9)
    assert m.calibration_summary(y, np.full(4000, 0.5))["calibration_slope"] is None


def test_brier_matches_hand_calculation():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.1, 0.6, 0.3])
    expected = np.mean([(0.2) ** 2, 0.1**2, 0.4**2, 0.3**2])
    assert m.calibration_summary(y, p)["brier"] == pytest.approx(expected)


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    rng = np.random.RandomState(2)
    p = rng.uniform(size=300)
    y = (rng.uniform(size=300) < p).astype(int)
    first = m.bootstrap_ci(y, p, m.auc_score, seed=7, resamples=200)
    assert first == m.bootstrap_ci(y, p, m.auc_score, seed=7, resamples=200)
    assert first != m.bootstrap_ci(y, p, m.auc_score, seed=8, resamples=200)
    assert first[0] < m.auc_score(y, p) < first[1]
    assert m.bootstrap_ci(np.zeros(20, dtype=int), np.linspace(0, 1, 20), m.auc_score, 1, 50) is None


def test_heldout_metrics_structure_and_fixed_threshold():
    rng = np.random.RandomState(3)
    p = rng.uniform(size=400)
    y = (rng.uniform(size=400) < p * 0.5).astype(int)
    r = m.heldout_metrics(y, p, 0.4, 0.1, p, seed=1, resamples=100)
    assert r["at_threshold"]["threshold"] == 0.4  # never altered inside the evaluation
    cm = r["at_threshold"]["confusion_matrix"]
    assert sum(cm.values()) == 400 and cm["true_positive"] + cm["false_negative"] == r["positives"]
    assert r["reference"]["pr_auc"] == pytest.approx(y.mean())
    assert set(r["confidence_intervals_95"]) >= {"roc_auc", "pr_auc", "brier", "recall_at_capacity"}
    assert r["calibration"]["reliability"]


def test_subgroup_report_requires_minimum_samples():
    rng = np.random.RandomState(4)
    n = 400
    groups = np.array(["big"] * 300 + ["small"] * 20 + ["rare"] * 80)
    p = rng.uniform(size=n)
    y = (rng.uniform(size=n) < p * 0.5).astype(int)
    y[300:320] = 1  # 'small' has few rows and no negatives
    y[320:] = 0
    y[320] = 1  # 'rare' has a single positive
    report = {g["group"]: g for g in m.subgroup_report(y, p, groups, 0.5, seed=1, resamples=100)}
    assert report["big"]["status"] == "ok" and report["big"]["metrics"]["roc_auc"] is not None
    assert report["big"]["metrics"]["confidence_intervals_95"]["roc_auc"] is not None
    assert report["small"]["status"] == "insufficient_data" and report["small"]["metrics"] is None
    assert "fewer than 50 rows" in report["small"]["reason"]
    assert report["rare"]["status"] == "insufficient_data"
    assert "positives" in report["rare"]["reason"]
    assert report["rare"]["rows"] == 80 and report["rare"]["positives"] == 1  # counts stay visible
    assert sum(g["rows"] for g in report.values()) == n


def test_paired_difference_is_zero_for_identical_scores():
    rng = np.random.RandomState(5)
    p = rng.uniform(size=200)
    y = (rng.uniform(size=200) < p).astype(int)
    r = m.paired_bootstrap_difference(y, p, p, m.auc_score, seed=1, resamples=100)
    assert r["difference"] == 0 and r["ci95"] == [0.0, 0.0]


def test_calibrators():
    rng = np.random.RandomState(6)
    raw = rng.uniform(0.01, 0.99, 3000)
    truth = 1 / (1 + np.exp(-0.4 * np.log(raw / (1 - raw))))  # raw scores are overconfident
    y = (rng.uniform(size=3000) < truth).astype(int)
    assert (Calibrator("none").fit(raw, y).predict(raw) == raw).all()
    for method in ("sigmoid", "isotonic"):
        calibrated = Calibrator(method).fit(raw, y).predict(raw)
        assert calibrated.min() >= 0 and calibrated.max() <= 1
        assert m.calibration_summary(y, calibrated)["brier"] < m.calibration_summary(y, raw)["brier"]
        order = np.argsort(raw)
        assert (np.diff(calibrated[order]) >= -1e-12).all()  # monotone, ranking preserved
    with pytest.raises(RuntimeError):
        Calibrator("sigmoid").predict(raw)
    with pytest.raises(ValueError):
        Calibrator("platt")
    assert CALIBRATION_METHODS[0] == "none"


def test_one_standard_error_rule_prefers_simpler_models():
    table = {
        "dummy": {"cv_pr_auc_mean": 0.13, "cv_pr_auc_se": 0.003},
        "logistic_regression": {"cv_pr_auc_mean": 0.34, "cv_pr_auc_se": 0.03},
        "gradient_boosting": {"cv_pr_auc_mean": 0.36, "cv_pr_auc_se": 0.03},
    }
    assert simplest_within_one_se(table) == "logistic_regression"
    table["gradient_boosting"]["cv_pr_auc_mean"] = 0.45
    assert simplest_within_one_se(table) == "gradient_boosting"
