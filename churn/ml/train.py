"""Retrospective banking ``Exited`` classification: selection, calibration, held-out evaluation.

Data discipline (also documented in docs/banking-ml-methodology.md):

1. ``train`` partition: cross-validated candidate comparison and the final refit.
2. ``calibration`` partition: calibrator choice (cross-fitted), calibrator fit, decision threshold.
3. ``test`` partition: read only by ``evaluate_run`` after ``fit_run`` has returned. ``fit_run``
   does not accept test data, so isolation holds by construction.

The task is retrospective classification of a cross-sectional snapshot. No prediction horizon
exists, so no output here is a prospective churn forecast.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import BANKING
from churn.ml import metrics as m
from churn.ml.contract import SOURCE_KINDS
from churn.ml.data import validate_csv
from churn.ml.lineage import REPORT_SCHEMA_VERSION, SPLIT_SEED, file_sha256, id_fingerprint
from churn.ml.model import CALIBRATION_METHODS, Calibrator, ScoringModel
from churn.ml.splits import FRACTIONS, PARTITIONS, assign_partitions

CV_FOLDS = 5
DEFAULT_REVIEW_CAPACITY = 0.10
MIN_CALIBRATION_CLASS = 10
AGE_BANDS = [18, 30, 40, 50, 60, 101]

# Attribute groups used for the sensitivity analyses (see docs/banking-leakage-review.md).
DEMOGRAPHIC_FEATURES = ("Gender", "Age", "Geography")
UNCERTAIN_AVAILABILITY_FEATURES = ("Tenure", "Balance", "NumOfProducts", "IsActiveMember", "HasCrCard")
FEATURE_SETS = {
    "full": tuple(BANKING.feature_columns),
    "no_demographics": tuple(c for c in BANKING.feature_columns if c not in DEMOGRAPHIC_FEATURES),
    "no_uncertain_availability": tuple(
        c for c in BANKING.feature_columns if c not in UNCERTAIN_AVAILABILITY_FEATURES),
}
# Simplest first; used by the one-standard-error rule.
COMPLEXITY = {"dummy": 0, "logistic_regression": 1, "gradient_boosting": 2}


def search_space(seed: int) -> dict[str, tuple[object, dict[str, list]]]:
    return {
        "dummy": (DummyClassifier(strategy="prior"), {}),
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=seed),
            {"classifier__C": [0.1, 1.0, 10.0]}),
        # Tree boosting can capture non-monotone effects (e.g. age) and interactions that a
        # linear model cannot. Depth, rate, and rounds are bounded to 8 combinations.
        "gradient_boosting": (
            HistGradientBoostingClassifier(random_state=seed, early_stopping=False,
                                           min_samples_leaf=20, l2_regularization=1.0),
            {"classifier__max_depth": [2, 3], "classifier__learning_rate": [0.05, 0.1],
             "classifier__max_iter": [100, 200]}),
    }


def make_pipeline(estimator, features: tuple[str, ...]) -> Pipeline:
    numeric = [c for c in BANKING.numeric_columns if c in features]
    categorical = [c for c in BANKING.categorical_columns if c in features]
    parts = [("numeric", StandardScaler(), numeric)]
    if categorical:
        parts.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                      categorical))
    # No imputer: the data contract rejects missing values, so imputation would only mask errors.
    return Pipeline([("preprocess", ColumnTransformer(parts, sparse_threshold=0)),
                     ("classifier", estimator)])


def select_candidate(x: pd.DataFrame, y: pd.Series, seed: int) -> tuple[dict, str, dict]:
    """Cross-validated comparison on the training partition only. Returns (table, name, params)."""
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed)
    features = FEATURE_SETS["full"]
    table: dict[str, dict] = {}
    for name, (estimator, grid) in search_space(seed).items():
        pipeline = make_pipeline(estimator, features)
        if grid:
            search = GridSearchCV(pipeline, grid, scoring="average_precision", cv=cv, refit=False)
            search.fit(x[list(features)], y)
            best = search.best_index_
            results = search.cv_results_
            mean, std = results["mean_test_score"][best], results["std_test_score"][best]
            params = {k.removeprefix("classifier__"): v for k, v in search.best_params_.items()}
            combos = len(results["params"])
        else:
            scores = cross_val_score(pipeline, x[list(features)], y, scoring="average_precision", cv=cv)
            mean, std, params, combos = scores.mean(), scores.std(), {}, 1
        table[name] = {"cv_pr_auc_mean": float(mean), "cv_pr_auc_std": float(std),
                       "cv_pr_auc_se": float(std / math.sqrt(CV_FOLDS)), "best_params": params,
                       "configurations_searched": combos}
    chosen = simplest_within_one_se(table)
    return table, chosen, table[chosen]["best_params"]


def simplest_within_one_se(table: dict[str, dict]) -> str:
    """Simplest candidate whose CV mean is within one standard error of the best candidate."""
    top = max(table, key=lambda n: table[n]["cv_pr_auc_mean"])
    floor = table[top]["cv_pr_auc_mean"] - table[top]["cv_pr_auc_se"]
    eligible = [n for n in table if table[n]["cv_pr_auc_mean"] >= floor]
    return min(eligible, key=lambda n: COMPLEXITY[n])


@dataclass
class FittedRun:
    model: ScoringModel
    calibration_method: str
    calibration_brier: dict[str, float]
    threshold: float
    calibration_oof: np.ndarray  # cross-fitted calibrated probabilities for the calibration rows
    calibration_raw: np.ndarray
    calibration_rows: int
    calibration_positives: int


def fit_run(train: pd.DataFrame, calibration: pd.DataFrame, features: tuple[str, ...],
            estimator_name: str, params: dict, seed: int) -> FittedRun:
    """Fit on ``train``; choose calibrator and threshold on ``calibration``. Takes no test data."""
    y_cal = calibration[BANKING.target_column].to_numpy()
    positives = int(y_cal.sum())
    if min(positives, len(y_cal) - positives) < MIN_CALIBRATION_CLASS:
        raise ValueError(f"The calibration partition needs at least {MIN_CALIBRATION_CLASS} "
                         "examples of each class")
    estimator = search_space(seed)[estimator_name][0].set_params(**params)
    pipeline = make_pipeline(estimator, features).fit(train[list(features)],
                                                      train[BANKING.target_column])
    raw = pipeline.predict_proba(calibration[list(features)])[:, 1]
    folds = StratifiedKFold(min(CV_FOLDS, positives, len(y_cal) - positives), shuffle=True,
                            random_state=seed)
    oof: dict[str, np.ndarray] = {}
    for method in CALIBRATION_METHODS:
        scores = np.zeros(len(raw))
        for fit_idx, eval_idx in folds.split(raw, y_cal):
            scores[eval_idx] = Calibrator(method).fit(raw[fit_idx], y_cal[fit_idx]).predict(raw[eval_idx])
        oof[method] = scores
    briers = {method: float(np.mean((oof[method] - y_cal) ** 2)) for method in CALIBRATION_METHODS}
    chosen = CALIBRATION_METHODS[0]
    for method in CALIBRATION_METHODS[1:]:  # simplest wins ties
        if briers[method] < briers[chosen] - 1e-12:
            chosen = method
    threshold = m.best_f1_threshold(y_cal, oof[chosen])
    calibrator = Calibrator(chosen).fit(raw, y_cal)
    return FittedRun(ScoringModel(pipeline, calibrator, tuple(features)), chosen, briers,
                     threshold, oof[chosen], raw, len(y_cal), positives)


def evaluate_run(run: FittedRun, test: pd.DataFrame, capacity: float, seed: int,
                 resamples: int) -> tuple[dict, np.ndarray, np.ndarray]:
    y = test[BANKING.target_column].to_numpy()
    raw = run.model.raw_proba(test)
    p = run.model.predict_proba(test)[:, 1]
    return m.heldout_metrics(y, p, run.threshold, capacity, raw, seed, resamples), raw, p


def age_band(ages: pd.Series) -> np.ndarray:
    bands = pd.cut(ages, bins=AGE_BANDS, right=False)
    return bands.map(lambda b: f"[{int(b.left)}, {int(b.right)})").to_numpy()


def _variant_summary(name: str, features: tuple[str, ...], run: FittedRun, result: dict,
                     p: np.ndarray, p_full: np.ndarray, y: np.ndarray, seed: int,
                     resamples: int) -> dict:
    ci = result["confidence_intervals_95"]
    summary = {
        "features": list(features), "excluded": [c for c in FEATURE_SETS["full"] if c not in features],
        "calibration_method": run.calibration_method, "threshold": run.threshold,
        "roc_auc": result["roc_auc"], "roc_auc_ci95": ci["roc_auc"],
        "pr_auc": result["pr_auc"], "pr_auc_ci95": ci["pr_auc"],
        "brier": result["calibration"]["brier"], "brier_ci95": ci["brier"],
        "recall_at_review_capacity": result["at_review_capacity"]["recall"],
        "recall_at_review_capacity_ci95": ci["recall_at_capacity"],
        "f1_at_threshold": result["at_threshold"]["f1"],
    }
    if name != "full":
        summary["roc_auc_difference_vs_full"] = m.paired_bootstrap_difference(
            y, p, p_full, m.auc_score, seed, resamples)
    return summary


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def train(csv_path: Path, output_dir: Path, source_kind: str, *, seed: int = SPLIT_SEED,
          review_capacity: float = DEFAULT_REVIEW_CAPACITY,
          resamples: int = m.BOOTSTRAP_RESAMPLES) -> dict:
    if not 0 < review_capacity < 1:
        raise ValueError("review_capacity must be between 0 and 1")
    dataset = validate_csv(csv_path, source_kind)
    frame = dataset.frame
    id_col, target = BANKING.customer_id_column, BANKING.target_column
    partition = assign_partitions(frame[id_col], frame[target], seed)
    parts = {name: frame[partition == name] for name in PARTITIONS}

    # 1. Candidate comparison: training partition only.
    candidates, estimator_name, params = select_candidate(parts["train"], parts["train"][target], seed)

    # 2. Fit, calibrate, choose threshold. No test data is passed in.
    run_config = {"seed": seed, "fractions": FRACTIONS, "cv_folds": CV_FOLDS,
                  "review_capacity": review_capacity, "bootstrap_resamples": resamples,
                  "search_space": {n: {"grid": g, "estimator": type(e).__name__}
                                   for n, (e, g) in search_space(seed).items()},
                  "feature_sets": FEATURE_SETS, "calibration_methods": CALIBRATION_METHODS}
    config_hash = hashlib.sha256(_canonical(run_config).encode()).hexdigest()
    version = f"{BANKING.key}-{dataset.sha256[:12]}-{config_hash[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"{version}.joblib"
    if artifact.exists():
        raise FileExistsError(f"Artifact already exists: {artifact}")
    runs = {name: fit_run(parts["train"], parts["calibration"], features, estimator_name, params, seed)
            for name, features in FEATURE_SETS.items()}

    # 3. Single held-out evaluation, thresholds and calibrators frozen.
    y_test = parts["test"][target].to_numpy()
    results, probabilities, raw_scores = {}, {}, {}
    for name, run in runs.items():
        results[name], raw_scores[name], probabilities[name] = evaluate_run(
            run, parts["test"], review_capacity, seed, resamples)
    primary, primary_run = results["full"], runs["full"]
    p_test = probabilities["full"]
    subgroups = {}
    for label, values in (("Geography", parts["test"]["Geography"].to_numpy()),
                          ("Gender", parts["test"]["Gender"].to_numpy()),
                          ("Age band", age_band(parts["test"]["Age"]))):
        subgroups[label] = m.subgroup_report(y_test, p_test, values, primary_run.threshold,
                                             seed, resamples)

    joblib.dump(primary_run.model, artifact)
    splits = pd.DataFrame({id_col: frame[id_col], "partition": partition}).sort_values(id_col)
    splits_file = output_dir / f"{version}.splits.csv"
    splits.to_csv(splits_file, index=False, lineterminator="\n")
    cal, test = parts["calibration"], parts["test"]
    evaluation = pd.concat([
        pd.DataFrame({id_col: cal[id_col].to_numpy(), "partition": "calibration",
                      "label": cal[target].to_numpy(), "raw_probability": primary_run.calibration_raw,
                      "probability": primary_run.calibration_oof}),
        pd.DataFrame({id_col: test[id_col].to_numpy(), "partition": "test", "label": y_test,
                      "raw_probability": raw_scores["full"], "probability": p_test}),
    ]).sort_values(id_col)
    evaluation_file = output_dir / f"{version}.evaluation.csv"
    evaluation.to_csv(evaluation_file, index=False, lineterminator="\n")

    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "version": version, "definition": BANKING.model_dump(),
        "framing": ("Retrospective Exited classification on a cross-sectional snapshot. No observation "
                    "date, feature cutoff, or prediction horizon exists: this is not a prospective "
                    "churn forecast."),
        "evidence_status": ("SYNTHETIC data: every metric describes generated data only and says "
                            "nothing about real banking performance."
                            if source_kind == "synthetic" else
                            "EXTERNAL data: metrics describe this file only, retrospectively."),
        "dataset": {
            "sha256": dataset.sha256, "rows": dataset.rows, "source_kind": dataset.source_kind,
            "contract_version": dataset.contract_version, "quality_status": dataset.report["status"],
            "quality_report_sha256": dataset.report_sha256},
        "run_config": run_config, "run_config_sha256": config_hash,
        "split": {
            "strategy": ("stratified random, deterministic from sorted customer ids; one row per "
                         "customer, no temporal ordering available"),
            "random_state": seed, "fractions": FRACTIONS,
            "roles": {"train": "fit candidates, cross-validated selection, final refit",
                      "calibration": "calibrator choice and fit, decision threshold",
                      "test": "evaluated once after everything else is frozen"},
            "rows": {n: len(p) for n, p in parts.items()},
            "positives": {n: int(p[target].sum()) for n, p in parts.items()},
            "prevalence_overall": float(frame[target].mean()),
            "customer_id_sha256": {n: id_fingerprint(p[id_col]) for n, p in parts.items()},
            "disjoint": True,
            "assignments_file": splits_file.name, "assignments_sha256": file_sha256(splits_file)},
        "selection": {
            "data": "train partition only",
            "metric": f"average precision, {CV_FOLDS}-fold stratified cross-validation",
            "rule": "simplest candidate whose mean is within one standard error of the best",
            "candidates": candidates, "selected": estimator_name, "selected_params": params,
            "note": ("Cross-validated scores are optimistic for the selected configuration because "
                     "it was chosen among several; only the held-out test is an unbiased estimate.")},
        "calibration": {
            "data": "calibration partition only; method chosen by cross-fitting within it",
            "rows": primary_run.calibration_rows, "positives": primary_run.calibration_positives,
            "cross_fitted_brier": primary_run.calibration_brier,
            "selected_method": primary_run.calibration_method},
        "threshold": {
            "rule": "maximizes F1 on cross-fitted calibrated calibration-partition scores",
            "value": primary_run.threshold, "selected_on": "calibration",
            "never_tuned_on_test": True},
        "review_capacity": {
            "share": review_capacity,
            "note": ("Illustrative capacity (share of customers a team could review), not derived from "
                     "any bank's workload. Ties in score are broken by raw model score.")},
        "held_out_test": primary,
        "sensitivity_analyses": {
            "note": ("Variants reuse the selected estimator and hyperparameters, are refit and "
                     "recalibrated on the same partitions, and are reported only. They do not "
                     "change the primary model. Differences are associations in one snapshot, "
                     "not causal effects."),
            "variants": {n: _variant_summary(n, FEATURE_SETS[n], runs[n], results[n],
                                             probabilities[n], p_test, y_test, seed, resamples)
                         for n in FEATURE_SETS}},
        "subgroups": {"note": (f"Computed on the held-out test partition at the fixed threshold. A group "
                               f"needs >= {m.MIN_SUBGROUP_ROWS} rows and >= {m.MIN_SUBGROUP_CLASS} of "
                               "each class, otherwise only counts are shown. Groups are descriptive "
                               "slices, not a fairness verdict."),
                      "groups": subgroups},
        "artifact_sha256": file_sha256(artifact),
        "evaluation_file": evaluation_file.name, "evaluation_sha256": file_sha256(evaluation_file),
        "trained_at": datetime.now(UTC).isoformat(),
        "limitations": [
            "Cross-sectional label; prediction horizon and feature cutoff unknown",
            "Features may have been recorded after the exit (see docs/banking-leakage-review.md)",
            "No prospective calibration or intervention effectiveness assessment",
            "Scores for training and calibration customers are in-sample; they are not performance",
        ],
    }
    if source_kind == "synthetic":
        report["limitations"].append("SYNTHETIC data: metrics say nothing about real banking performance")
    (output_dir / f"{version}.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the banking Exited classifier")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--source-kind", choices=SOURCE_KINDS, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--review-capacity", type=float, default=DEFAULT_REVIEW_CAPACITY)
    parser.add_argument("--bootstrap-resamples", type=int, default=m.BOOTSTRAP_RESAMPLES)
    args = parser.parse_args()
    report = train(args.csv, args.output, args.source_kind, seed=args.seed,
                   review_capacity=args.review_capacity, resamples=args.bootstrap_resamples)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
