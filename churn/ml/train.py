from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import BANKING
from churn.ml.contract import SOURCE_KINDS
from churn.ml.data import validate_csv
from churn.ml.lineage import REPORT_SCHEMA_VERSION, SPLIT_SEED, file_sha256, id_fingerprint


def train(csv_path: Path, output_dir: Path, source_kind: str) -> dict:
    dataset = validate_csv(csv_path, source_kind)
    frame = dataset.frame
    x = frame[list(BANKING.feature_columns)]
    y = frame[BANKING.target_column]
    ids = frame[BANKING.customer_id_column]
    x_dev, x_test, y_dev, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SPLIT_SEED, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_dev, y_dev, test_size=0.25, random_state=SPLIT_SEED, stratify=y_dev
    )
    partitions = {"train": x_train.index, "validation": x_val.index, "test": x_test.index}
    if (set(partitions["train"]) & set(partitions["validation"])
            or set(partitions["train"]) & set(partitions["test"])
            or set(partitions["validation"]) & set(partitions["test"])
            or sum(len(index) for index in partitions.values()) != len(frame)):
        raise RuntimeError("Split partitions must be disjoint and cover every row")
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), list(BANKING.numeric_columns)),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         list(BANKING.categorical_columns)),
    ])
    candidates = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SPLIT_SEED),
    }
    validation = {}
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocess", clone(prep)), ("classifier", estimator)])
        pipeline.fit(x_train, y_train)
        scores = pipeline.predict_proba(x_val)[:, 1]
        validation[name] = {
            "roc_auc": float(roc_auc_score(y_val, scores)),
            "pr_auc": float(average_precision_score(y_val, scores)),
            "brier": float(brier_score_loss(y_val, scores)),
        }
        fitted[name] = pipeline
    winner = max(validation, key=lambda name: (validation[name]["pr_auc"], -validation[name]["brier"]))
    model = fitted[winner]
    test_scores = model.predict_proba(x_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_scores)),
        "pr_auc": float(average_precision_score(y_test, test_scores)),
        "brier": float(brier_score_loss(y_test, test_scores)),
        "prevalence": float(y_test.mean()),
        "test_rows": len(y_test),
    }
    version = f"{BANKING.key}-{dataset.sha256[:12]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"{version}.joblib"
    if artifact.exists():
        raise FileExistsError(f"Artifact already exists: {artifact}")
    joblib.dump(model, artifact)
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "version": version, "definition": BANKING.model_dump(),
        "dataset": {
            "sha256": dataset.sha256, "rows": dataset.rows, "source_kind": dataset.source_kind,
            "contract_version": dataset.contract_version,
            "quality_status": dataset.report["status"],
            "quality_report_sha256": dataset.report_sha256,
        },
        "split": {
            "strategy": "stratified random; one row per customer, no temporal ordering available",
            "random_state": SPLIT_SEED, "fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
            "rows": {name: len(index) for name, index in partitions.items()},
            "customer_id_sha256": {name: id_fingerprint(ids.loc[index])
                                   for name, index in partitions.items()},
        },
        "artifact_sha256": file_sha256(artifact),
        "trained_at": datetime.now(UTC).isoformat(),
        "selection_metric": "validation PR-AUC", "selected_model": winner,
        "validation": validation, "held_out_test": metrics,
        "limitations": ["Cross-sectional label; prediction horizon and feature cutoff unknown",
                        "No prospective calibration or intervention effectiveness assessment",
                        "Scores for imported customers include rows used for training"],
    }
    if dataset.source_kind == "synthetic":
        report["limitations"].append("SYNTHETIC data: metrics say nothing about real banking performance")
    (output_dir / f"{version}.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train banking churn model from a local CSV")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--source-kind", choices=SOURCE_KINDS, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    print(json.dumps(train(args.csv, args.output, args.source_kind), indent=2))


if __name__ == "__main__":
    main()
