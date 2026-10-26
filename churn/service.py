from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session

from churn.config import BANKING, risk_band
from churn.db import (
    Customer,
    Dataset,
    EvaluationPrediction,
    ModelVersion,
    Organization,
    Prediction,
    SplitAssignment,
)
from churn.ml.data import validate_csv
from churn.ml.lineage import REPORT_SCHEMA_VERSION, file_sha256, id_fingerprint


def import_dataset(session: Session, csv_path: Path, organization_name: str,
                   source_kind: str) -> Dataset:
    validated = validate_csv(csv_path, source_kind)
    with session.begin():
        organization = session.scalar(select(Organization).where(Organization.name == organization_name))
        if organization is None:
            organization = Organization(name=organization_name)
            session.add(organization)
            session.flush()
        existing = session.scalar(select(Dataset).where(
            Dataset.organization_id == organization.id, Dataset.sha256 == validated.sha256))
        if existing:
            # The same bytes must never be relabelled or silently re-admitted under other rules.
            if (existing.source_kind != source_kind
                    or existing.contract_version != validated.contract_version
                    or existing.quality_report_sha256 != validated.report_sha256):
                raise ValueError(
                    "This exact file is already imported as "
                    f"{existing.source_kind} under {existing.contract_version}; "
                    "it cannot be re-imported with a different source kind or contract")
            return existing
        external_ids = [str(value) for value in validated.frame[BANKING.customer_id_column]]
        for start in range(0, len(external_ids), 500):
            clash = session.scalars(select(Customer.external_id).where(
                Customer.organization_id == organization.id,
                Customer.external_id.in_(external_ids[start:start + 500])).limit(1)).first()
            if clash is not None:
                raise ValueError(f"Customer {clash} already exists in organization")
        dataset = Dataset(organization_id=organization.id, definition_key=BANKING.key,
                          sha256=validated.sha256, rows=validated.rows, source_kind=source_kind,
                          contract_version=validated.contract_version,
                          quality_status=validated.report["status"],
                          quality_report_sha256=validated.report_sha256,
                          quality_report=validated.report)
        session.add(dataset)
        session.flush()
        records = validated.frame.to_dict(orient="records")
        session.execute(insert(Customer), [
            {"id": str(uuid4()), "organization_id": organization.id,
             "external_id": str(row[BANKING.customer_id_column]), "dataset_id": dataset.id,
             "attributes": {column: row[column] for column in BANKING.feature_columns},
             "observed_exit": int(row[BANKING.target_column])}
            for row in records])
    return dataset


def dataset_lineage_status(dataset: Dataset) -> str:
    """'verified' only when the dataset was imported under a recorded contract and quality report."""
    return "verified" if dataset.contract_version and dataset.quality_report_sha256 else "unverified"


def model_lineage_status(model: ModelVersion) -> str:
    """'verified' only when an artifact hash and dataset/split lineage were recorded at registration."""
    return "verified" if model.artifact_sha256 and model.lineage else "unverified"


def _verified_file(directory: Path, name: str, expected_sha256: str) -> Path:
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(path)
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"{name} does not match the hash recorded in its training report")
    return path


def register_model(session: Session, report_path: Path, organization_name: str) -> ModelVersion:
    report = json.loads(report_path.read_text())
    if report.get("report_schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("Training report predates split/evaluation lineage; retrain with the current code")
    lineage = report["dataset"]
    directory = report_path.resolve().parent
    artifact = _verified_file(directory, f"{report['version']}.joblib", report["artifact_sha256"])
    splits_path = _verified_file(directory, report["split"]["assignments_file"],
                                 report["split"]["assignments_sha256"])
    evaluation_path = _verified_file(directory, report["evaluation_file"], report["evaluation_sha256"])
    splits = pd.read_csv(splits_path, dtype=str)
    evaluation = pd.read_csv(evaluation_path, dtype={BANKING.customer_id_column: str})
    id_col = BANKING.customer_id_column
    for name, digest in report["split"]["customer_id_sha256"].items():
        if id_fingerprint(splits.loc[splits["partition"] == name, id_col]) != digest:
            raise ValueError(f"Split assignments for {name!r} do not match the training report")
    if set(splits["partition"]) != set(report["split"]["rows"]) or splits[id_col].duplicated().any():
        raise ValueError("Split assignments are inconsistent with the training report")
    with session.begin():
        dataset = session.scalar(select(Dataset).join(Organization).where(
            Organization.name == organization_name, Dataset.sha256 == lineage["sha256"]))
        if dataset is None:
            raise ValueError("Import this exact dataset into this organization before registering its model")
        mismatches = [name for name, expected in (
            ("definition", report["definition"]["key"] == dataset.definition_key),
            ("source_kind", lineage["source_kind"] == dataset.source_kind),
            ("contract_version", lineage["contract_version"] == dataset.contract_version),
            ("quality_report_sha256", lineage["quality_report_sha256"] == dataset.quality_report_sha256),
            ("rows", lineage["rows"] == dataset.rows)) if not expected]
        if mismatches:
            raise ValueError(f"Training report does not match the imported dataset: {mismatches}")
        if session.get(ModelVersion, report["version"]):
            raise ValueError("Model version already registered")
        customers = dict(session.execute(select(Customer.external_id, Customer.id).where(
            Customer.dataset_id == dataset.id)).all())
        if set(splits[id_col]) != set(customers):
            raise ValueError("Split assignments do not cover exactly the customers of the imported dataset")
        if not set(evaluation[id_col]) <= set(customers):
            raise ValueError("Evaluation predictions reference customers outside the imported dataset")
        merged = evaluation.merge(splits, on=id_col, suffixes=("", "_assigned"))
        if len(merged) != len(evaluation) or (merged["partition"] != merged["partition_assigned"]).any():
            raise ValueError("Evaluation predictions disagree with the split assignments")
        if set(evaluation["partition"]) - {"calibration", "test"}:
            raise ValueError("Evaluation predictions may only cover calibration and test customers")
        keys = ("framing", "evidence_status", "split", "selection", "calibration", "threshold",
                "review_capacity", "held_out_test", "sensitivity_analyses", "subgroups", "limitations",
                "run_config_sha256")
        held_out = report["held_out_test"]
        model = ModelVersion(
            id=report["version"], dataset_id=dataset.id, definition_key=BANKING.key,
            artifact_path=str(artifact), artifact_sha256=report["artifact_sha256"],
            lineage={"dataset": lineage, "split": report["split"]},
            evaluation={key: report[key] for key in keys},
            metrics={"roc_auc": held_out["roc_auc"], "pr_auc": held_out["pr_auc"],
                     "brier": held_out["calibration"]["brier"], "prevalence": held_out["prevalence"],
                     "test_rows": held_out["rows"],
                     "recall_at_review_capacity": held_out["at_review_capacity"]["recall"],
                     "f1_at_threshold": held_out["at_threshold"]["f1"]},
            trained_at=datetime.fromisoformat(report["trained_at"]), status="candidate")
        session.add(model)
        session.flush()
        session.execute(insert(SplitAssignment), [
            {"model_id": model.id, "customer_id": customers[row[id_col]], "partition": row["partition"]}
            for row in splits.to_dict(orient="records")])
        session.execute(insert(EvaluationPrediction), [
            {"model_id": model.id, "customer_id": customers[row[id_col]], "partition": row["partition"],
             "label": int(row["label"]), "raw_probability": float(row["raw_probability"]),
             "probability": float(row["probability"])}
            for row in evaluation.to_dict(orient="records")])
    return model


def load_pipeline(model: ModelVersion):
    # joblib is pickle: only load files whose hash was recorded at registration and still matches.
    if not model.artifact_sha256:
        raise ValueError(f"Model {model.id} has unverified lineage (no artifact hash); it will not be loaded")
    if file_sha256(Path(model.artifact_path)) != model.artifact_sha256:
        raise ValueError(f"Artifact for {model.id} changed after registration")
    return joblib.load(model.artifact_path)


def score_customer(session: Session, customer: Customer, model: ModelVersion,
                   score_kind: str = "retrospective_on_demand") -> Prediction:
    if customer.dataset_id != model.dataset_id:
        raise ValueError("Customer and model dataset versions differ")
    pipeline = load_pipeline(model)
    row = pd.DataFrame([customer.attributes], columns=BANKING.feature_columns)
    probability = float(pipeline.predict_proba(row)[0, 1])
    prediction = Prediction(customer_id=customer.id, model_id=model.id, probability=probability,
                            risk_band=risk_band(probability), score_kind=score_kind)
    session.add(prediction)
    session.commit()
    session.refresh(prediction)
    return prediction


def score_customers(session: Session, customers: list[Customer], model: ModelVersion,
                    score_kind: str = "retrospective_batch") -> int:
    """Score imported, already-labeled customers. Training and calibration customers are scored
    in-sample; the partition of each is recorded separately in ``split_assignments``."""
    if any(customer.dataset_id != model.dataset_id for customer in customers):
        raise ValueError("Customer and model dataset versions differ")
    if not customers:
        return 0
    pipeline = load_pipeline(model)
    frame = pd.DataFrame([customer.attributes for customer in customers], columns=BANKING.feature_columns)
    probabilities = pipeline.predict_proba(frame)[:, 1]
    session.add_all(Prediction(customer_id=customer.id, model_id=model.id,
                               probability=float(probability), risk_band=risk_band(float(probability)),
                               score_kind=score_kind)
                    for customer, probability in zip(customers, probabilities, strict=True))
    session.commit()
    return len(customers)


def latest_model(session: Session, organization_id: str) -> ModelVersion | None:
    return session.scalar(select(ModelVersion).join(Dataset).where(
        Dataset.organization_id == organization_id).order_by(ModelVersion.trained_at.desc()))


def latest_prediction_subquery():
    return (select(Prediction.customer_id, func.max(Prediction.scored_at).label("latest"))
            .group_by(Prediction.customer_id).subquery())

