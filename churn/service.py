from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from uuid import uuid4

from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session

from churn.config import BANKING, risk_band
from churn.db import Customer, Dataset, ModelVersion, Organization, Prediction
from churn.ml.data import validate_csv
from churn.ml.lineage import REPORT_SCHEMA_VERSION, file_sha256


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


def register_model(session: Session, report_path: Path, organization_name: str) -> ModelVersion:
    report = json.loads(report_path.read_text())
    if report.get("report_schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("Training report predates dataset lineage; retrain with the current code")
    lineage = report["dataset"]
    artifact = report_path.with_suffix(".joblib").resolve()
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    if file_sha256(artifact) != report["artifact_sha256"]:
        raise ValueError("Model artifact does not match the hash recorded in its training report")
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
        model = ModelVersion(id=report["version"], dataset_id=dataset.id,
                             definition_key=BANKING.key, artifact_path=str(artifact),
                             artifact_sha256=report["artifact_sha256"],
                             lineage={"dataset": lineage, "split": report["split"]},
                             metrics=report["held_out_test"],
                             trained_at=datetime.fromisoformat(report["trained_at"]), status="candidate")
        session.add(model)
    return model


def load_pipeline(model: ModelVersion):
    # joblib is pickle: only load files that still match the hash recorded at registration.
    if file_sha256(Path(model.artifact_path)) != model.artifact_sha256:
        raise ValueError(f"Artifact for {model.id} changed after registration")
    return joblib.load(model.artifact_path)


def score_customer(session: Session, customer: Customer, model: ModelVersion) -> Prediction:
    if customer.dataset_id != model.dataset_id:
        raise ValueError("Customer and model dataset versions differ")
    pipeline = load_pipeline(model)
    row = pd.DataFrame([customer.attributes], columns=BANKING.feature_columns)
    probability = float(pipeline.predict_proba(row)[0, 1])
    prediction = Prediction(customer_id=customer.id, model_id=model.id,
                            probability=probability, risk_band=risk_band(probability))
    session.add(prediction)
    session.commit()
    session.refresh(prediction)
    return prediction


def score_customers(session: Session, customers: list[Customer], model: ModelVersion) -> int:
    if any(customer.dataset_id != model.dataset_id for customer in customers):
        raise ValueError("Customer and model dataset versions differ")
    if not customers:
        return 0
    pipeline = load_pipeline(model)
    frame = pd.DataFrame([customer.attributes for customer in customers], columns=BANKING.feature_columns)
    probabilities = pipeline.predict_proba(frame)[:, 1]
    session.add_all(Prediction(customer_id=customer.id, model_id=model.id,
                               probability=float(probability), risk_band=risk_band(float(probability)))
                    for customer, probability in zip(customers, probabilities, strict=True))
    session.commit()
    return len(customers)


def latest_model(session: Session, organization_id: str) -> ModelVersion | None:
    return session.scalar(select(ModelVersion).join(Dataset).where(
        Dataset.organization_id == organization_id).order_by(ModelVersion.trained_at.desc()))


def latest_prediction_subquery():
    return (select(Prediction.customer_id, func.max(Prediction.scored_at).label("latest"))
            .group_by(Prediction.customer_id).subquery())

