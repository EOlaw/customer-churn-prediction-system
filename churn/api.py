from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import and_, case, func, select
from sqlalchemy.orm import Session

from churn.config import BANKING
from churn.db import (
    Customer,
    Dataset,
    ModelVersion,
    Organization,
    Prediction,
    SplitAssignment,
    session_factory,
)
from churn.migrate import schema_status
from churn.service import dataset_lineage_status, latest_model, model_lineage_status, score_customer

factory, engine = session_factory()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("API_KEY"):
        raise RuntimeError("API_KEY must be set")
    status, version = schema_status(engine)
    if status != "current":
        raise RuntimeError(f"Database schema is {status} (version {version}). "
                           "Run `python -m churn.cli migrate` (SQLite is backed up first) or `init-db`.")
    yield
    engine.dispose()


app = FastAPI(title="Customer Churn Intelligence API", version="0.1.0", lifespan=lifespan)
origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if item.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                   allow_headers=["X-API-Key", "Content-Type"])


def get_session():
    with factory() as session:
        yield session


def authorized_org(session: Annotated[Session, Depends(get_session)],
                   x_api_key: Annotated[str | None, Header()] = None) -> Organization:
    configured = os.getenv("API_KEY", "")
    if not configured or not x_api_key or not secrets.compare_digest(x_api_key, configured):
        raise HTTPException(status_code=401, detail="Invalid API key")
    name = os.getenv("ORGANIZATION_NAME", "Portfolio demo")
    organization = session.scalar(select(Organization).where(Organization.name == name))
    if organization is None:
        raise HTTPException(status_code=404, detail="Organization has no imported dataset")
    return organization


Auth = Annotated[Organization, Depends(authorized_org)]
Db = Annotated[Session, Depends(get_session)]


@app.get("/health/live")
def live():
    return {"status": "live"}


@app.get("/health/ready")
def ready(session: Db):
    session.execute(select(1))
    return {"status": "ready"}


@app.get("/api/v1/definitions")
def definitions(_: Auth):
    return [BANKING.model_dump()]


@app.get("/api/v1/datasets")
def datasets(org: Auth, session: Db):
    items = session.scalars(select(Dataset).where(Dataset.organization_id == org.id)
                            .order_by(Dataset.imported_at.desc())).all()
    return [{"id": d.id, "definition_key": d.definition_key, "sha256": d.sha256,
             "rows": d.rows, "source_kind": d.source_kind,
             "lineage_status": dataset_lineage_status(d),
             "contract_version": d.contract_version, "quality_status": d.quality_status,
             "quality_report_sha256": d.quality_report_sha256,
             "quality_issues": None if d.quality_report is None else [
                 {"code": i["code"], "severity": i["severity"], "column": i["column"],
                  "count": i["count"]} for i in d.quality_report["issues"]],
             "imported_at": d.imported_at} for d in items]


@app.get("/api/v1/models")
def models(org: Auth, session: Db):
    items = session.scalars(select(ModelVersion).join(Dataset).where(
        Dataset.organization_id == org.id).order_by(ModelVersion.trained_at.desc())).all()
    return [{"id": m.id, "status": m.status, "metrics": m.metrics,
             "dataset_id": m.dataset_id, "lineage": m.lineage,
             "lineage_status": model_lineage_status(m), "evaluation_recorded": m.evaluation is not None,
             "artifact_sha256": m.artifact_sha256, "trained_at": m.trained_at} for m in items]


@app.get("/api/v1/models/{model_id}/evaluation")
def model_evaluation(model_id: str, org: Auth, session: Db):
    model = session.scalar(select(ModelVersion).join(Dataset).where(
        ModelVersion.id == model_id, Dataset.organization_id == org.id))
    if model is None:
        raise HTTPException(404, "Model not found")
    return {"id": model.id, "lineage_status": model_lineage_status(model),
            "evaluation": model.evaluation,
            "reason": None if model.evaluation else
            "No split assignments or held-out evaluation were recorded for this model (legacy, unverified)."}


SCORE_NOTE = ("Training and calibration customers are scored in-sample. Only the held-out test "
              "partition estimates out-of-sample behavior, and only retrospectively.")


def evidence(partition: str | None) -> dict:
    return {"partition": partition or "unknown",
            "in_sample": None if partition is None else partition != "test"}


def partitions_for(session: Session, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """(model_id, customer_id) -> partition, for the recorded split assignments only."""
    found: dict[tuple[str, str], str] = {}
    by_model: dict[str, list[str]] = {}
    for model_id, customer_id in pairs:
        by_model.setdefault(model_id, []).append(customer_id)
    for model_id, ids in by_model.items():
        for start in range(0, len(ids), 500):
            rows = session.execute(select(SplitAssignment.customer_id, SplitAssignment.partition).where(
                SplitAssignment.model_id == model_id,
                SplitAssignment.customer_id.in_(ids[start:start + 500]))).all()
            found.update({(model_id, customer_id): partition for customer_id, partition in rows})
    return found


def prediction_view(prediction: Prediction | None, partition: str | None) -> dict | None:
    if prediction is None:
        return None
    return {"id": prediction.id, "probability": prediction.probability,
            "risk_band": prediction.risk_band, "scored_at": prediction.scored_at,
            "model_id": prediction.model_id, "score_kind": prediction.score_kind or "legacy_unknown",
            **evidence(partition)}


def customer_view(customer: Customer, prediction: Prediction | None, partition: str | None) -> dict:
    return {"id": customer.id, "external_id": customer.external_id,
            "attributes": customer.attributes, "observed_exit": customer.observed_exit,
            "prediction": prediction_view(prediction, partition)}


def current_prediction(session: Session, customer_id: str) -> Prediction | None:
    return session.scalar(select(Prediction).where(Prediction.customer_id == customer_id)
                          .order_by(Prediction.scored_at.desc(), Prediction.id.desc()).limit(1))


def latest_prediction_ids(model_id: str | None = None):
    ranked = select(
        Prediction.id.label("prediction_id"), Prediction.customer_id,
        func.row_number().over(partition_by=Prediction.customer_id,
                               order_by=(Prediction.scored_at.desc(), Prediction.id.desc())).label("position"),
    )
    if model_id is not None:
        ranked = ranked.where(Prediction.model_id == model_id)
    ranked = ranked.subquery()
    return select(ranked.c.prediction_id, ranked.c.customer_id).where(ranked.c.position == 1).subquery()


def predictions_by_customer(session: Session, customer_ids: list[str]) -> dict[str, Prediction]:
    if not customer_ids:
        return {}
    latest = latest_prediction_ids()
    result = {}
    for start in range(0, len(customer_ids), 500):
        rows = session.scalars(select(Prediction).join(
            latest, latest.c.prediction_id == Prediction.id).where(
            Prediction.customer_id.in_(customer_ids[start:start + 500]))).all()
        result.update({row.customer_id: row for row in rows})
    return result


def customer_views(session: Session, rows: list[Customer]) -> list[dict]:
    latest = predictions_by_customer(session, [c.id for c in rows])
    partitions = partitions_for(session, [(p.model_id, p.customer_id) for p in latest.values()])
    return [customer_view(c, latest.get(c.id),
                          partitions.get((latest[c.id].model_id, c.id)) if c.id in latest else None)
            for c in rows]


@app.get("/api/v1/customers")
def customers(org: Auth, session: Db, search: str = "", risk: str | None = None,
              page: int = Query(1, ge=1), size: int = Query(20, ge=1, le=100)):
    query = select(Customer).where(Customer.organization_id == org.id)
    if search:
        query = query.where(Customer.external_id.ilike(f"%{search[:100]}%"))
    if risk:
        if risk not in {"low", "medium", "high", "critical"}:
            raise HTTPException(422, "Unknown risk band")
        latest = latest_prediction_ids()
        query = query.join(latest, latest.c.customer_id == Customer.id).join(
            Prediction, Prediction.id == latest.c.prediction_id).where(Prediction.risk_band == risk)
    rows = session.scalars(query.order_by(Customer.external_id).offset((page - 1) * size).limit(size)).all()
    total = session.scalar(select(func.count()).select_from(query.subquery())) or 0
    return {"items": customer_views(session, list(rows)), "page": page, "size": size, "total": total}


@app.get("/api/v1/customers/{customer_id}")
def customer_detail(customer_id: str, org: Auth, session: Db):
    customer = session.scalar(select(Customer).where(Customer.id == customer_id,
                                                     Customer.organization_id == org.id))
    if not customer:
        raise HTTPException(404, "Customer not found")
    view = customer_views(session, [customer])[0]
    history = session.scalars(select(Prediction).where(Prediction.customer_id == customer.id)
                              .order_by(Prediction.scored_at.desc()).limit(20)).all()
    partitions = partitions_for(session, [(p.model_id, p.customer_id) for p in history])
    view["history"] = [prediction_view(p, partitions.get((p.model_id, p.customer_id))) for p in history]
    return view


class PredictionRequest(BaseModel):
    customer_id: str


@app.post("/api/v1/predictions")
def predict(body: PredictionRequest, org: Auth, session: Db):
    customer = session.scalar(select(Customer).where(Customer.id == body.customer_id,
                                                     Customer.organization_id == org.id))
    if not customer:
        raise HTTPException(404, "Customer not found")
    model = latest_model(session, org.id)
    if not model:
        raise HTTPException(409, "No registered model")
    if model_lineage_status(model) != "verified":
        raise HTTPException(409, "The latest model has unverified lineage and will not be loaded")
    try:
        prediction = score_customer(session, customer, model)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    partition = partitions_for(session, [(model.id, customer.id)]).get((model.id, customer.id))
    return prediction_view(prediction, partition)


def model_summary(model: ModelVersion) -> dict:
    held_out = (model.evaluation or {}).get("held_out_test")
    return {
        "id": model.id, "status": model.status, "metrics": model.metrics,
        "lineage_status": model_lineage_status(model),
        "evaluation_recorded": model.evaluation is not None,
        "held_out": None if held_out is None else {
            "rows": held_out["rows"], "positives": held_out["positives"],
            "prevalence": held_out["prevalence"], "roc_auc": held_out["roc_auc"],
            "pr_auc": held_out["pr_auc"], "brier": held_out["calibration"]["brier"],
            "recall_at_review_capacity": held_out["at_review_capacity"]["recall"],
            "review_capacity_share": held_out["at_review_capacity"]["capacity_share"],
            "precision": held_out["at_threshold"]["precision"], "recall": held_out["at_threshold"]["recall"],
            "f1": held_out["at_threshold"]["f1"],
            "confidence_intervals_95": held_out["confidence_intervals_95"]},
        "evidence_status": (model.evaluation or {}).get("evidence_status"),
    }


@app.get("/api/v1/analytics/overview")
def overview(org: Auth, session: Db):
    count = session.scalar(select(func.count()).select_from(Customer).where(
        Customer.organization_id == org.id)) or 0
    source_kind = session.scalar(select(Dataset.source_kind).where(
        Dataset.organization_id == org.id).order_by(Dataset.imported_at.desc()))
    limits = ["Retrospective classification of a cross-sectional snapshot: no prediction horizon exists.",
              ("Future operational scoring is unavailable: it needs dated observations, a feature "
               "cutoff, and a defined outcome window."), SCORE_NOTE]
    if count == 0:
        return {"customers": 0, "observed_churn_rate": None, "model": None, "partitions": None,
                "source_kind": None, "definition": BANKING.key, "latest_scoring_date": None,
                "score_kinds": {}, "operational_scoring": {"available": False}, "limitations": limits}
    exits = session.scalar(select(func.count()).select_from(Customer).where(
        Customer.organization_id == org.id, Customer.observed_exit == 1)) or 0
    model = latest_model(session, org.id)
    partitions, kinds, latest_date = None, {}, None
    if model is not None:
        assigned = dict(session.execute(select(SplitAssignment.partition, func.count()).where(
            SplitAssignment.model_id == model.id).group_by(SplitAssignment.partition)).all())
        latest = latest_prediction_ids(model.id)
        band_high = func.sum(case((Prediction.risk_band.in_(("high", "critical")), 1), else_=0))
        scored = {row[0]: row[1:] for row in session.execute(
            select(func.coalesce(SplitAssignment.partition, "unknown"), func.count(),
                   func.avg(Prediction.probability), band_high)
            .select_from(Prediction).join(latest, latest.c.prediction_id == Prediction.id)
            .outerjoin(SplitAssignment, and_(SplitAssignment.model_id == Prediction.model_id,
                                             SplitAssignment.customer_id == Prediction.customer_id))
            .group_by(func.coalesce(SplitAssignment.partition, "unknown"))).all()}
        partitions = {}
        for name in ("train", "calibration", "test", "unknown"):
            n_scored, mean, high = scored.get(name, (0, None, 0))
            customers_in = (count - sum(assigned.values())) if name == "unknown" else assigned.get(name, 0)
            partitions[name] = {"customers": customers_in, "scored": n_scored,
                                "mean_score": None if mean is None else float(mean),
                                "high_risk": int(high or 0), **evidence(None if name == "unknown" else name)}
        kinds = {(kind or "legacy_unknown"): n for kind, n in session.execute(
            select(Prediction.score_kind, func.count()).where(Prediction.model_id == model.id)
            .group_by(Prediction.score_kind)).all()}
        latest_date = session.scalar(select(func.max(Prediction.scored_at)).where(
            Prediction.model_id == model.id))
    return {
        "customers": count, "observed_churn_rate": exits / count,
        "definition": BANKING.key, "source_kind": source_kind,
        "model": None if model is None else model_summary(model),
        "partitions": partitions, "score_kinds": kinds, "latest_scoring_date": latest_date,
        "operational_scoring": {"available": False, "reason": limits[1]},
        "limitations": limits,
    }


PARTITION_FILTERS = ("test", "train", "calibration", "unknown", "all")


@app.get("/api/v1/analytics/risk-distribution")
def risk_distribution(org: Auth, session: Db, partition: str = "test"):
    """Risk bands of the current model's latest retrospective scores within one partition.
    The default is the held-out test partition; 'train' and 'calibration' are in-sample."""
    if partition not in PARTITION_FILTERS:
        raise HTTPException(422, f"partition must be one of {PARTITION_FILTERS}")
    model = latest_model(session, org.id)
    counts = {name: 0 for name in ("low", "medium", "high", "critical")}
    if model is not None:
        latest = latest_prediction_ids(model.id)
        label = func.coalesce(SplitAssignment.partition, "unknown")
        query = (select(Prediction.risk_band, func.count()).select_from(Prediction)
                 .join(latest, latest.c.prediction_id == Prediction.id)
                 .join(Customer, Customer.id == Prediction.customer_id)
                 .outerjoin(SplitAssignment, and_(SplitAssignment.model_id == Prediction.model_id,
                                                  SplitAssignment.customer_id == Prediction.customer_id))
                 .where(Customer.organization_id == org.id).group_by(Prediction.risk_band))
        if partition != "all":
            query = query.where(label == partition)
        counts.update(dict(session.execute(query).all()))
    return {"partition": partition, "model_id": None if model is None else model.id,
            "in_sample": {"test": False, "all": None, "unknown": None}.get(partition, True),
            "note": SCORE_NOTE,
            "items": [{"risk_band": name, "count": n} for name, n in counts.items()]}


@app.get("/api/v1/analytics/segments")
def segments(org: Auth, session: Db):
    """Observed exit rate by geography from labels, plus the model's mean score on the held-out
    test customers only. In-sample scores are never averaged into a segment."""
    customers = session.scalars(select(Customer).where(Customer.organization_id == org.id)).all()
    model = latest_model(session, org.id)
    test_scores: dict[str, float] = {}
    if model is not None:
        latest = latest_prediction_ids(model.id)
        test_scores = dict(session.execute(
            select(Prediction.customer_id, Prediction.probability)
            .join(latest, latest.c.prediction_id == Prediction.id)
            .join(SplitAssignment, and_(SplitAssignment.model_id == Prediction.model_id,
                                        SplitAssignment.customer_id == Prediction.customer_id))
            .where(SplitAssignment.partition == "test")).all())
    groups: dict[str, dict] = {}
    for customer in customers:
        name = str(customer.attributes.get("Geography", "Unknown"))
        group = groups.setdefault(name, {"segment": name, "customers": 0, "exited": 0,
                                         "score_total": 0.0, "heldout_scored": 0})
        group["customers"] += 1
        group["exited"] += customer.observed_exit or 0
        if customer.id in test_scores:
            group["score_total"] += test_scores[customer.id]
            group["heldout_scored"] += 1
    return [{"segment": g["segment"], "customers": g["customers"],
             "observed_rate": g["exited"] / g["customers"],
             "heldout_customers": g["heldout_scored"],
             "heldout_mean_probability": g["score_total"] / g["heldout_scored"] if g["heldout_scored"] else None}
            for g in groups.values()]
