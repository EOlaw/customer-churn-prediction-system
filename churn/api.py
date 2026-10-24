from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from churn.config import BANKING
from churn.db import Customer, Dataset, ModelVersion, Organization, Prediction, session_factory
from churn.service import latest_model, score_customer

factory, engine = session_factory()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("API_KEY"):
        raise RuntimeError("API_KEY must be set")
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
             "contract_version": d.contract_version, "quality_status": d.quality_status,
             "quality_report_sha256": d.quality_report_sha256,
             "quality_issues": [{"code": i["code"], "severity": i["severity"],
                                 "column": i["column"], "count": i["count"]}
                                for i in d.quality_report["issues"]],
             "imported_at": d.imported_at} for d in items]


@app.get("/api/v1/models")
def models(org: Auth, session: Db):
    items = session.scalars(select(ModelVersion).join(Dataset).where(
        Dataset.organization_id == org.id).order_by(ModelVersion.trained_at.desc())).all()
    return [{"id": m.id, "status": m.status, "metrics": m.metrics,
             "dataset_id": m.dataset_id, "lineage": m.lineage,
             "artifact_sha256": m.artifact_sha256, "trained_at": m.trained_at} for m in items]


def customer_view(customer: Customer, prediction: Prediction | None) -> dict:
    return {"id": customer.id, "external_id": customer.external_id,
            "attributes": customer.attributes, "observed_exit": customer.observed_exit,
            "prediction": None if prediction is None else {
                "id": prediction.id, "probability": prediction.probability,
                "risk_band": prediction.risk_band, "scored_at": prediction.scored_at,
                "model_id": prediction.model_id}}


def current_prediction(session: Session, customer_id: str) -> Prediction | None:
    return session.scalar(select(Prediction).where(Prediction.customer_id == customer_id)
                          .order_by(Prediction.scored_at.desc(), Prediction.id.desc()).limit(1))


def latest_prediction_ids():
    ranked = select(
        Prediction.id.label("prediction_id"), Prediction.customer_id,
        func.row_number().over(partition_by=Prediction.customer_id,
                               order_by=(Prediction.scored_at.desc(), Prediction.id.desc())).label("position"),
    ).subquery()
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
    latest_by_customer = predictions_by_customer(session, [customer.id for customer in rows])
    items = [customer_view(customer, latest_by_customer.get(customer.id)) for customer in rows]
    total = session.scalar(select(func.count()).select_from(query.subquery())) or 0
    return {"items": items, "page": page, "size": size, "total": total}


@app.get("/api/v1/customers/{customer_id}")
def customer_detail(customer_id: str, org: Auth, session: Db):
    customer = session.scalar(select(Customer).where(Customer.id == customer_id,
                                                     Customer.organization_id == org.id))
    if not customer:
        raise HTTPException(404, "Customer not found")
    view = customer_view(customer, current_prediction(session, customer.id))
    history = session.scalars(select(Prediction).where(Prediction.customer_id == customer.id)
                              .order_by(Prediction.scored_at.desc()).limit(20)).all()
    view["history"] = [{"id": p.id, "probability": p.probability, "risk_band": p.risk_band,
                        "model_id": p.model_id, "scored_at": p.scored_at} for p in history]
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
    try:
        prediction = score_customer(session, customer, model)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return customer_view(customer, prediction)["prediction"]


@app.get("/api/v1/analytics/overview")
def overview(org: Auth, session: Db):
    customers = session.scalars(select(Customer).where(Customer.organization_id == org.id)).all()
    count = len(customers)
    if count == 0:
        return {"customers": 0, "observed_churn_rate": None, "mean_model_score": None,
                "high_risk": None, "latest_scoring_date": None, "model": None,
                "source_kind": None}
    latest_by_customer = predictions_by_customer(session, [c.id for c in customers])
    latest = list(latest_by_customer.values())
    scored = latest
    model = latest_model(session, org.id)
    return {
        "customers": count,
        "observed_churn_rate": sum(c.observed_exit == 1 for c in customers) / count,
        "mean_model_score": sum(p.probability for p in scored) / len(scored) if scored else None,
        "high_risk": sum(p.risk_band in ("high", "critical") for p in scored) if scored else None,
        "latest_scoring_date": max(p.scored_at for p in scored) if scored else None,
        "model": None if model is None else {"id": model.id, "metrics": model.metrics,
                                                "status": model.status},
        "definition": BANKING.key,
        "source_kind": session.scalar(select(Dataset.source_kind).where(
            Dataset.organization_id == org.id).order_by(Dataset.imported_at.desc())),
        "scored_customers": len(scored),
    }


@app.get("/api/v1/analytics/risk-distribution")
def risk_distribution(org: Auth, session: Db):
    customers = session.scalars(select(Customer).where(Customer.organization_id == org.id)).all()
    latest_by_customer = predictions_by_customer(session, [c.id for c in customers])
    counts = {name: 0 for name in ("low", "medium", "high", "critical")}
    for customer in customers:
        prediction = latest_by_customer.get(customer.id)
        if prediction:
            counts[prediction.risk_band] += 1
    return [{"risk_band": name, "count": count} for name, count in counts.items()]


@app.get("/api/v1/analytics/segments")
def segments(org: Auth, session: Db):
    customers = session.scalars(select(Customer).where(Customer.organization_id == org.id)).all()
    latest_by_customer = predictions_by_customer(session, [c.id for c in customers])
    groups: dict[str, dict] = {}
    for customer in customers:
        name = str(customer.attributes.get("Geography", "Unknown"))
        group = groups.setdefault(name, {"segment": name, "customers": 0, "exited": 0,
                                         "score_total": 0.0, "scored": 0})
        group["customers"] += 1
        group["exited"] += customer.observed_exit or 0
        prediction = latest_by_customer.get(customer.id)
        if prediction:
            group["score_total"] += prediction.probability
            group["scored"] += 1
    return [{"segment": g["segment"], "customers": g["customers"],
             "observed_rate": g["exited"] / g["customers"],
             "mean_probability": g["score_total"] / g["scored"] if g["scored"] else None}
            for g in groups.values()]
