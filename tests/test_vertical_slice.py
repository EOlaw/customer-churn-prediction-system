from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from churn import api
from churn.db import Customer, session_factory
from churn.demo import make_demo_data
from churn.migrate import init_schema
from churn.ml.data import DataQualityError, validate_csv
from churn.service import import_dataset, register_model, score_customers


def test_validation_rejects_duplicate_ids(tmp_path):
    path = make_demo_data(tmp_path / "demo.csv", rows=100)
    text = path.read_text().splitlines()
    text[2] = text[1]
    path.write_text("\n".join(text) + "\n")
    with pytest.raises(DataQualityError, match="duplicated identifier"):
        validate_csv(path, "synthetic")


def test_training_to_dashboard_and_auth(tmp_path, monkeypatch, trained):
    path, artifacts, report = trained
    assert report["held_out_test"]["rows"] == 120
    assert set(report["selection"]["candidates"]) == {"dummy", "logistic_regression", "gradient_boosting"}
    assert report["dataset"]["sha256"] == validate_csv(path, "synthetic").sha256
    assert report["definition"]["prediction_horizon"] is None

    factory, engine = session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    init_schema(engine)
    with factory() as session:
        dataset = import_dataset(session, path, "Portfolio demo", "synthetic")
        model = register_model(session, artifacts / f"{report['version']}.json",
                               "Portfolio demo")
        customers = session.scalars(select(Customer).where(Customer.dataset_id == dataset.id)).all()
        assert score_customers(session, list(customers), model) == 600

    monkeypatch.setattr(api, "factory", factory)
    monkeypatch.setattr(api, "engine", engine)
    monkeypatch.setenv("API_KEY", "test-secret")
    with TestClient(api.app) as client:
        assert client.get("/api/v1/analytics/overview").status_code == 401
        headers = {"X-API-Key": "test-secret"}
        overview = client.get("/api/v1/analytics/overview", headers=headers)
        assert overview.status_code == 200
        assert overview.json()["customers"] == 600
        assert "mean_model_score" not in overview.json()  # no pooled in-sample average
        assert sum(v["scored"] for v in overview.json()["partitions"].values()) == 600
        assert overview.json()["model"]["id"] == report["version"]
        assert overview.json()["source_kind"] == "synthetic"
        distribution = client.get("/api/v1/analytics/risk-distribution", headers=headers).json()
        assert distribution["partition"] == "test" and distribution["in_sample"] is False
        assert sum(item["count"] for item in distribution["items"]) == 120
        everyone = client.get("/api/v1/analytics/risk-distribution?partition=all", headers=headers).json()
        assert sum(item["count"] for item in everyone["items"]) == 600
        customer = client.get("/api/v1/customers?size=1", headers=headers).json()["items"][0]
        assert customer["prediction"]["model_id"] == report["version"]
        band = customer["prediction"]["risk_band"]
        filtered = client.get(f"/api/v1/customers?risk={band}&size=10", headers=headers).json()
        assert all(item["prediction"]["risk_band"] == band for item in filtered["items"])
        assert filtered["total"] >= len(filtered["items"])
        assert client.get(f"/api/v1/customers/{customer['id']}", headers=headers).status_code == 200
    assert json.loads((artifacts / f"{report['version']}.json").read_text())["dataset"]["rows"] == 600
