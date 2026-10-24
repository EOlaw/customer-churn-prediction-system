from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from churn import api
from churn.db import Base, Customer, session_factory
from churn.demo import make_demo_data
from churn.ml.data import DataQualityError, validate_csv
from churn.ml.train import train
from churn.service import import_dataset, register_model, score_customers


def test_validation_rejects_duplicate_ids(tmp_path):
    path = make_demo_data(tmp_path / "demo.csv", rows=100)
    text = path.read_text().splitlines()
    text[2] = text[1]
    path.write_text("\n".join(text) + "\n")
    with pytest.raises(DataQualityError, match="Duplicate"):
        validate_csv(path)


def test_training_to_dashboard_and_auth(tmp_path, monkeypatch):
    path = make_demo_data(tmp_path / "demo.csv")
    report = train(path, tmp_path / "artifacts")
    assert report["held_out_test"]["test_rows"] == 120
    assert set(report["validation"]) == {"dummy", "logistic_regression"}
    assert report["dataset_sha256"] == validate_csv(path).sha256
    assert report["definition"]["prediction_horizon"] is None

    factory, engine = session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    with factory() as session:
        dataset = import_dataset(session, path, "Portfolio demo", "synthetic")
        model = register_model(session, tmp_path / "artifacts" / f"{report['version']}.json")
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
        assert overview.json()["scored_customers"] == 600
        assert overview.json()["model"]["id"] == report["version"]
        assert overview.json()["source_kind"] == "synthetic"
        distribution = client.get("/api/v1/analytics/risk-distribution", headers=headers).json()
        assert sum(item["count"] for item in distribution) == 600
        customer = client.get("/api/v1/customers?size=1", headers=headers).json()["items"][0]
        assert customer["prediction"]["model_id"] == report["version"]
        band = customer["prediction"]["risk_band"]
        filtered = client.get(f"/api/v1/customers?risk={band}&size=10", headers=headers).json()
        assert all(item["prediction"]["risk_band"] == band for item in filtered["items"])
        assert filtered["total"] >= len(filtered["items"])
        assert client.get(f"/api/v1/customers/{customer['id']}", headers=headers).status_code == 200
    assert json.loads((tmp_path / "artifacts" / f"{report['version']}.json").read_text())["dataset_rows"] == 600
