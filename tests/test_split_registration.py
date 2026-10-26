"""Registration of split assignments and evaluation predictions, and the corrected analytics."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from churn import api
from churn.db import (
    Customer,
    EvaluationPrediction,
    ModelVersion,
    Prediction,
    SplitAssignment,
    session_factory,
)
from churn.migrate import init_schema
from churn.ml import metrics as m
from churn.ml.lineage import file_sha256, id_fingerprint
from churn.service import import_dataset, register_model, score_customer, score_customers

ORG = "Portfolio demo"
HEADERS = {"X-API-Key": "secret"}


@pytest.fixture
def env(trained, tmp_path):
    csv_path, artifacts, report = trained
    copy = tmp_path / "artifacts"
    shutil.copytree(artifacts, copy)
    factory, engine = session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    init_schema(engine)
    return csv_path, copy, report, factory, engine


@pytest.fixture
def registered(env):
    csv_path, copy, report, factory, _ = env
    with factory() as session:
        import_dataset(session, csv_path, ORG, "synthetic")
        register_model(session, copy / f"{report['version']}.json", ORG)
    return env


@pytest.fixture
def client(registered, monkeypatch):
    *_, factory, engine = registered
    monkeypatch.setattr(api, "factory", factory)
    monkeypatch.setattr(api, "engine", engine)
    monkeypatch.setenv("API_KEY", "secret")
    with TestClient(api.app) as test_client:
        yield test_client


def score_all(registered):
    _, _, report, factory, _ = registered
    with factory() as session:
        model = session.get(ModelVersion, report["version"])
        customers = session.scalars(select(Customer)).all()
        return score_customers(session, list(customers), model)


# ---- what registration stores ---------------------------------------------------------------

def test_split_assignments_are_stored_per_customer_and_run(registered):
    _, _, report, factory, _ = registered
    with factory() as session:
        rows = session.execute(select(Customer.external_id, SplitAssignment.partition).join(
            SplitAssignment, SplitAssignment.customer_id == Customer.id).where(
            SplitAssignment.model_id == report["version"])).all()
        assert len(rows) == 600
        for name, digest in report["split"]["customer_id_sha256"].items():
            assert id_fingerprint(e for e, p in rows if p == name) == digest
        counts = dict(session.execute(select(SplitAssignment.partition, func.count()).group_by(
            SplitAssignment.partition)).all())
        assert counts == {"train": 360, "calibration": 120, "test": 120}


def test_evaluation_predictions_are_separate_from_operational_scores(registered):
    _, _, _, factory, _ = registered
    with factory() as session:
        evaluation = session.scalars(select(EvaluationPrediction)).all()
        assert len(evaluation) == 240 and {e.partition for e in evaluation} == {"calibration", "test"}
        assert session.scalar(select(func.count()).select_from(Prediction)) == 0  # nothing scored yet
    assert score_all(registered) == 600
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(Prediction)) == 600
        assert session.scalar(select(func.count()).select_from(EvaluationPrediction)) == 240
        assert {p.score_kind for p in session.scalars(select(Prediction)).all()} == {"retrospective_batch"}


def test_heldout_metrics_recompute_from_the_database(registered):
    _, _, report, factory, _ = registered
    with factory() as session:
        rows = session.scalars(select(EvaluationPrediction).where(
            EvaluationPrediction.partition == "test")).all()
    y = np.array([r.label for r in rows])
    p = np.array([r.probability for r in rows])
    assert report["held_out_test"]["roc_auc"] == pytest.approx(m.auc_score(y, p))
    at = m.threshold_metrics(y, p, report["threshold"]["value"])
    assert report["held_out_test"]["at_threshold"]["tp"] == at["tp"]
    with factory() as session:
        stored = session.get(ModelVersion, report["version"])
        assert stored.metrics["roc_auc"] == pytest.approx(report["held_out_test"]["roc_auc"])
        assert stored.evaluation["split"]["rows"] == report["split"]["rows"]
        assert stored.lineage["split"]["customer_id_sha256"] == report["split"]["customer_id_sha256"]


def test_model_labels_match_customer_labels(registered):
    _, _, _, factory, _ = registered
    with factory() as session:
        pairs = session.execute(select(EvaluationPrediction.label, Customer.observed_exit).join(
            Customer, Customer.id == EvaluationPrediction.customer_id)).all()
        assert pairs and all(label == exited for label, exited in pairs)


# ---- registration refuses inconsistent splits ----------------------------------------------

def rewrite(path, mutate_csv=None, report_path=None, field=None):
    frame = pd.read_csv(path, dtype={"CustomerId": str})
    mutate_csv(frame)
    frame.to_csv(path, index=False, lineterminator="\n")
    if report_path is not None:
        report = json.loads(report_path.read_text())
        field(report, file_sha256(path))
        report_path.write_text(json.dumps(report))


def test_registration_rejects_tampered_split_or_evaluation_files(env):
    csv_path, copy, report, factory, _ = env
    with factory() as session:
        import_dataset(session, csv_path, ORG, "synthetic")
    report_path = copy / f"{report['version']}.json"
    splits, evaluation = copy / report["split"]["assignments_file"], copy / report["evaluation_file"]
    original_splits, original_evaluation = splits.read_bytes(), evaluation.read_bytes()

    splits.write_bytes(original_splits + b"x")
    with factory() as session, pytest.raises(ValueError, match="does not match the hash"):
        register_model(session, report_path, ORG)
    splits.write_bytes(original_splits)

    evaluation.write_bytes(original_evaluation + b"x")
    with factory() as session, pytest.raises(ValueError, match="does not match the hash"):
        register_model(session, report_path, ORG)
    evaluation.write_bytes(original_evaluation)

    # a consistent-looking rewrite (hash updated) still fails the partition fingerprint check
    def move_one(frame):
        i = frame.index[frame.partition == "test"][0]
        frame.loc[i, "partition"] = "train"
    rewrite(splits, move_one, report_path, lambda r, h: r["split"].update(assignments_sha256=h))
    with factory() as session, pytest.raises(ValueError, match="do not match the training report"):
        register_model(session, report_path, ORG)
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(SplitAssignment)) == 0  # atomic


def test_registration_rejects_evaluation_rows_that_contradict_the_split(env):
    csv_path, copy, report, factory, _ = env
    with factory() as session:
        import_dataset(session, csv_path, ORG, "synthetic")
    report_path, evaluation = copy / f"{report['version']}.json", copy / report["evaluation_file"]

    def relabel(frame):
        frame.loc[frame.index[frame.partition == "test"][0], "partition"] = "calibration"
    rewrite(evaluation, relabel, report_path, lambda r, h: r.update(evaluation_sha256=h))
    with factory() as session, pytest.raises(ValueError, match="disagree with the split"):
        register_model(session, report_path, ORG)

    def train_row(frame):
        frame.loc[0, "partition"] = "train"
    rewrite(evaluation, train_row, report_path, lambda r, h: r.update(evaluation_sha256=h))
    with factory() as session, pytest.raises(ValueError):
        register_model(session, report_path, ORG)


def test_registration_requires_the_exact_dataset_version(env, tmp_path):
    csv_path, copy, report, factory, _ = env
    subset = tmp_path / "subset.csv"  # same ids but fewer rows: a different dataset version
    subset.write_text("\n".join(csv_path.read_text().splitlines()[:301]) + "\n")
    with factory() as session:
        import_dataset(session, subset, ORG, "synthetic")
    with factory() as session, pytest.raises(ValueError, match="Import this exact dataset"):
        register_model(session, copy / f"{report['version']}.json", ORG)  # hash differs


# ---- scoring semantics ---------------------------------------------------------------------

def test_scoring_requires_verified_lineage(registered):
    _, _, report, factory, _ = registered
    with factory() as session:
        model = session.get(ModelVersion, report["version"])
        model.artifact_sha256 = None  # as in a migrated legacy row
        customer = session.scalars(select(Customer)).first()
        with pytest.raises(ValueError, match="unverified lineage"):
            score_customer(session, customer, model)


def test_on_demand_scores_are_labelled(registered):
    _, _, report, factory, _ = registered
    with factory() as session:
        model = session.get(ModelVersion, report["version"])
        customer = session.scalars(select(Customer)).first()
        assert score_customer(session, customer, model).score_kind == "retrospective_on_demand"


# ---- corrected analytics -------------------------------------------------------------------

def test_overview_separates_partitions_and_has_no_pooled_score(client, registered):
    score_all(registered)
    body = client.get("/api/v1/analytics/overview", headers=HEADERS).json()
    assert "mean_model_score" not in body and "high_risk" not in body and "scored_customers" not in body
    parts = body["partitions"]
    assert {n: parts[n]["customers"] for n in ("train", "calibration", "test", "unknown")} == {
        "train": 360, "calibration": 120, "test": 120, "unknown": 0}
    assert {n: parts[n]["scored"] for n in ("train", "calibration", "test")} == {
        "train": 360, "calibration": 120, "test": 120}
    assert parts["test"]["in_sample"] is False
    assert parts["train"]["in_sample"] is True and parts["calibration"]["in_sample"] is True
    assert body["score_kinds"] == {"retrospective_batch": 600}
    assert body["operational_scoring"]["available"] is False
    assert any("no prediction horizon" in item for item in body["limitations"])
    assert body["source_kind"] == "synthetic" and body["observed_churn_rate"] == pytest.approx(
        80 / 600)  # label-based, valid on all customers
    model = body["model"]
    assert model["lineage_status"] == "verified" and model["evidence_status"].startswith("SYNTHETIC")
    held = model["held_out"]
    assert held["rows"] == 120 and held["confidence_intervals_95"]["roc_auc"]
    assert held["roc_auc"] == pytest.approx(registered[2]["held_out_test"]["roc_auc"])


def test_partition_means_use_only_that_partitions_scores(client, registered):
    score_all(registered)
    _, _, _, factory, _ = registered
    body = client.get("/api/v1/analytics/overview", headers=HEADERS).json()
    with factory() as session:
        rows = session.execute(select(SplitAssignment.partition, Prediction.probability).join(
            Prediction, (Prediction.customer_id == SplitAssignment.customer_id)
            & (Prediction.model_id == SplitAssignment.model_id))).all()
    for name in ("train", "calibration", "test"):
        values = [p for part, p in rows if part == name]
        assert body["partitions"][name]["mean_score"] == pytest.approx(np.mean(values))


def test_overview_before_scoring_shows_unscored_not_zero_scores(client):
    body = client.get("/api/v1/analytics/overview", headers=HEADERS).json()
    assert all(v["scored"] == 0 and v["mean_score"] is None for v in body["partitions"].values())
    assert body["score_kinds"] == {} and body["latest_scoring_date"] is None
    assert body["model"]["held_out"] is not None  # held-out evaluation exists without batch scoring


def test_risk_distribution_defaults_to_heldout_and_labels_in_sample(client, registered):
    score_all(registered)
    default = client.get("/api/v1/analytics/risk-distribution", headers=HEADERS).json()
    assert default["partition"] == "test" and default["in_sample"] is False
    assert sum(i["count"] for i in default["items"]) == 120
    totals = {}
    for name in ("train", "calibration", "test", "unknown", "all"):
        body = client.get(f"/api/v1/analytics/risk-distribution?partition={name}", headers=HEADERS).json()
        totals[name] = sum(i["count"] for i in body["items"])
        assert body["in_sample"] is {"test": False, "all": None, "unknown": None}.get(name, True)
    assert totals == {"train": 360, "calibration": 120, "test": 120, "unknown": 0, "all": 600}
    assert client.get("/api/v1/analytics/risk-distribution?partition=bogus", headers=HEADERS).status_code == 422


def test_segments_average_heldout_scores_only(client, registered):
    score_all(registered)
    _, _, _, factory, _ = registered
    body = client.get("/api/v1/analytics/segments", headers=HEADERS).json()
    assert sum(s["customers"] for s in body) == 600
    assert sum(s["heldout_customers"] for s in body) == 120
    assert all("mean_probability" not in s for s in body)
    with factory() as session:
        rows = session.execute(select(Customer.attributes, Prediction.probability).join(
            Prediction, Prediction.customer_id == Customer.id).join(
            SplitAssignment, (SplitAssignment.customer_id == Customer.id)
            & (SplitAssignment.model_id == Prediction.model_id)).where(
            SplitAssignment.partition == "test")).all()
    germany = [p for attrs, p in rows if attrs["Geography"] == "Germany"]
    entry = next(s for s in body if s["segment"] == "Germany")
    assert entry["heldout_mean_probability"] == pytest.approx(np.mean(germany))


def test_customer_predictions_carry_partition_and_kind(client, registered):
    score_all(registered)
    items = client.get("/api/v1/customers?size=100", headers=HEADERS).json()["items"]
    seen = {i["prediction"]["partition"]: i["prediction"] for i in items}
    assert set(seen) <= {"train", "calibration", "test"}
    for item in items:
        prediction = item["prediction"]
        assert prediction["in_sample"] is (prediction["partition"] != "test")
        assert prediction["score_kind"] == "retrospective_batch"
    detail = client.get(f"/api/v1/customers/{items[0]['id']}", headers=HEADERS).json()
    assert detail["history"][0]["partition"] == items[0]["prediction"]["partition"]


def test_unscored_customers_have_no_prediction(client):
    items = client.get("/api/v1/customers", headers=HEADERS).json()["items"]
    assert all(i["prediction"] is None for i in items)


def test_predict_endpoint_labels_on_demand_scores(client, registered):
    customer = client.get("/api/v1/customers?size=1", headers=HEADERS).json()["items"][0]
    body = client.post("/api/v1/predictions", json={"customer_id": customer["id"]}, headers=HEADERS).json()
    assert body["score_kind"] == "retrospective_on_demand"
    assert body["partition"] in {"train", "calibration", "test"}
    assert body["in_sample"] is (body["partition"] != "test")


def test_models_and_evaluation_endpoints(client, registered):
    _, _, report, _, _ = registered
    models = client.get("/api/v1/models", headers=HEADERS).json()
    assert models[0]["lineage_status"] == "verified" and models[0]["evaluation_recorded"] is True
    body = client.get(f"/api/v1/models/{report['version']}/evaluation", headers=HEADERS).json()
    assert body["evaluation"]["held_out_test"]["roc_auc"] == pytest.approx(
        report["held_out_test"]["roc_auc"])
    assert body["evaluation"]["sensitivity_analyses"]["variants"].keys() == {
        "full", "no_demographics", "no_uncertain_availability"}
    assert body["evaluation"]["subgroups"]["groups"]["Geography"]
    assert client.get("/api/v1/models/nope/evaluation", headers=HEADERS).status_code == 404
    assert client.get(f"/api/v1/models/{report['version']}/evaluation").status_code == 401


def test_evaluation_is_scoped_to_the_organization(registered, monkeypatch):
    _, _, report, factory, engine = registered
    monkeypatch.setattr(api, "factory", factory)
    monkeypatch.setattr(api, "engine", engine)
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("ORGANIZATION_NAME", "Someone else")
    with factory() as session:
        from churn.db import Organization
        session.add(Organization(name="Someone else"))
        session.commit()
    with TestClient(api.app) as other:
        assert other.get(f"/api/v1/models/{report['version']}/evaluation", headers=HEADERS).status_code == 404
        overview = other.get("/api/v1/analytics/overview", headers=HEADERS).json()
        assert overview["customers"] == 0 and overview["model"] is None
