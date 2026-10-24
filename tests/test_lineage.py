from __future__ import annotations

import json
import shutil

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from churn import api
from churn.config import BANKING
from churn.db import Base, Customer, Dataset, session_factory
from churn.demo import make_demo_data
from churn.ml.contract import IGNORED_COLUMNS
from churn.ml.data import validate_csv
from churn.ml.lineage import file_sha256, id_fingerprint
from churn.ml.train import train
from churn.service import import_dataset, register_model, score_customer, score_customers


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("lineage")
    csv_path = make_demo_data(root / "demo.csv")
    report = train(csv_path, root / "artifacts", "synthetic")
    return csv_path, root / "artifacts", report


@pytest.fixture
def workspace(trained, tmp_path):
    csv_path, artifacts, report = trained
    copy = tmp_path / "artifacts"
    shutil.copytree(artifacts, copy)
    factory, engine = session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    return csv_path, copy / f"{report['version']}.json", report, factory, engine


def edit_report(path, mutate):
    report = json.loads(path.read_text())
    mutate(report)
    path.write_text(json.dumps(report))


def test_training_report_traces_to_validated_dataset(trained):
    csv_path, artifacts, report = trained
    validated = validate_csv(csv_path, "synthetic")
    assert report["dataset"] == {
        "sha256": validated.sha256, "rows": 600, "source_kind": "synthetic",
        "contract_version": validated.contract_version, "quality_status": "passed",
        "quality_report_sha256": validated.report_sha256}
    assert report["version"].endswith(validated.sha256[:12])
    assert report["artifact_sha256"] == file_sha256(artifacts / f"{report['version']}.joblib")
    assert report["split"]["rows"] == {"train": 360, "validation": 120, "test": 120}
    assert any("SYNTHETIC" in item for item in report["limitations"])


def test_split_is_disjoint_complete_and_reproducible(trained, tmp_path):
    csv_path, _, report = trained
    fingerprints = report["split"]["customer_id_sha256"]
    assert len(set(fingerprints.values())) == 3
    again = train(csv_path, tmp_path / "again", "synthetic")
    assert again["split"] == report["split"]
    assert again["held_out_test"] == report["held_out_test"]
    ids = validate_csv(csv_path, "synthetic").frame[BANKING.customer_id_column]
    assert id_fingerprint(ids) not in fingerprints.values()


def test_features_exclude_identifiers_target_and_ignored_columns():
    features = set(BANKING.feature_columns)
    assert BANKING.customer_id_column not in features
    assert BANKING.target_column not in features
    assert not features & set(IGNORED_COLUMNS)


def test_registration_stores_lineage_and_scoring_works(workspace):
    csv_path, report_path, report, factory, _ = workspace
    with factory() as session:
        dataset = import_dataset(session, csv_path, "Org A", "synthetic")
        assert dataset.contract_version == report["dataset"]["contract_version"]
        assert dataset.quality_report_sha256 == report["dataset"]["quality_report_sha256"]
        assert dataset.quality_report["status"] == "passed"
        model = register_model(session, report_path, "Org A")
        assert model.dataset_id == dataset.id
        assert model.lineage["dataset"] == report["dataset"]
        assert model.artifact_sha256 == report["artifact_sha256"]
        customers = session.scalars(select(Customer)).all()
        assert score_customers(session, list(customers), model) == 600


def test_registration_rejects_each_lineage_mismatch(workspace):
    csv_path, report_path, _, factory, _ = workspace
    with factory() as session:
        import_dataset(session, csv_path, "Org A", "synthetic")
    original = report_path.read_text()
    cases = {
        "source_kind": lambda r: r["dataset"].update(source_kind="external"),
        "contract_version": lambda r: r["dataset"].update(contract_version="banking-csv-contract/9.9.9"),
        "quality_report_sha256": lambda r: r["dataset"].update(quality_report_sha256="0" * 64),
        "rows": lambda r: r["dataset"].update(rows=601),
        "definition": lambda r: r["definition"].update(key="retail-v1"),
    }
    for name, mutate in cases.items():
        report_path.write_text(original)
        edit_report(report_path, mutate)
        with factory() as session, pytest.raises(ValueError, match=name):
            register_model(session, report_path, "Org A")
    report_path.write_text(original)
    edit_report(report_path, lambda r: r["dataset"].update(sha256="f" * 64))
    with factory() as session, pytest.raises(ValueError, match="Import this exact dataset"):
        register_model(session, report_path, "Org A")


def test_registration_is_scoped_to_the_organization(workspace):
    csv_path, report_path, _, factory, _ = workspace
    with factory() as session:
        import_dataset(session, csv_path, "Org A", "synthetic")
    with factory() as session, pytest.raises(ValueError, match="Import this exact dataset"):
        register_model(session, report_path, "Org B")


def test_old_report_schema_is_rejected(workspace):
    _, report_path, _, factory, _ = workspace
    edit_report(report_path, lambda r: r.pop("report_schema_version"))
    with factory() as session, pytest.raises(ValueError, match="predates dataset lineage"):
        register_model(session, report_path, "Org A")


def test_tampered_artifact_is_never_registered_or_loaded(workspace):
    csv_path, report_path, _, factory, _ = workspace
    with factory() as session:
        import_dataset(session, csv_path, "Org A", "synthetic")
    artifact = report_path.with_suffix(".joblib")
    good = artifact.read_bytes()
    artifact.write_bytes(good + b"x")
    with factory() as session, pytest.raises(ValueError, match="does not match the hash"):
        register_model(session, report_path, "Org A")
    artifact.write_bytes(good)
    with factory() as session:
        model = register_model(session, report_path, "Org A")
        customer = session.scalars(select(Customer)).first()
        artifact.write_bytes(good + b"x")  # changed after registration
        with pytest.raises(ValueError, match="changed after registration"):
            score_customer(session, customer, model)


def test_same_file_cannot_be_relabelled_on_reimport(workspace):
    csv_path, _, _, factory, _ = workspace
    with factory() as session:
        first = import_dataset(session, csv_path, "Org A", "synthetic")
    with factory() as session:
        assert import_dataset(session, csv_path, "Org A", "synthetic").id == first.id
    with factory() as session, pytest.raises(ValueError, match="already imported as synthetic"):
        import_dataset(session, csv_path, "Org A", "external")
    with factory() as session:
        assert len(session.scalars(select(Dataset)).all()) == 1


def test_invalid_file_imports_nothing(workspace, tmp_path):
    csv_path, _, _, factory, _ = workspace
    broken = tmp_path / "broken.csv"
    lines = csv_path.read_text().splitlines()
    lines[5] = lines[4]
    broken.write_text("\n".join(lines) + "\n")
    with factory() as session, pytest.raises(ValueError, match="Dataset failed"):
        import_dataset(session, broken, "Org A", "synthetic")
    with factory() as session:
        assert session.scalars(select(Customer)).first() is None
        assert session.scalars(select(Dataset)).first() is None


def test_overlapping_customer_ids_are_rejected_atomically(workspace, tmp_path):
    csv_path, _, _, factory, _ = workspace
    subset = tmp_path / "subset.csv"  # different bytes, same customer identifiers
    subset.write_text("\n".join(csv_path.read_text().splitlines()[:301]) + "\n")
    with factory() as session:
        import_dataset(session, csv_path, "Org A", "synthetic")
    with factory() as session, pytest.raises(ValueError, match="already exists"):
        import_dataset(session, subset, "Org A", "synthetic")
    with factory() as session:
        assert len(session.scalars(select(Dataset)).all()) == 1


def test_api_exposes_dataset_and_model_lineage(workspace, monkeypatch):
    csv_path, report_path, report, factory, engine = workspace
    with factory() as session:
        import_dataset(session, csv_path, "Portfolio demo", "synthetic")
        register_model(session, report_path, "Portfolio demo")
    monkeypatch.setattr(api, "factory", factory)
    monkeypatch.setattr(api, "engine", engine)
    monkeypatch.setenv("API_KEY", "secret")
    headers = {"X-API-Key": "secret"}
    with TestClient(api.app) as client:
        dataset = client.get("/api/v1/datasets", headers=headers).json()[0]
        assert dataset["source_kind"] == "synthetic"
        assert dataset["quality_status"] == "passed"
        assert dataset["quality_report_sha256"] == report["dataset"]["quality_report_sha256"]
        model = client.get("/api/v1/models", headers=headers).json()[0]
        assert model["lineage"]["dataset"]["sha256"] == dataset["sha256"]
        assert model["dataset_id"] == dataset["id"]
