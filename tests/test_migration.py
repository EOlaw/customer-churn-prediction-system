from __future__ import annotations

import sqlite3
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text

from churn import api
from churn.db import session_factory
from churn.migrate import (
    LATEST,
    current_version,
    init_schema,
    migrate,
    pending,
    schema_status,
)
from tests.legacy_schema import V0_DDL, V1_ALTERS

ORG_ID, DATASET_ID, MODEL_ID = "org-1", "ds-1", "legacy-model"


def make_legacy(path, version=0, customers=5):
    connection = sqlite3.connect(path)
    connection.executescript(V0_DDL)
    if version >= 1:
        for statement in V1_ALTERS:
            connection.execute(statement)
    connection.execute("INSERT INTO organizations VALUES (?, 'Portfolio demo')", (ORG_ID,))
    connection.execute("INSERT INTO datasets (id, organization_id, definition_key, sha256, rows, "
                       "source_kind, imported_at) VALUES (?, ?, 'banking-account-exit-v1', ?, ?, "
                       "'synthetic', '2026-10-24 03:00:00')", (DATASET_ID, ORG_ID, "c" * 64, customers))
    connection.execute("INSERT INTO model_versions (id, dataset_id, definition_key, artifact_path, "
                       "metrics, trained_at, status) VALUES (?, ?, 'banking-account-exit-v1', "
                       "'/nonexistent/legacy.joblib', '{\"roc_auc\": 0.7}', '2026-10-24 03:00:00', "
                       "'candidate')", (MODEL_ID, DATASET_ID))
    for i in range(customers):
        connection.execute("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?)",
                           (f"c{i}", ORG_ID, f"DEMO-{i}", DATASET_ID,
                            '{"Geography": "France", "Tenure": 3}', i % 2))
        connection.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, 'low', '2026-10-24 04:00:00')",
                           (f"p{i}", f"c{i}", MODEL_ID, 0.1 * i))
    connection.commit()
    connection.close()
    return path


def snapshot(path):
    connection = sqlite3.connect(path)
    try:
        return {table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                for table in ("organizations", "datasets", "customers", "model_versions", "predictions")}
    finally:
        connection.close()


def engine_for(path):
    return create_engine(f"sqlite:///{path}")


def test_v0_database_is_detected_and_reported_outdated(tmp_path):
    path = make_legacy(tmp_path / "old.db")
    engine = engine_for(path)
    assert current_version(engine) == 0 and schema_status(engine) == ("outdated", 0)
    assert [m.version for m in pending(engine)] == [1, 2]
    make_legacy(tmp_path / "mid.db", version=1)
    assert current_version(engine_for(tmp_path / "mid.db")) == 1


def test_migration_backs_up_then_preserves_every_existing_row(tmp_path):
    path = make_legacy(tmp_path / "old.db")
    before = snapshot(path)
    applied, backup = migrate(engine_for(path))
    assert [m.version for m in applied] == [1, 2]
    assert backup is not None and backup.exists() and ".pre-migration-v0-" in backup.name
    assert snapshot(backup) == before  # the backup is the untouched original
    after = snapshot(path)
    assert after["organizations"] == before["organizations"]
    assert after["customers"] == before["customers"]
    # predictions only gain the NULL score_kind column
    assert [row[:-1] for row in after["predictions"]] == before["predictions"]
    assert {row[-1] for row in after["predictions"]} == {None}
    assert after["datasets"][0][:7] == before["datasets"][0]  # original columns identical
    assert after["model_versions"][0][:7] == before["model_versions"][0]
    assert current_version(engine_for(path)) == LATEST


def test_legacy_lineage_is_not_invented(tmp_path):
    path = make_legacy(tmp_path / "old.db")
    migrate(engine_for(path))
    connection = sqlite3.connect(path)
    dataset = connection.execute("SELECT contract_version, quality_status, quality_report_sha256, "
                                 "quality_report FROM datasets").fetchone()
    model = connection.execute("SELECT artifact_sha256, lineage, evaluation FROM model_versions").fetchone()
    scores = connection.execute("SELECT DISTINCT score_kind FROM predictions").fetchall()
    assert dataset == (None, None, None, None)
    assert model == (None, None, None)
    assert scores == [(None,)]
    assert connection.execute("SELECT COUNT(*) FROM split_assignments").fetchone() == (0,)
    assert connection.execute("SELECT COUNT(*) FROM evaluation_predictions").fetchone() == (0,)


def test_phase2_database_keeps_its_lineage_values(tmp_path):
    path = make_legacy(tmp_path / "mid.db", version=1)
    applied, backup = migrate(engine_for(path))
    assert [m.version for m in applied] == [2] and ".pre-migration-v1-" in backup.name
    connection = sqlite3.connect(path)
    assert connection.execute("SELECT contract_version, artifact_sha256 FROM datasets, model_versions"
                              ).fetchone() == ("banking-csv-contract/1.0.0", "b")
    assert connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone() == (2,)


def test_migration_is_idempotent_and_stamps_history(tmp_path):
    path = make_legacy(tmp_path / "old.db")
    engine = engine_for(path)
    migrate(engine)
    applied, backup = migrate(engine)
    assert applied == [] and backup is None
    with engine.connect() as connection:
        rows = connection.execute(text("SELECT version, note FROM schema_migrations ORDER BY version")).all()
    assert [r[0] for r in rows] == [1, 2]
    assert all("migrated from v0" in r[1] for r in rows)
    assert len(list(tmp_path.glob("*.bak"))) == 1  # no second backup for a no-op


def test_migrated_schema_matches_a_fresh_one(tmp_path):
    old = make_legacy(tmp_path / "old.db")
    migrate(engine_for(old))
    _, fresh_engine = session_factory(f"sqlite:///{tmp_path / 'fresh.db'}")
    init_schema(fresh_engine)
    migrated = inspect(engine_for(old))
    fresh = inspect(fresh_engine)
    assert set(migrated.get_table_names()) == set(fresh.get_table_names())
    for table in fresh.get_table_names():
        assert {c["name"] for c in migrated.get_columns(table)} == {
            c["name"] for c in fresh.get_columns(table)}, table


def test_failed_backup_leaves_database_untouched(tmp_path, monkeypatch):
    path = make_legacy(tmp_path / "old.db")
    before = snapshot(path)
    import churn.migrate as mig

    def boom(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(mig, "backup_sqlite", boom)
    with pytest.raises(OSError):
        migrate(engine_for(path))
    assert snapshot(path) == before
    assert current_version(engine_for(path)) == 0
    assert "schema_migrations" not in inspect(engine_for(path)).get_table_names()


def test_init_schema_refuses_outdated_and_stamps_fresh(tmp_path):
    engine = engine_for(make_legacy(tmp_path / "old.db"))
    with pytest.raises(RuntimeError, match="migrate"):
        init_schema(engine)
    fresh = engine_for(tmp_path / "fresh.db")
    assert schema_status(fresh) == ("empty", None)
    init_schema(fresh)
    init_schema(fresh)  # no-op
    assert schema_status(fresh) == ("current", LATEST)


def test_api_refuses_to_start_on_an_outdated_database(tmp_path, monkeypatch):
    path = make_legacy(tmp_path / "old.db")
    factory, engine = session_factory(f"sqlite:///{path}")
    monkeypatch.setattr(api, "factory", factory)
    monkeypatch.setattr(api, "engine", engine)
    monkeypatch.setenv("API_KEY", "secret")
    with pytest.raises(RuntimeError, match="python -m churn.cli migrate"), TestClient(api.app):
        pass


def test_api_reports_migrated_legacy_rows_as_unverified(tmp_path, monkeypatch):
    path = make_legacy(tmp_path / "old.db")
    migrate(engine_for(path))
    factory, engine = session_factory(f"sqlite:///{path}")
    monkeypatch.setattr(api, "factory", factory)
    monkeypatch.setattr(api, "engine", engine)
    monkeypatch.setenv("API_KEY", "secret")
    headers = {"X-API-Key": "secret"}
    with TestClient(api.app) as client:
        dataset = client.get("/api/v1/datasets", headers=headers).json()[0]
        assert dataset["lineage_status"] == "unverified"
        assert dataset["contract_version"] is None and dataset["quality_issues"] is None
        model = client.get("/api/v1/models", headers=headers).json()[0]
        assert model["lineage_status"] == "unverified" and model["evaluation_recorded"] is False
        evaluation = client.get(f"/api/v1/models/{MODEL_ID}/evaluation", headers=headers).json()
        assert evaluation["evaluation"] is None and "unverified" in evaluation["reason"]

        overview = client.get("/api/v1/analytics/overview", headers=headers).json()
        assert overview["model"]["lineage_status"] == "unverified"
        assert overview["model"]["held_out"] is None  # nothing is invented
        assert overview["partitions"]["unknown"] == {
            "customers": 5, "scored": 5, "mean_score": pytest.approx(0.2), "high_risk": 0,
            "partition": "unknown", "in_sample": None}
        assert overview["partitions"]["test"]["scored"] == 0
        assert overview["score_kinds"] == {"legacy_unknown": 5}

        risk = client.get("/api/v1/analytics/risk-distribution", headers=headers).json()
        assert sum(i["count"] for i in risk["items"]) == 0  # held-out default: no recorded test split
        unknown = client.get("/api/v1/analytics/risk-distribution?partition=unknown", headers=headers).json()
        assert sum(i["count"] for i in unknown["items"]) == 5 and unknown["in_sample"] is None
        segments = client.get("/api/v1/analytics/segments", headers=headers).json()
        assert segments[0]["heldout_customers"] == 0 and segments[0]["heldout_mean_probability"] is None

        customer = client.get("/api/v1/customers?size=1", headers=headers).json()["items"][0]
        assert customer["prediction"]["partition"] == "unknown"
        assert customer["prediction"]["in_sample"] is None
        assert customer["prediction"]["score_kind"] == "legacy_unknown"
        refused = client.post("/api/v1/predictions", json={"customer_id": customer["id"]}, headers=headers)
        assert refused.status_code == 409 and "unverified" in refused.json()["detail"]


def test_cli_migrate_check_and_apply(tmp_path):
    path = make_legacy(tmp_path / "old.db")
    env = {"DATABASE_URL": f"sqlite:///{path}", "PATH": "/usr/bin:/bin"}

    def run(*args):
        return subprocess.run([sys.executable, "-m", "churn.cli", *args], capture_output=True,
                              text=True, env=env, cwd=".", check=False)
    check = run("migrate", "--check")
    assert check.returncode == 1 and "pending" in check.stdout
    done = run("migrate")
    assert done.returncode == 0, done.stderr
    assert "backup written" in done.stdout and "unverified" in done.stdout
    assert run("migrate", "--check").returncode == 0
