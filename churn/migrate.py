"""Explicit, ordered schema migrations with a backup step for file-based SQLite databases.

Rules: migrations only add columns and tables; they never rewrite or delete existing rows and
never back-fill lineage fields that were not recorded at the time (those stay NULL and the API
reports the row as ``unverified``). A database that predates this module is recognised from
its columns, so no manual stamping is needed.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import JSON, Column, DateTime, Integer, MetaData, String, Table, inspect, text
from sqlalchemy.engine import Connection, Engine

from churn.db import Base


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    apply: Callable[[Connection], None]


def _columns(connection: Connection, table: str) -> set[str]:
    return {column["name"] for column in inspect(connection).get_columns(table)}


def _add_columns(connection: Connection, table: str, columns: dict[str, object]) -> None:
    existing = _columns(connection, table)
    for name, type_ in columns.items():
        if name not in existing:
            ddl = type_.compile(dialect=connection.dialect)  # type: ignore[attr-defined]
            connection.execute(text(f'ALTER TABLE {table} ADD COLUMN {name} {ddl}'))


def _v1_phase2_lineage(connection: Connection) -> None:
    _add_columns(connection, "datasets", {
        "contract_version": String(80), "quality_status": String(30),
        "quality_report_sha256": String(64), "quality_report": JSON()})
    _add_columns(connection, "model_versions", {"artifact_sha256": String(64), "lineage": JSON()})


def _v2_phase3_evaluation(connection: Connection) -> None:
    _add_columns(connection, "model_versions", {"evaluation": JSON()})
    _add_columns(connection, "predictions", {"score_kind": String(30)})
    for name in ("split_assignments", "evaluation_predictions"):
        Base.metadata.tables[name].create(connection, checkfirst=True)


MIGRATIONS = (
    Migration(1, "phase2 dataset and model lineage columns", _v1_phase2_lineage),
    Migration(2, "phase3 evaluation, split assignments, score kind", _v2_phase3_evaluation),
)
LATEST = MIGRATIONS[-1].version
_registry = MetaData()
_versions = Table("schema_migrations", _registry,
                  Column("version", Integer, primary_key=True), Column("name", String(200)),
                  Column("applied_at", DateTime(timezone=True)), Column("note", String(500)))


def current_version(engine: Engine) -> int | None:
    """None for an empty database; 0 for an unversioned legacy schema."""
    tables = set(inspect(engine).get_table_names())
    if "schema_migrations" in tables:
        with engine.connect() as connection:
            return connection.execute(text("SELECT MAX(version) FROM schema_migrations")).scalar() or 0
    if not tables:
        return None
    with engine.connect() as connection:
        return 1 if "contract_version" in _columns(connection, "datasets") else 0


def schema_status(engine: Engine) -> tuple[str, int | None]:
    version = current_version(engine)
    if version is None:
        return "empty", None
    return ("current" if version == LATEST else "outdated"), version


def _stamp(connection: Connection, migration: Migration, note: str) -> None:
    connection.execute(_versions.insert().values(
        version=migration.version, name=migration.name, applied_at=datetime.now(UTC), note=note))


def init_schema(engine: Engine) -> None:
    """Create the latest schema in an empty database; refuse to touch an existing one."""
    status, version = schema_status(engine)
    if status == "outdated":
        raise RuntimeError(f"Database is at schema version {version}, latest is {LATEST}. "
                           "Run `python -m churn.cli migrate` (a backup is made first for SQLite).")
    if status == "current":
        return
    Base.metadata.create_all(engine)
    _versions.create(engine, checkfirst=True)
    with engine.begin() as connection:
        for migration in MIGRATIONS:
            _stamp(connection, migration, "created directly at latest schema")


def backup_sqlite(engine: Engine, from_version: int) -> Path | None:
    if engine.dialect.name != "sqlite" or not engine.url.database or engine.url.database == ":memory:":
        return None
    source = Path(engine.url.database)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = source.with_name(f"{source.name}.pre-migration-v{from_version}-{stamp}.bak")
    if target.exists():
        raise FileExistsError(target)
    with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
        src.backup(dst)
    return target


def pending(engine: Engine) -> list[Migration]:
    version = current_version(engine)
    if version is None:
        return []
    return [m for m in MIGRATIONS if m.version > version]


def migrate(engine: Engine) -> tuple[list[Migration], Path | None]:
    """Apply pending migrations. Returns (applied, backup path or None)."""
    todo = pending(engine)
    if not todo:
        return [], None
    start = current_version(engine) or 0
    backup = backup_sqlite(engine, start)
    _versions.create(engine, checkfirst=True)
    with engine.begin() as connection:
        if connection.execute(text("SELECT COUNT(*) FROM schema_migrations")).scalar() == 0:
            for baseline in (m for m in MIGRATIONS if m.version <= start):
                _stamp(connection, baseline, "baseline inferred from existing columns")
    for migration in todo:
        with engine.begin() as connection:
            migration.apply(connection)
            _stamp(connection, migration, f"migrated from v{start}")
    return todo, backup
