from __future__ import annotations

import os
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Organization(Base):
    __tablename__ = "organizations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(120), unique=True)


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    definition_key: Mapped[str] = mapped_column(String(120))
    sha256: Mapped[str] = mapped_column(String(64))
    rows: Mapped[int] = mapped_column(Integer)
    source_kind: Mapped[str] = mapped_column(String(20))
    # NULL on rows migrated from an older schema: lineage is unverified, never back-filled.
    contract_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    quality_status: Mapped[str | None] = mapped_column(String(30), nullable=True)
    quality_report_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    quality_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (UniqueConstraint("organization_id", "sha256"),)


class Customer(Base):
    __tablename__ = "customers"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    external_id: Mapped[str] = mapped_column(String(120))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    attributes: Mapped[dict] = mapped_column(JSON)
    observed_exit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    predictions: Mapped[list[Prediction]] = relationship(back_populates="customer")
    __table_args__ = (UniqueConstraint("organization_id", "external_id"),)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    definition_key: Mapped[str] = mapped_column(String(120))
    artifact_path: Mapped[str] = mapped_column(String(500))
    artifact_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lineage: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Full held-out evaluation, split summary, calibration, subgroups; NULL when never recorded.
    evaluation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20), default="candidate")


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    probability: Mapped[float] = mapped_column(Float)
    risk_band: Mapped[str] = mapped_column(String(16))
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # "retrospective_batch" | "retrospective_on_demand"; NULL = legacy row of unknown origin.
    score_kind: Mapped[str | None] = mapped_column(String(30), nullable=True)
    customer: Mapped[Customer] = relationship(back_populates="predictions")


class SplitAssignment(Base):
    """Partition of one customer for one training run. Rows are written once, at registration."""

    __tablename__ = "split_assignments"
    model_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), primary_key=True)
    partition: Mapped[str] = mapped_column(String(20))
    __table_args__ = (Index("ix_split_assignments_model_partition", "model_id", "partition"),)


class EvaluationPrediction(Base):
    """Out-of-sample probabilities produced during training, kept apart from `predictions`."""

    __tablename__ = "evaluation_predictions"
    model_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), primary_key=True)
    partition: Mapped[str] = mapped_column(String(20))
    label: Mapped[int] = mapped_column(Integer)
    raw_probability: Mapped[float] = mapped_column(Float)
    probability: Mapped[float] = mapped_column(Float)


def session_factory(database_url: str | None = None):
    url = database_url or os.getenv("DATABASE_URL", "sqlite:///./churn.db")
    engine = create_engine(url, pool_pre_ping=True)
    return sessionmaker(bind=engine, expire_on_commit=False), engine

