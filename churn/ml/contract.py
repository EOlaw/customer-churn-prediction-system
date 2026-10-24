"""Versioned CSV contract for the banking ``Exited`` use case.

Bump ``CONTRACT_VERSION`` whenever a rule below changes: the version is stored with every
dataset, quality report, and training report so results stay traceable to the rules that
admitted the data.
"""

from __future__ import annotations

from dataclasses import dataclass

CONTRACT_VERSION = "banking-csv-contract/1.0.0"
SOURCE_KINDS = ("external", "synthetic")
MAX_FILE_BYTES = 50 * 1024 * 1024
MIN_ROWS = 50
MIN_CLASS_EXAMPLES = 10
MAX_ID_LENGTH = 120  # matches customers.external_id


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: str  # "id" | "integer" | "float" | "category"
    minimum: float | None = None
    maximum: float | None = None
    categories: tuple[str, ...] = ()
    description: str = ""


COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("CustomerId", "id", description="Unique customer identifier; never a feature"),
    ColumnSpec("CreditScore", "integer", 300, 850, description="Credit score"),
    ColumnSpec("Geography", "category", categories=("France", "Germany", "Spain")),
    ColumnSpec("Gender", "category", categories=("Female", "Male")),
    ColumnSpec("Age", "integer", 18, 100, description="Age in years"),
    ColumnSpec("Tenure", "integer", 0, 10, description="Years as a customer"),
    ColumnSpec("Balance", "float", 0, 10_000_000, description="Account balance"),
    ColumnSpec("NumOfProducts", "integer", 1, 4, description="Products held"),
    ColumnSpec("HasCrCard", "integer", 0, 1, description="Binary flag"),
    ColumnSpec("IsActiveMember", "integer", 0, 1, description="Binary flag"),
    ColumnSpec("EstimatedSalary", "float", 0, 10_000_000, description="Estimated salary"),
    ColumnSpec("Exited", "integer", 0, 1, description="Target: 1 = exited in the snapshot"),
)
SPEC_BY_NAME = {spec.name: spec for spec in COLUMNS}

# Present in common copies of the dataset; carry no signal and are never read as features.
# Surname is personal data and is intentionally not stored by the platform.
IGNORED_COLUMNS = ("RowNumber", "Surname")

# Columns reported in related public variants of this dataset that are outcome-derived or
# otherwise unavailable before the exit. Their presence is an error, not a warning: the
# file is not the dataset this contract describes, and silently ignoring them would hide a
# leakage hazard from whoever trains on the file next.
LEAKAGE_SUSPECT_COLUMNS = ("Complain", "Satisfaction Score", "Card Type", "Point Earned")
