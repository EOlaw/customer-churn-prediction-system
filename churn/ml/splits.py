"""Deterministic train / calibration / test partitioning.

Assignments depend only on (customer id, label, seed), never on file row order: rows are sorted
by identifier before the stratified split. Each partition has one job:

* ``train``       fit candidates, cross-validated model selection, final refit
* ``calibration`` choose and fit the probability calibrator and the decision threshold
* ``test``        touched once, after everything else is frozen
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

PARTITIONS = ("train", "calibration", "test")
FRACTIONS = {"train": 0.6, "calibration": 0.2, "test": 0.2}


def assign_partitions(ids: pd.Series, labels: pd.Series, seed: int) -> pd.Series:
    """Return a partition name for every row, aligned to ``ids.index``."""
    if ids.duplicated().any():
        raise ValueError("Customer identifiers must be unique before splitting")
    ordered = pd.DataFrame({"id": ids.astype(str), "y": labels}).sort_values("id", kind="stable")
    rest, test = train_test_split(ordered.index, test_size=FRACTIONS["test"], random_state=seed,
                                  stratify=ordered.loc[:, "y"])
    train, calibration = train_test_split(
        rest, test_size=FRACTIONS["calibration"] / (1 - FRACTIONS["test"]), random_state=seed,
        stratify=ordered.loc[rest, "y"])
    assignment = pd.Series(index=ids.index, dtype=object)
    assignment.loc[train], assignment.loc[calibration], assignment.loc[test] = (
        "train", "calibration", "test")
    verify_partitions(assignment)
    return assignment


def verify_partitions(assignment: pd.Series) -> None:
    if assignment.isna().any() or not set(assignment.unique()) <= set(PARTITIONS):
        raise RuntimeError("Every row must be assigned to exactly one known partition")
    if set(assignment.unique()) != set(PARTITIONS):
        raise RuntimeError("All three partitions must be non-empty")
