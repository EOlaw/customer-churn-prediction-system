"""Serializable scoring artifact: preprocessing + classifier + optional probability calibrator.

Kept in its own module so a pickled artifact only references stable import paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CALIBRATION_METHODS = ("none", "sigmoid", "isotonic")  # ordered simplest first
_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class Calibrator:
    """Maps raw model probabilities to calibrated ones. Fit on the calibration partition only."""

    def __init__(self, method: str):
        if method not in CALIBRATION_METHODS:
            raise ValueError(f"Unknown calibration method {method!r}")
        self.method = method
        self._fitted: object | None = None

    def fit(self, raw: np.ndarray, y: np.ndarray) -> Calibrator:
        if self.method == "sigmoid":
            self._fitted = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self._fitted = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, y)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        if self.method == "none":
            return raw
        if self._fitted is None:
            raise RuntimeError("Calibrator is not fitted")
        if self.method == "sigmoid":
            return self._fitted.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]  # type: ignore[attr-defined]
        return np.asarray(self._fitted.predict(raw))  # type: ignore[attr-defined]


@dataclass
class ScoringModel:
    """What gets pickled: ``predict_proba`` accepts a frame with all banking features."""

    pipeline: object
    calibrator: Calibrator
    feature_columns: tuple[str, ...]

    def raw_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(frame[list(self.feature_columns)])[:, 1]  # type: ignore[attr-defined]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.calibrator.predict(self.raw_proba(frame)), 0.0, 1.0)
        return np.column_stack([1 - p, p])
