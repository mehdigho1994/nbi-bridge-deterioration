"""Probability calibration fitted only on chronological validation data."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """Post-hoc calibrator for already-computed scores.

    Fit exclusively on validation (y_val, p_val). Never fit on test.
    Methods: ``sigmoid`` (Platt scaling via logistic on scores) and
    ``isotonic`` (isotonic regression).
    """

    def __init__(self, method: str = "sigmoid") -> None:
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError(method)
        self.method = method
        self._model: Any = None

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> ProbabilityCalibrator:
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6).reshape(-1, 1)
        if self.method == "sigmoid":
            clf = LogisticRegression(max_iter=1000)
            clf.fit(y_prob, y_true)
            self._model = clf
        else:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(y_prob.ravel(), y_true)
            self._model = iso
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Calibrator not fitted")
        y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "sigmoid":
            return self._model.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        return np.asarray(self._model.predict(y_prob), dtype=float)
