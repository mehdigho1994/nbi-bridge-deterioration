"""Baselines: persistence, Markov transitions, logistic regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS


@dataclass
class BaselinePredictions:
    """Container for baseline probability vectors on a test set."""

    name: str
    y_prob: np.ndarray


def persistence_drop_proba(test: pd.DataFrame) -> np.ndarray:
    """Persistence baseline for the *drop* task.

    Assumes ratings stay the same ⇒ predicted P(drop) = 0 for every row.
    (This is the natural analogue of 'rating stays the same' for a drop event.)
    """
    return np.zeros(len(test), dtype=float)


class MarkovDropModel:
    """Estimate P(min rating decreases by >=1 next year | current min rating).

    Transition counts are taken only from training bridge-years with a defined
    primary target. Per-component Markov models are also estimated for reporting.
    """

    def __init__(self) -> None:
        self.p_drop_given_min: dict[int, float] = {}
        self.global_rate: float = 0.0
        self.component_transitions: dict[str, pd.DataFrame] = {}

    def fit(self, train: pd.DataFrame) -> MarkovDropModel:
        y = train["y_drop"].astype(float)
        self.global_rate = float(y.mean()) if len(y) else 0.0

        rated = train["min_rating"].dropna()
        for rating in sorted(rated.unique()):
            mask = train["min_rating"] == rating
            subset = y[mask]
            self.p_drop_given_min[int(rating)] = (
                float(subset.mean()) if len(subset) else self.global_rate
            )

        for comp in ("deck_rating", "super_rating", "sub_rating"):
            next_col = f"next_{comp}"
            if next_col not in train.columns:
                continue
            tmp = train[[comp, next_col]].dropna()
            if tmp.empty:
                continue
            tmp = tmp.astype(int)
            ct = pd.crosstab(tmp[comp], tmp[next_col], normalize="index")
            self.component_transitions[comp] = ct
        return self

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        probs = []
        for val in test["min_rating"]:
            if pd.isna(val):
                probs.append(self.global_rate)
            else:
                probs.append(
                    self.p_drop_given_min.get(int(val), self.global_rate)
                )
        return np.asarray(probs, dtype=float)


def build_logistic_pipeline() -> Pipeline:
    """Logistic regression with imputation, scaling, and one-hot categoricals."""
    numeric = list(FEATURE_COLUMNS)
    categorical = list(CATEGORICAL_FEATURES)
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=0,  # overridden per seed in train loop
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _prep_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CATEGORICAL_FEATURES:
        out[c] = out[c].fillna("").astype(str)
    return out


def fit_logistic(
    train: pd.DataFrame, seed: int
) -> tuple[Pipeline, np.ndarray]:
    """Fit logistic regression; return model (caller predicts on test)."""
    pipe = build_logistic_pipeline()
    pipe.set_params(clf__random_state=seed)
    x_cols = list(FEATURE_COLUMNS) + list(CATEGORICAL_FEATURES)
    train = _prep_frame(train)
    pipe.fit(train[x_cols], train["y_drop"].astype(int))
    return pipe, np.asarray([])  # placeholder; predict separately


def predict_logistic(model: Pipeline, test: pd.DataFrame) -> np.ndarray:
    x_cols = list(FEATURE_COLUMNS) + list(CATEGORICAL_FEATURES)
    test = _prep_frame(test)
    return model.predict_proba(test[x_cols])[:, 1]
