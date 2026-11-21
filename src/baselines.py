"""Baselines: persistence, Markov, logistic regression (weighted / unweighted)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS


def persistence_drop_proba(test: pd.DataFrame) -> np.ndarray:
    """P(drop)=0 for all rows (rating assumed unchanged)."""
    return np.zeros(len(test), dtype=float)


class MarkovDropModel:
    """P(drop | current min rating) estimated on training rows only."""

    def __init__(self) -> None:
        self.p_drop_given_min: dict[int, float] = {}
        self.global_rate: float = 0.0
        self.component_transitions: dict[str, pd.DataFrame] = {}

    def fit(self, train: pd.DataFrame, target_col: str = "y_drop") -> MarkovDropModel:
        y = train[target_col].astype(float)
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
            self.component_transitions[comp] = pd.crosstab(
                tmp[comp], tmp[next_col], normalize="index"
            )
        return self

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        probs = []
        for val in test["min_rating"]:
            if pd.isna(val):
                probs.append(self.global_rate)
            else:
                probs.append(self.p_drop_given_min.get(int(val), self.global_rate))
        return np.asarray(probs, dtype=float)


def build_logistic_pipeline(
    class_weight: str | None,
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> Pipeline:
    numeric = numeric if numeric is not None else list(FEATURE_COLUMNS)
    categorical = categorical if categorical is not None else list(CATEGORICAL_FEATURES)
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
        class_weight=class_weight,
        random_state=0,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _prep_frame(df: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in categorical:
        if c in out.columns:
            out[c] = out[c].fillna("").astype(str)
    return out


def fit_logistic(
    train: pd.DataFrame,
    seed: int,
    *,
    class_weight: str | None = "balanced",
    target_col: str = "y_drop",
    feature_columns: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    numeric = feature_columns if feature_columns is not None else list(FEATURE_COLUMNS)
    categorical = (
        categorical_features
        if categorical_features is not None
        else list(CATEGORICAL_FEATURES)
    )
    # Only keep categoricals that are requested
    cat_in = [c for c in categorical if c in train.columns]
    num_in = [c for c in numeric if c in train.columns]
    pipe = build_logistic_pipeline(class_weight, numeric=num_in, categorical=cat_in)
    pipe.set_params(clf__random_state=seed)
    train = _prep_frame(train, cat_in)
    x_cols = num_in + cat_in
    pipe.fit(train[x_cols], train[target_col].astype(int))
    pipe._x_cols_ = x_cols  # type: ignore[attr-defined]
    return pipe


def predict_logistic(model: Pipeline, test: pd.DataFrame) -> np.ndarray:
    x_cols: list[str] = getattr(model, "_x_cols_")
    cat = [c for c in x_cols if c in CATEGORICAL_FEATURES]
    test = _prep_frame(test, cat)
    return model.predict_proba(test[x_cols])[:, 1]
