"""HistGradientBoosting models with chronological hyperparameter selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import HGB_MAX_ITER_GRID
from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS


def build_hgb_pipeline(
    seed: int,
    *,
    max_iter: int = 150,
    class_weight: str | None = None,
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> Pipeline:
    """HGB without random early_stopping; max_iter chosen externally on time val."""
    numeric = numeric if numeric is not None else list(FEATURE_COLUMNS)
    categorical = categorical if categorical is not None else list(CATEGORICAL_FEATURES)
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
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
        ],
        remainder="drop",
    )
    # class_weight supported on HistGradientBoostingClassifier in recent sklearn
    kwargs: dict = dict(
        max_depth=4,
        learning_rate=0.08,
        max_iter=max_iter,
        min_samples_leaf=50,
        l2_regularization=0.1,
        early_stopping=False,
        random_state=seed,
    )
    if class_weight is not None:
        kwargs["class_weight"] = class_weight
    clf = HistGradientBoostingClassifier(**kwargs)
    return Pipeline([("pre", pre), ("clf", clf)])


def _prep(df: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in categorical:
        if c in out.columns:
            out[c] = out[c].fillna("").astype(str)
    return out


def select_hgb_max_iter(
    train: pd.DataFrame,
    val: pd.DataFrame,
    seed: int,
    *,
    class_weight: str | None = None,
    target_col: str = "y_drop",
    feature_columns: list[str] | None = None,
    categorical_features: list[str] | None = None,
    grid: tuple[int, ...] = HGB_MAX_ITER_GRID,
) -> tuple[int, dict[str, float]]:
    """Choose max_iter by Average Precision on a chronological validation year."""
    if val.empty:
        return grid[-1], {"note": "empty validation; using max grid value"}

    numeric = feature_columns if feature_columns is not None else list(FEATURE_COLUMNS)
    categorical = (
        categorical_features
        if categorical_features is not None
        else list(CATEGORICAL_FEATURES)
    )
    cat_in = [c for c in categorical if c in train.columns]
    num_in = [c for c in numeric if c in train.columns]
    x_cols = num_in + cat_in
    train_p = _prep(train, cat_in)
    val_p = _prep(val, cat_in)
    y_val = val[target_col].astype(int).to_numpy()

    best_iter = grid[0]
    best_ap = -1.0
    scores: dict[str, float] = {}
    for m in grid:
        pipe = build_hgb_pipeline(
            seed, max_iter=m, class_weight=class_weight, numeric=num_in, categorical=cat_in
        )
        pipe.fit(train_p[x_cols], train_p[target_col].astype(int))
        prob = pipe.predict_proba(val_p[x_cols])[:, 1]
        try:
            ap = float(average_precision_score(y_val, prob))
        except ValueError:
            ap = float("nan")
        scores[str(m)] = ap
        if ap == ap and ap > best_ap:
            best_ap = ap
            best_iter = m
    return best_iter, scores


def fit_hgb(
    train: pd.DataFrame,
    seed: int,
    *,
    max_iter: int = 150,
    class_weight: str | None = None,
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
    cat_in = [c for c in categorical if c in train.columns]
    num_in = [c for c in numeric if c in train.columns]
    pipe = build_hgb_pipeline(
        seed, max_iter=max_iter, class_weight=class_weight, numeric=num_in, categorical=cat_in
    )
    train = _prep(train, cat_in)
    x_cols = num_in + cat_in
    pipe.fit(train[x_cols], train[target_col].astype(int))
    pipe._x_cols_ = x_cols  # type: ignore[attr-defined]
    pipe._max_iter_ = max_iter  # type: ignore[attr-defined]
    pipe._class_weight_ = class_weight  # type: ignore[attr-defined]
    return pipe


def predict_hgb(model: Pipeline, test: pd.DataFrame) -> np.ndarray:
    x_cols: list[str] = getattr(model, "_x_cols_")
    cat = [c for c in CATEGORICAL_FEATURES if c in x_cols]
    test = _prep(test, cat)
    return model.predict_proba(test[x_cols])[:, 1]
