"""Main ML model: sklearn HistGradientBoostingClassifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS


def build_hgb_pipeline(seed: int) -> Pipeline:
    """Gradient boosting; numerics keep NaNs (native HGB support); cats one-hot."""
    numeric = list(FEATURE_COLUMNS)
    categorical = list(CATEGORICAL_FEATURES)

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    clf = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=200,
        min_samples_leaf=50,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        class_weight="balanced",
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def fit_hgb(train: pd.DataFrame, seed: int) -> Pipeline:
    """Fit HistGradientBoosting on training frame."""
    pipe = build_hgb_pipeline(seed)
    x_cols = list(FEATURE_COLUMNS) + list(CATEGORICAL_FEATURES)
    y = train["y_drop"].astype(int)
    # Ensure categoricals are strings for the imputer/encoder.
    train = train.copy()
    for c in CATEGORICAL_FEATURES:
        train[c] = train[c].fillna("").astype(str)
    pipe.fit(train[x_cols], y)
    return pipe


def predict_hgb(model: Pipeline, test: pd.DataFrame) -> np.ndarray:
    """Return P(drop=1) for test rows."""
    x_cols = list(FEATURE_COLUMNS) + list(CATEGORICAL_FEATURES)
    test = test.copy()
    for c in CATEGORICAL_FEATURES:
        test[c] = test[c].fillna("").astype(str)
    return model.predict_proba(test[x_cols])[:, 1]
