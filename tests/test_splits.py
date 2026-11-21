"""Unit tests: time splits never put test-year rows into training."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from src.splits import (
    TimeSplit,
    apply_time_split,
    assert_features_not_from_future,
    assert_no_test_year_in_train,
)


def _fake_panel() -> pd.DataFrame:
    rows = []
    for bridge in ("51|A", "51|B", "51|C"):
        for year in range(2019, 2025):
            rows.append(
                {
                    "bridge_id": bridge,
                    "inventory_year": year,
                    "y_drop": 0.0 if year % 2 == 0 else 1.0,
                    "min_rating": 7.0,
                    "next_min_rating": 6.0,
                    "next_year": year + 1,
                }
            )
    return pd.DataFrame(rows)


def test_apply_time_split_bounds() -> None:
    df = _fake_panel()
    split = TimeSplit(name="t", train_max_year=2022, test_year=2023)
    train, test = apply_time_split(df, split)
    assert train["inventory_year"].max() <= 2022
    assert set(test["inventory_year"].unique()) == {2023}
    assert (train["inventory_year"] >= 2023).sum() == 0


def test_no_test_year_in_train_assertion() -> None:
    train = pd.DataFrame({"inventory_year": [2020, 2021, 2023]})
    with pytest.raises(AssertionError, match="Leakage"):
        assert_no_test_year_in_train(train, test_year=2023)


def test_features_reject_future_columns() -> None:
    df = _fake_panel()
    with pytest.raises(AssertionError, match="Target/future"):
        assert_features_not_from_future(df, list(FEATURE_COLUMNS) + ["y_drop"])
    # Allowed feature set should pass.
    assert_features_not_from_future(
        df, list(FEATURE_COLUMNS) + list(CATEGORICAL_FEATURES)
    )


def test_bridge_future_rows_not_in_train_features_year() -> None:
    """Even if a bridge appears in test, its test-year row must not be in train."""
    df = _fake_panel()
    split = TimeSplit(name="t", train_max_year=2022, test_year=2023)
    train, test = apply_time_split(df, split)
    # Same bridges can appear in both, but not the same year.
    shared = set(train["bridge_id"]) & set(test["bridge_id"])
    assert shared  # expected overlap across years
    merged = train.merge(
        test[["bridge_id", "inventory_year"]],
        on="bridge_id",
        suffixes=("_train", "_test"),
    )
    assert (merged["inventory_year_train"] >= merged["inventory_year_test"]).sum() == 0
