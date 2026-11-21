"""Unit tests: chronological splits and strict label isolation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import TemporalSplitSpec
from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from src.splits import (
    TimeSplit,
    apply_temporal_split,
    apply_time_split,
    assert_features_not_from_future,
    assert_no_feature_year_in_train,
    assert_strict_label_isolation,
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


def test_apply_time_split_bounds_legacy() -> None:
    df = _fake_panel()
    split = TimeSplit(name="t", train_max_year=2022, test_year=2023)
    train, test = apply_time_split(df, split)
    assert train["inventory_year"].max() <= 2022
    assert set(test["inventory_year"].unique()) == {2023}
    assert (train["inventory_year"] >= 2023).sum() == 0


def test_no_test_year_in_train_assertion() -> None:
    train = pd.DataFrame({"inventory_year": [2020, 2021, 2023]})
    with pytest.raises(AssertionError, match="Leakage"):
        assert_no_feature_year_in_train(train, test_year=2023)


def test_features_reject_future_and_bridge_id() -> None:
    df = _fake_panel()
    with pytest.raises(AssertionError, match="Forbidden"):
        assert_features_not_from_future(df, list(FEATURE_COLUMNS) + ["y_drop"])
    with pytest.raises(AssertionError, match="Forbidden"):
        assert_features_not_from_future(df, list(FEATURE_COLUMNS) + ["bridge_id"])
    assert_features_not_from_future(
        df, list(FEATURE_COLUMNS) + list(CATEGORICAL_FEATURES)
    )


def test_bridge_future_feature_rows_not_in_train() -> None:
    df = _fake_panel()
    split = TimeSplit(name="t", train_max_year=2022, test_year=2023)
    train, test = apply_time_split(df, split)
    shared = set(train["bridge_id"]) & set(test["bridge_id"])
    assert shared
    merged = train.merge(
        test[["bridge_id", "inventory_year"]],
        on="bridge_id",
        suffixes=("_train", "_test"),
    )
    assert (merged["inventory_year_train"] >= merged["inventory_year_test"]).sum() == 0


def test_strict_protocol_rejects_label_overlap() -> None:
    df = _fake_panel()
    # Deliberately bad strict split: train_max=2022 ⇒ labels include 2023 = test feature year
    bad = TemporalSplitSpec(
        name="bad_strict",
        protocol="strict",
        train_max_feature_year=2022,
        validation_feature_year=None,
        test_feature_year=2023,
    )
    with pytest.raises(AssertionError, match="Strict protocol"):
        apply_temporal_split(df, bad)


def test_strict_protocol_accepts_isolated_labels() -> None:
    df = _fake_panel()
    ok = TemporalSplitSpec(
        name="ok_strict",
        protocol="strict",
        train_max_feature_year=2021,
        validation_feature_year=None,
        test_feature_year=2023,
    )
    train, val, test = apply_temporal_split(df, ok)
    assert train["next_year"].max() < 2023
    assert set(test["inventory_year"].unique()) == {2023}


def test_assert_strict_helper_direct() -> None:
    train = pd.DataFrame({"next_year": [2021, 2022]})
    assert_strict_label_isolation(train, train.iloc[0:0], test_feature_year=2023)
    with pytest.raises(AssertionError):
        assert_strict_label_isolation(
            pd.DataFrame({"next_year": [2023]}),
            train.iloc[0:0],
            test_feature_year=2023,
        )
