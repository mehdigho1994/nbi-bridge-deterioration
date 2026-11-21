"""Time-based train/test splits with leakage checks.

Never split rows randomly. Training uses feature-years <= T; test uses T+1.
Features are already built from past/current rows only; this module additionally
asserts that no test-year inventory rows appear in the training frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import TIME_SPLITS, split_slug


@dataclass(frozen=True)
class TimeSplit:
    """One chronological split."""

    name: str
    train_max_year: int
    test_year: int


def default_splits() -> list[TimeSplit]:
    """Return configured rolling time splits."""
    return [
        TimeSplit(
            name=split_slug(tmax, ty),
            train_max_year=tmax,
            test_year=ty,
        )
        for tmax, ty in TIME_SPLITS
    ]


def apply_time_split(
    df: pd.DataFrame, split: TimeSplit, target_col: str = "y_drop"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) frames with defined primary target.

    Train: inventory_year <= train_max_year and target defined.
    Test: inventory_year == test_year and target defined.

    Raises AssertionError if any test-year row leaks into train, or if train
    contains any inventory_year > train_max_year.
    """
    labeled = df[df[target_col].notna()].copy()
    train = labeled[labeled["inventory_year"] <= split.train_max_year].copy()
    test = labeled[labeled["inventory_year"] == split.test_year].copy()

    assert_no_test_year_in_train(train, split.test_year)
    assert_train_years_bounded(train, split.train_max_year)
    # Test must not include future-of-test feature years.
    if not test.empty:
        assert int(test["inventory_year"].max()) == split.test_year
        assert int(test["inventory_year"].min()) == split.test_year
    return train, test


def assert_no_test_year_in_train(train: pd.DataFrame, test_year: int) -> None:
    """Fail if any training row has inventory_year == test_year (or later)."""
    if train.empty:
        return
    leaked = train["inventory_year"] >= test_year
    if leaked.any():
        cols = [c for c in ("bridge_id", "inventory_year") if c in train.columns]
        bad = train.loc[leaked, cols].head(5)
        raise AssertionError(
            f"Leakage: {int(leaked.sum())} train rows have inventory_year "
            f">= test_year={test_year}. Examples:\n{bad}"
        )


def assert_train_years_bounded(train: pd.DataFrame, train_max_year: int) -> None:
    """Fail if training includes years beyond the allowed train max."""
    if train.empty:
        return
    if int(train["inventory_year"].max()) > train_max_year:
        raise AssertionError(
            f"Train max year {int(train['inventory_year'].max())} exceeds "
            f"train_max_year={train_max_year}"
        )


def assert_features_not_from_future(
    df: pd.DataFrame, feature_cols: list[str], year_col: str = "inventory_year"
) -> None:
    """Sanity check used in tests: feature frame years are self-consistent.

    This does not prove lag correctness (covered by unit tests on synthetic
    sequences) but ensures we never pass a 'future_year' column as a feature.
    """
    forbidden = {"next_year", "next_min_rating", "next_deck_rating",
                 "next_super_rating", "next_sub_rating", "y_drop"}
    overlap = forbidden.intersection(feature_cols)
    if overlap:
        raise AssertionError(f"Target/future columns passed as features: {overlap}")
    if year_col not in df.columns:
        raise AssertionError(f"Missing {year_col}")
