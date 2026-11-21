"""Chronological train / validation / test splits."""

from __future__ import annotations

import pandas as pd

from src.config import (
    STANDARD_SPLITS,
    STRICT_SPLITS,
    TemporalSplitSpec,
)


def default_standard_splits() -> list[TemporalSplitSpec]:
    return list(STANDARD_SPLITS)


def default_strict_splits() -> list[TemporalSplitSpec]:
    return list(STRICT_SPLITS)


def apply_temporal_split(
    df: pd.DataFrame,
    spec: TemporalSplitSpec,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_fit, validation, test) with defined target.

    train_fit: inventory_year <= train_max_feature_year
    validation: inventory_year == validation_feature_year (may be empty)
    test: inventory_year == test_feature_year

    Raises if temporal integrity checks fail.
    """
    target_col = target_col or spec.target_col
    labeled = df[df[target_col].notna()].copy()

    train = labeled[
        labeled["inventory_year"] <= spec.train_max_feature_year
    ].copy()
    if spec.validation_feature_year is None:
        val = labeled.iloc[0:0].copy()
    else:
        val = labeled[
            labeled["inventory_year"] == spec.validation_feature_year
        ].copy()
    test = labeled[labeled["inventory_year"] == spec.test_feature_year].copy()

    assert_no_feature_year_in_train(train, spec.test_feature_year)
    assert_train_years_bounded(train, spec.train_max_feature_year)
    if not test.empty:
        assert int(test["inventory_year"].min()) == spec.test_feature_year
        assert int(test["inventory_year"].max()) == spec.test_feature_year

    if spec.protocol == "strict":
        assert_strict_label_isolation(train, val, spec.test_feature_year)

    # Validation must not overlap test or include years after test.
    if not val.empty and (val["inventory_year"] >= spec.test_feature_year).any():
        raise AssertionError("Validation feature years must be < test feature year")

    return train, val, test


def assert_no_feature_year_in_train(train: pd.DataFrame, test_year: int) -> None:
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
    if train.empty:
        return
    if int(train["inventory_year"].max()) > train_max_year:
        raise AssertionError(
            f"Train max year {int(train['inventory_year'].max())} exceeds "
            f"train_max_year={train_max_year}"
        )


def assert_strict_label_isolation(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test_feature_year: int,
) -> None:
    """Require max label observation year in train(+val) < test feature year."""
    for name, part in (("train", train), ("validation", val)):
        if part.empty or "next_year" not in part.columns:
            continue
        max_label = part["next_year"].max()
        if pd.notna(max_label) and int(max_label) >= test_feature_year:
            raise AssertionError(
                f"Strict protocol violated in {name}: max label year "
                f"{int(max_label)} >= test feature year {test_feature_year}"
            )


def assert_features_not_from_future(
    df: pd.DataFrame, feature_cols: list[str], year_col: str = "inventory_year"
) -> None:
    forbidden = {
        "next_year",
        "next_min_rating",
        "next_deck_rating",
        "next_super_rating",
        "next_sub_rating",
        "y_drop",
        "y_any_component_drop",
        "y_deck_drop",
        "y_super_drop",
        "y_sub_drop",
        "bridge_id",
    }
    overlap = forbidden.intersection(feature_cols)
    if overlap:
        raise AssertionError(f"Forbidden columns passed as features: {overlap}")
    if year_col not in df.columns:
        raise AssertionError(f"Missing {year_col}")


# Backwards-compatible aliases used by older tests
def apply_time_split(df, split, target_col="y_drop"):
    """Legacy wrapper: TimeSplit(train_max, test) → train, test (val unused)."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Legacy:
        name: str
        train_max_year: int
        test_year: int

    if hasattr(split, "train_max_feature_year"):
        spec = split
    else:
        spec = TemporalSplitSpec(
            name=getattr(split, "name", "legacy"),
            protocol="standard",
            train_max_feature_year=split.train_max_year,
            validation_feature_year=None,
            test_feature_year=split.test_year,
            target_col=target_col,
        )
    train, _val, test = apply_temporal_split(df, spec, target_col=target_col)
    return train, test


# Re-export name expected by old tests
from dataclasses import dataclass as _dataclass  # noqa: E402


@_dataclass(frozen=True)
class TimeSplit:
    name: str
    train_max_year: int
    test_year: int
