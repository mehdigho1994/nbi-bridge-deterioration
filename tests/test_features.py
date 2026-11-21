"""Lag / calendar-year feature tests."""

from __future__ import annotations

import pandas as pd

from src.config import COL_DECK, COL_SUB, COL_SUPER
from src.features import add_features
from src.targets import attach_targets


def _row(year: int, deck: str, super_: str, sub: str, bridge: str = "51|G") -> dict:
    return {
        "bridge_id": bridge,
        "inventory_year": year,
        COL_DECK: deck,
        COL_SUPER: super_,
        COL_SUB: sub,
        "STATE_CODE_001": "51",
        "STRUCTURE_NUMBER_008": "G",
        "YEAR_BUILT_027": "1990",
        "ADT_029": "1000",
        "STRUCTURE_KIND_043A": "3",
        "STRUCTURE_TYPE_043B": "02",
        "DATE_OF_INSPECT_090": "0622",
    }


def test_lag_requires_consecutive_calendar_years() -> None:
    df = pd.DataFrame(
        [
            _row(2018, "7", "7", "7"),
            _row(2020, "6", "6", "6"),  # gap: not adjacent
        ]
    )
    out, _ = attach_targets(df)
    feat = add_features(out)
    row = feat[feat["inventory_year"] == 2020].iloc[0]
    assert row["has_hist_1y"] == 0.0
    assert pd.isna(row["min_delta_1y"])


def test_years_since_uses_calendar_span_across_gap() -> None:
    df = pd.DataFrame(
        [
            _row(2016, "5", "5", "5"),
            _row(2017, "5", "5", "5"),
            _row(2018, "5", "5", "5"),
            _row(2022, "5", "5", "5"),  # gap after unchanged rating
        ]
    )
    out, _ = attach_targets(df)
    feat = add_features(out)
    row = feat[feat["inventory_year"] == 2022].iloc[0]
    # Last change year is first year 2016 (series start); calendar span = 6
    assert row["years_since_rating_change"] == 6.0
