"""Unit tests for primary target construction and N/missing exclusion."""

from __future__ import annotations

import pandas as pd

from src.config import COL_DECK, COL_SUB, COL_SUPER
from src.targets import attach_targets, compute_min_rating, primary_drop_target


def _row(year: int, deck: str, super_: str, sub: str, bridge: str = "51|A") -> dict:
    return {
        "bridge_id": bridge,
        "inventory_year": year,
        COL_DECK: deck,
        COL_SUPER: super_,
        COL_SUB: sub,
        "STATE_CODE_001": "51",
        "STRUCTURE_NUMBER_008": "A",
        "YEAR_BUILT_027": "1990",
        "ADT_029": "1000",
        "STRUCTURE_KIND_043A": "3",
        "STRUCTURE_TYPE_043B": "02",
        "DATE_OF_INSPECT_090": "0622",
    }


def test_compute_min_rating_numeric() -> None:
    assert compute_min_rating("7", "6", "8") == 6.0


def test_compute_min_rating_excludes_N() -> None:
    assert compute_min_rating("7", "N", "8") is None
    assert compute_min_rating("", "6", "8") is None


def test_primary_drop_target() -> None:
    assert primary_drop_target(7.0, 6.0) == 1.0
    assert primary_drop_target(7.0, 7.0) == 0.0
    assert primary_drop_target(7.0, 8.0) == 0.0
    assert primary_drop_target(None, 6.0) is None
    assert primary_drop_target(7.0, None) is None


def test_attach_targets_drop_and_exclusion_counts() -> None:
    # Bridge A: 7->6 drop; Bridge B: next year has N → excluded.
    df = pd.DataFrame(
        [
            _row(2020, "7", "7", "7", "51|A"),
            _row(2021, "6", "7", "7", "51|A"),
            _row(2020, "5", "5", "5", "51|B"),
            _row(2021, "N", "5", "5", "51|B"),
            _row(2020, "N", "8", "8", "51|C"),  # current N
            _row(2021, "8", "8", "8", "51|C"),
        ]
    )
    out, report = attach_targets(df)

    a_2020 = out[(out["bridge_id"] == "51|A") & (out["inventory_year"] == 2020)].iloc[0]
    assert a_2020["y_drop"] == 1.0

    b_2020 = out[(out["bridge_id"] == "51|B") & (out["inventory_year"] == 2020)].iloc[0]
    assert pd.isna(b_2020["y_drop"])

    c_2020 = out[(out["bridge_id"] == "51|C") & (out["inventory_year"] == 2020)].iloc[0]
    assert pd.isna(c_2020["y_drop"])

    assert report.n_with_next_year == 3
    assert report.n_excluded_next_N_or_missing >= 1
    assert report.n_excluded_current_N_or_missing >= 1
    assert report.n_primary_defined == 1
    assert report.n_primary_positive == 1


def test_non_consecutive_years_not_treated_as_next() -> None:
    df = pd.DataFrame(
        [
            _row(2018, "7", "7", "7"),
            _row(2020, "5", "5", "5"),  # gap — not next year
        ]
    )
    out, report = attach_targets(df)
    row = out[out["inventory_year"] == 2018].iloc[0]
    assert pd.isna(row["y_drop"])
    assert report.n_with_next_year == 0
