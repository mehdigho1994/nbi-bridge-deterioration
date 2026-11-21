"""Target construction for deterioration prediction.

Primary target (binary): did min(deck, super, sub) decrease by >= 1 next year?
Secondary (optional multiclass): next-year rating for each component.

Policy for 'N' / missing: if the current or next year's min rating cannot be
computed because any of items 58–60 is N/missing, the primary target is
undefined and that bridge-year is excluded from the primary task. Counts of
excluded rows are returned for documentation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import COL_DECK, COL_SUB, COL_SUPER
from src.load_nbi import parse_condition


@dataclass
class TargetBuildReport:
    """Counts documenting exclusion decisions for the primary target."""

    n_bridge_years: int
    n_with_next_year: int
    n_excluded_current_N_or_missing: int
    n_excluded_next_N_or_missing: int
    n_primary_defined: int
    n_primary_positive: int
    n_primary_negative: int


def compute_min_rating(deck: object, super_: object, sub: object) -> float | None:
    """Return min of three numeric ratings, or None if any is N/missing."""
    vals = [parse_condition(deck), parse_condition(super_), parse_condition(sub)]
    if any(v is None for v in vals):
        return None
    return float(min(v for v in vals if v is not None))


def primary_drop_target(
    current_min: float | None, next_min: float | None
) -> float | None:
    """1 if next_min <= current_min - 1, else 0; None if either min undefined."""
    if (
        current_min is None
        or next_min is None
        or (isinstance(current_min, float) and np.isnan(current_min))
        or (isinstance(next_min, float) and np.isnan(next_min))
        or pd.isna(current_min)
        or pd.isna(next_min)
    ):
        return None
    return 1.0 if float(next_min) <= float(current_min) - 1.0 else 0.0


def attach_targets(panel: pd.DataFrame) -> tuple[pd.DataFrame, TargetBuildReport]:
    """Add current/next ratings and primary/secondary targets to a sorted panel.

    Expects columns: bridge_id, inventory_year, DECK/SUPER/SUB condition cols.
    Rows must be unique on (bridge_id, inventory_year).
    """
    df = panel.sort_values(["bridge_id", "inventory_year"]).copy()

    df["deck_rating"] = df[COL_DECK].map(parse_condition)
    df["super_rating"] = df[COL_SUPER].map(parse_condition)
    df["sub_rating"] = df[COL_SUB].map(parse_condition)
    df["min_rating"] = [
        compute_min_rating(d, s, u)
        for d, s, u in zip(df[COL_DECK], df[COL_SUPER], df[COL_SUB], strict=True)
    ]

    # Lead by one inventory year within each bridge (no cross-bridge leakage).
    g = df.groupby("bridge_id", sort=False)
    df["next_year"] = g["inventory_year"].shift(-1)
    df["next_min_rating"] = g["min_rating"].shift(-1)
    df["next_deck_rating"] = g["deck_rating"].shift(-1)
    df["next_super_rating"] = g["super_rating"].shift(-1)
    df["next_sub_rating"] = g["sub_rating"].shift(-1)

    # Only treat as "next year" if inventory years are consecutive.
    consecutive = df["next_year"] == df["inventory_year"] + 1
    for col in (
        "next_min_rating",
        "next_deck_rating",
        "next_super_rating",
        "next_sub_rating",
    ):
        df.loc[~consecutive, col] = np.nan
    df.loc[~consecutive, "next_year"] = np.nan

    df["y_drop"] = [
        primary_drop_target(c, n)
        for c, n in zip(df["min_rating"], df["next_min_rating"], strict=True)
    ]

    n_bridge_years = len(df)
    n_with_next = int(consecutive.sum())
    current_missing = df["min_rating"].isna()
    next_missing = consecutive & df["next_min_rating"].isna()
    # Exclusion categories (a row can contribute to both if both ends missing).
    n_excl_current = int((consecutive & current_missing).sum())
    n_excl_next = int(next_missing.sum())
    defined = df["y_drop"].notna()
    n_pos = int((df.loc[defined, "y_drop"] == 1).sum())
    n_neg = int((df.loc[defined, "y_drop"] == 0).sum())

    report = TargetBuildReport(
        n_bridge_years=n_bridge_years,
        n_with_next_year=n_with_next,
        n_excluded_current_N_or_missing=n_excl_current,
        n_excluded_next_N_or_missing=n_excl_next,
        n_primary_defined=int(defined.sum()),
        n_primary_positive=n_pos,
        n_primary_negative=n_neg,
    )
    return df, report
