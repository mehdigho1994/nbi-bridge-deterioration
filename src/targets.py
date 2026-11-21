"""Target construction for inventory-year deterioration tasks.

Primary target (unchanged definition)
-------------------------------------
y_drop = 1 if min(deck, super, sub) decreases by >= 1 from year t to t+1.

Supplementary targets (evaluated separately; do not replace the primary)
------------------------------------------------------------------------
- y_any_component_drop: any of the three components drops by >= 1
- y_deck_drop / y_super_drop / y_sub_drop: per-component drops

Missing / N policy: a component-specific target is undefined if that component
is N/missing at t or t+1. The min-based primary target requires all three
numeric at both years. The any-component target requires at least one component
comparable at both years; others may be missing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import COL_DECK, COL_SUB, COL_SUPER
from src.load_nbi import parse_condition


@dataclass
class TargetBuildReport:
    n_bridge_years: int
    n_with_next_year: int
    n_excluded_current_N_or_missing: int
    n_excluded_next_N_or_missing: int
    n_primary_defined: int
    n_primary_positive: int
    n_primary_negative: int
    alt_defined: dict[str, int]
    alt_positive: dict[str, int]


def compute_min_rating(deck: object, super_: object, sub: object) -> float | None:
    vals = [parse_condition(deck), parse_condition(super_), parse_condition(sub)]
    if any(v is None for v in vals):
        return None
    return float(min(v for v in vals if v is not None))


def _is_missing(v: object) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def primary_drop_target(
    current_min: float | None, next_min: float | None
) -> float | None:
    if _is_missing(current_min) or _is_missing(next_min):
        return None
    return 1.0 if float(next_min) <= float(current_min) - 1.0 else 0.0


def component_drop_target(
    current: float | None, nxt: float | None
) -> float | None:
    if _is_missing(current) or _is_missing(nxt):
        return None
    return 1.0 if float(nxt) <= float(current) - 1.0 else 0.0


def any_component_drop_target(
    deck_t: float | None,
    super_t: float | None,
    sub_t: float | None,
    deck_tp1: float | None,
    super_tp1: float | None,
    sub_tp1: float | None,
) -> float | None:
    """1 if any comparable component drops by >=1; None if none comparable."""
    pairs = [
        (deck_t, deck_tp1),
        (super_t, super_tp1),
        (sub_t, sub_tp1),
    ]
    comparable = False
    any_drop = False
    for cur, nxt in pairs:
        if _is_missing(cur) or _is_missing(nxt):
            continue
        comparable = True
        if float(nxt) <= float(cur) - 1.0:
            any_drop = True
    if not comparable:
        return None
    return 1.0 if any_drop else 0.0


def attach_targets(panel: pd.DataFrame) -> tuple[pd.DataFrame, TargetBuildReport]:
    """Add current/next ratings and primary + supplementary targets."""
    df = panel.sort_values(["bridge_id", "inventory_year"]).copy()

    df["deck_rating"] = df[COL_DECK].map(parse_condition)
    df["super_rating"] = df[COL_SUPER].map(parse_condition)
    df["sub_rating"] = df[COL_SUB].map(parse_condition)
    df["min_rating"] = [
        compute_min_rating(d, s, u)
        for d, s, u in zip(df[COL_DECK], df[COL_SUPER], df[COL_SUB], strict=True)
    ]

    g = df.groupby("bridge_id", sort=False)
    df["next_year"] = g["inventory_year"].shift(-1)
    df["next_min_rating"] = g["min_rating"].shift(-1)
    df["next_deck_rating"] = g["deck_rating"].shift(-1)
    df["next_super_rating"] = g["super_rating"].shift(-1)
    df["next_sub_rating"] = g["sub_rating"].shift(-1)

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
    df["y_deck_drop"] = [
        component_drop_target(c, n)
        for c, n in zip(df["deck_rating"], df["next_deck_rating"], strict=True)
    ]
    df["y_super_drop"] = [
        component_drop_target(c, n)
        for c, n in zip(df["super_rating"], df["next_super_rating"], strict=True)
    ]
    df["y_sub_drop"] = [
        component_drop_target(c, n)
        for c, n in zip(df["sub_rating"], df["next_sub_rating"], strict=True)
    ]
    df["y_any_component_drop"] = [
        any_component_drop_target(d0, s0, u0, d1, s1, u1)
        for d0, s0, u0, d1, s1, u1 in zip(
            df["deck_rating"],
            df["super_rating"],
            df["sub_rating"],
            df["next_deck_rating"],
            df["next_super_rating"],
            df["next_sub_rating"],
            strict=True,
        )
    ]

    n_bridge_years = len(df)
    n_with_next = int(consecutive.sum())
    current_missing = df["min_rating"].isna()
    next_missing = consecutive & df["next_min_rating"].isna()
    defined = df["y_drop"].notna()
    n_pos = int((df.loc[defined, "y_drop"] == 1).sum())
    n_neg = int((df.loc[defined, "y_drop"] == 0).sum())

    alt_defined: dict[str, int] = {}
    alt_positive: dict[str, int] = {}
    for col in (
        "y_any_component_drop",
        "y_deck_drop",
        "y_super_drop",
        "y_sub_drop",
    ):
        m = df[col].notna()
        alt_defined[col] = int(m.sum())
        alt_positive[col] = int((df.loc[m, col] == 1).sum())

    report = TargetBuildReport(
        n_bridge_years=n_bridge_years,
        n_with_next_year=n_with_next,
        n_excluded_current_N_or_missing=int((consecutive & current_missing).sum()),
        n_excluded_next_N_or_missing=int(next_missing.sum()),
        n_primary_defined=int(defined.sum()),
        n_primary_positive=n_pos,
        n_primary_negative=n_neg,
        alt_defined=alt_defined,
        alt_positive=alt_positive,
    )
    return df, report
