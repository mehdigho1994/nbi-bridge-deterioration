"""Feature engineering using only current and past information per bridge."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    COL_ADT,
    COL_STRUCTURE_KIND,
    COL_STRUCTURE_TYPE,
    COL_YEAR_BUILT,
    MAX_LAG_YEARS,
)
from src.load_nbi import normalize_code


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.map(normalize_code), errors="coerce")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model features. Never uses future rows.

    Lag deltas use groupby.shift(lag) gated by calendar-year equality
    (past_year == inventory_year - lag). Missing years therefore yield NaN
    deltas rather than treating non-consecutive rows as adjacent years.

    years_since_rating_change uses calendar inventory years relative to the
    most recent prior year (including the current row) where min_rating
    differed from the previous observed value, or the first observed year.
    """
    out = df.sort_values(["bridge_id", "inventory_year"]).copy()
    g = out.groupby("bridge_id", sort=False)

    for col in ("deck_rating", "super_rating", "sub_rating", "min_rating"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for lag in range(1, MAX_LAG_YEARS + 1):
        past_min = g["min_rating"].shift(lag)
        past_year = g["inventory_year"].shift(lag)
        valid = past_year == out["inventory_year"] - lag
        delta = out["min_rating"] - past_min
        out[f"min_delta_{lag}y"] = delta.where(valid)
        out[f"has_hist_{lag}y"] = valid.astype(float)
        for component in ("deck_rating", "super_rating", "sub_rating"):
            past_c = g[component].shift(lag)
            out[f"{component}_delta_{lag}y"] = (out[component] - past_c).where(valid)

    # Calendar years since last change in min_rating (within observed history).
    prev_min = g["min_rating"].shift(1)
    changed = out["min_rating"].ne(prev_min) | prev_min.isna()
    # Year of last change: forward-fill the inventory_year at change points.
    change_year = out["inventory_year"].where(changed)
    change_year = change_year.groupby(out["bridge_id"]).ffill()
    out["years_since_rating_change"] = (
        out["inventory_year"] - change_year
    ).astype(float)

    year_built = _to_float(out[COL_YEAR_BUILT])
    out["bridge_age"] = out["inventory_year"] - year_built
    out.loc[out["bridge_age"] < 0, "bridge_age"] = np.nan
    out.loc[out["bridge_age"] > 250, "bridge_age"] = np.nan

    out["adt"] = _to_float(out[COL_ADT])
    out["log_adt"] = np.log1p(out["adt"].clip(lower=0))

    out["structure_kind"] = out[COL_STRUCTURE_KIND].map(normalize_code)
    out["structure_type"] = out[COL_STRUCTURE_TYPE].map(normalize_code)
    out["structure_kind_type"] = out["structure_kind"] + "_" + out["structure_type"]

    # Explicitly never use bridge_id as a model feature.
    return out


FEATURE_COLUMNS: tuple[str, ...] = (
    "deck_rating",
    "super_rating",
    "sub_rating",
    "min_rating",
    "min_delta_1y",
    "min_delta_2y",
    "min_delta_3y",
    "has_hist_1y",
    "has_hist_2y",
    "has_hist_3y",
    "deck_rating_delta_1y",
    "super_rating_delta_1y",
    "sub_rating_delta_1y",
    "years_since_rating_change",
    "bridge_age",
    "log_adt",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("structure_kind", "structure_type")

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "current_ratings": (
        "deck_rating",
        "super_rating",
        "sub_rating",
        "min_rating",
    ),
    "history": (
        "min_delta_1y",
        "min_delta_2y",
        "min_delta_3y",
        "has_hist_1y",
        "has_hist_2y",
        "has_hist_3y",
        "deck_rating_delta_1y",
        "super_rating_delta_1y",
        "sub_rating_delta_1y",
        "years_since_rating_change",
    ),
    "structure_ops": (
        "bridge_age",
        "log_adt",
        "structure_kind",
        "structure_type",
    ),
}
