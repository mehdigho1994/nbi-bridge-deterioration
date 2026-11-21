"""Feature engineering using only current and past information per bridge.

No future rows are used. Lag features are computed with groupby.shift(positive).
"""

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
    """Add model features to a panel that already has rating columns.

    Required input columns include:
      bridge_id, inventory_year, deck/super/sub/min_rating, YEAR_BUILT, ADT,
      STRUCTURE_KIND, STRUCTURE_TYPE.
    """
    out = df.sort_values(["bridge_id", "inventory_year"]).copy()
    g = out.groupby("bridge_id", sort=False)

    # Current ratings (already present from targets); ensure numeric.
    for col in ("deck_rating", "super_rating", "sub_rating", "min_rating"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Rating changes over previous 1–3 inventory years (past only).
    for lag in range(1, MAX_LAG_YEARS + 1):
        past_min = g["min_rating"].shift(lag)
        past_year = g["inventory_year"].shift(lag)
        # Only use lag if the inventory year gap matches the lag.
        valid = past_year == out["inventory_year"] - lag
        delta = out["min_rating"] - past_min
        out[f"min_delta_{lag}y"] = delta.where(valid)
        out[f"has_hist_{lag}y"] = valid.astype(float)

        for component in ("deck_rating", "super_rating", "sub_rating"):
            past_c = g[component].shift(lag)
            out[f"{component}_delta_{lag}y"] = (out[component] - past_c).where(valid)

    # Years since last change in min rating (within observed history).
    changed = out["min_rating"] != g["min_rating"].shift(1)
    # First row of each bridge: treat as "changed" so counter resets.
    first = g.cumcount() == 0
    changed = changed.fillna(True) | first
    # Running count of years since change.
    # Assign a group id that increments at each change, then count within group.
    change_id = changed.groupby(out["bridge_id"]).cumsum()
    out["years_since_rating_change"] = (
        out.groupby(["bridge_id", change_id], sort=False).cumcount()
    ).astype(float)

    # Bridge age at inventory year.
    year_built = _to_float(out[COL_YEAR_BUILT])
    out["bridge_age"] = out["inventory_year"] - year_built
    out.loc[out["bridge_age"] < 0, "bridge_age"] = np.nan
    out.loc[out["bridge_age"] > 250, "bridge_age"] = np.nan

    # ADT (log1p for scale); missing stays NaN (HGB handles NaN).
    out["adt"] = _to_float(out[COL_ADT])
    out["log_adt"] = np.log1p(out["adt"].clip(lower=0))

    # Structure material / type as categorical codes (string categories).
    out["structure_kind"] = out[COL_STRUCTURE_KIND].map(normalize_code)
    out["structure_type"] = out[COL_STRUCTURE_TYPE].map(normalize_code)
    # Combined code for one-hot / categorical encoding downstream.
    out["structure_kind_type"] = out["structure_kind"] + "_" + out["structure_type"]

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
