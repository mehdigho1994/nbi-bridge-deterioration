"""Score a single bridge history JSON (inspection-system integration sketch).

Input schema matches docs/field_mapping.md placeholders resolved to NBI-like
keys (or the mapped aliases listed below).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import DECISION_THRESHOLD, EXAMPLES, RESULTS
from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, add_features
from src.targets import attach_targets

# Accept either NBI column names or integration aliases from field_mapping.md.
ALIAS_TO_NBI = {
    "state_code": "STATE_CODE_001",
    "structure_number": "STRUCTURE_NUMBER_008",
    "year_built": "YEAR_BUILT_027",
    "adt": "ADT_029",
    "structure_kind": "STRUCTURE_KIND_043A",
    "structure_type": "STRUCTURE_TYPE_043B",
    "deck_condition": "DECK_COND_058",
    "superstructure_condition": "SUPERSTRUCTURE_COND_059",
    "substructure_condition": "SUBSTRUCTURE_COND_060",
    "inspection_date": "DATE_OF_INSPECT_090",
    # Direct NBI names pass through:
    "STATE_CODE_001": "STATE_CODE_001",
    "STRUCTURE_NUMBER_008": "STRUCTURE_NUMBER_008",
    "YEAR_BUILT_027": "YEAR_BUILT_027",
    "ADT_029": "ADT_029",
    "STRUCTURE_KIND_043A": "STRUCTURE_KIND_043A",
    "STRUCTURE_TYPE_043B": "STRUCTURE_TYPE_043B",
    "DECK_COND_058": "DECK_COND_058",
    "SUPERSTRUCTURE_COND_059": "SUPERSTRUCTURE_COND_059",
    "SUBSTRUCTURE_COND_060": "SUBSTRUCTURE_COND_060",
    "DATE_OF_INSPECT_090": "DATE_OF_INSPECT_090",
}


def _normalize_record(rec: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in rec.items():
        key = ALIAS_TO_NBI.get(k, k)
        out[key] = v
    return out


def history_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert JSON payload to a mini panel for one bridge.

    Expected shape:
    {
      "bridge_id": "51|...",
      "state_code": "51",
      "structure_number": "...",
      "year_built": 1985,
      "structure_kind": "3",
      "structure_type": "02",
      "inspections": [
         {"inventory_year": 2022, "deck_condition": "7", ... , "adt": 1200},
         ...
      ]
    }
    """
    bridge_meta = {
        k: payload.get(k)
        for k in (
            "bridge_id",
            "state_code",
            "structure_number",
            "year_built",
            "structure_kind",
            "structure_type",
        )
        if k in payload
    }
    inspections = payload.get("inspections") or payload.get("history")
    if not inspections:
        raise ValueError("JSON must contain an 'inspections' (or 'history') list.")

    rows = []
    for insp in inspections:
        merged = {**bridge_meta, **insp}
        merged = _normalize_record(merged)
        if "inventory_year" not in merged and "year" in merged:
            merged["inventory_year"] = merged["year"]
        rows.append(merged)

    df = pd.DataFrame(rows)
    # Ensure required NBI-like columns exist.
    for col in (
        "STATE_CODE_001",
        "STRUCTURE_NUMBER_008",
        "YEAR_BUILT_027",
        "ADT_029",
        "STRUCTURE_KIND_043A",
        "STRUCTURE_TYPE_043B",
        "DECK_COND_058",
        "SUPERSTRUCTURE_COND_059",
        "SUBSTRUCTURE_COND_060",
        "DATE_OF_INSPECT_090",
    ):
        if col not in df.columns:
            df[col] = np.nan

    if "bridge_id" not in df.columns or df["bridge_id"].isna().all():
        df["bridge_id"] = (
            df["STATE_CODE_001"].astype(str).str.strip()
            + "|"
            + df["STRUCTURE_NUMBER_008"].astype(str).str.strip()
        )
    df["inventory_year"] = pd.to_numeric(df["inventory_year"], errors="coerce").astype(
        "Int64"
    )
    if df["inventory_year"].isna().any():
        raise ValueError("Each inspection needs an inventory_year (or year).")
    df["inventory_year"] = df["inventory_year"].astype(int)
    return df.sort_values("inventory_year")


def score_history(
    payload: dict[str, Any],
    model_path: Path | None = None,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, Any]:
    """Return drop probability for the latest inspection year in the history."""
    model_path = model_path or (RESULTS / "hgb_model.joblib")
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run run_all.py first."
        )
    bundle = joblib.load(model_path)
    model = bundle["model"]

    raw = history_to_frame(payload)
    with_targets, _ = attach_targets(raw)
    featured = add_features(with_targets)
    latest = featured.sort_values("inventory_year").iloc[[-1]]
    x_cols = list(FEATURE_COLUMNS) + list(CATEGORICAL_FEATURES)
    # Fill missing categoricals for encoder robustness.
    for c in CATEGORICAL_FEATURES:
        latest[c] = latest[c].fillna("").astype(str)

    proba = float(model.predict_proba(latest[x_cols])[:, 1][0])
    return {
        "bridge_id": str(latest["bridge_id"].iloc[0]),
        "inventory_year": int(latest["inventory_year"].iloc[0]),
        "min_rating": (
            None
            if pd.isna(latest["min_rating"].iloc[0])
            else float(latest["min_rating"].iloc[0])
        ),
        "p_drop_next_year": proba,
        "predict_drop": bool(proba >= threshold),
        "threshold": threshold,
        "model_split": bundle.get("split"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict drop probability from JSON.")
    parser.add_argument(
        "json_path",
        nargs="?",
        default=str(EXAMPLES / "bridge_history_example.json"),
        help="Path to bridge history JSON.",
    )
    parser.add_argument(
        "--model",
        default=str(RESULTS / "hgb_model.joblib"),
        help="Path to joblib model bundle.",
    )
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    result = score_history(payload, model_path=Path(args.model))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
