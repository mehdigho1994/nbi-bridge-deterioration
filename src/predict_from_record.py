"""Score a single bridge history JSON (BridgeWise integration sketch).

Returns an uncalibrated ranking score by default. Do not treat uncalibrated
scores as reliable event probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import EXAMPLES, LEGACY_DECISION_THRESHOLD, RESULTS
from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, add_features
from src.targets import attach_targets

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
    return {ALIAS_TO_NBI.get(k, k): v for k, v in rec.items()}


def history_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
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
        merged = _normalize_record({**bridge_meta, **insp})
        if "inventory_year" not in merged and "year" in merged:
            merged["inventory_year"] = merged["year"]
        rows.append(merged)

    df = pd.DataFrame(rows)
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
    df["inventory_year"] = pd.to_numeric(df["inventory_year"], errors="coerce")
    if df["inventory_year"].isna().any():
        raise ValueError("Each inspection needs an inventory_year (or year).")
    df["inventory_year"] = df["inventory_year"].astype(int)
    return df.sort_values("inventory_year")


def score_history(
    payload: dict[str, Any],
    model_path: Path | None = None,
    threshold: float = LEGACY_DECISION_THRESHOLD,
) -> dict[str, Any]:
    model_path = model_path or (RESULTS / "hgb_model.joblib")
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run: python -m src.train_eval --core-only"
        )
    bundle = joblib.load(model_path)
    model = bundle["model"]

    raw = history_to_frame(payload)
    with_targets, _ = attach_targets(raw)
    featured = add_features(with_targets)
    latest = featured.sort_values("inventory_year").iloc[[-1]].copy()
    x_cols = list(FEATURE_COLUMNS) + list(CATEGORICAL_FEATURES)
    for c in CATEGORICAL_FEATURES:
        latest[c] = latest[c].fillna("").astype(str)

    score = float(model.predict_proba(latest[x_cols])[:, 1][0])
    return {
        "bridge_id": str(latest["bridge_id"].iloc[0]),
        "inventory_year": int(latest["inventory_year"].iloc[0]),
        "min_rating": (
            None
            if pd.isna(latest["min_rating"].iloc[0])
            else float(latest["min_rating"].iloc[0])
        ),
        "uncalibrated_ranking_score": score,
        "predict_drop_at_legacy_threshold": bool(score >= threshold),
        "legacy_threshold": threshold,
        "score_type": bundle.get("score_type", "uncalibrated_ranking_score"),
        "model_split": bundle.get("split"),
        "disclaimer": bundle.get(
            "disclaimer",
            "Uncalibrated scores are for ranking only, not reliable probabilities.",
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict drop score from JSON.")
    parser.add_argument(
        "json_path",
        nargs="?",
        default=str(EXAMPLES / "bridge_history_example.json"),
    )
    parser.add_argument("--model", default=str(RESULTS / "hgb_model.joblib"))
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    print(json.dumps(score_history(payload, model_path=Path(args.model)), indent=2))


if __name__ == "__main__":
    main()
