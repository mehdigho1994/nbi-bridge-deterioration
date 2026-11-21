"""Inspection-to-inspection deterioration diagnostic (supplementary study)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.baselines import MarkovDropModel
from src.config import COL_INSPECT_DATE, RESULTS
from src.evaluate import binary_metrics, topk_metrics, write_json
from src.features import add_features
from src.load_nbi import normalize_code
from src.models import fit_hgb, predict_hgb, select_hgb_max_iter
from src.targets import primary_drop_target


def parse_inspect_mmyy(value: object) -> tuple[int | None, int | None]:
    """Parse NBI item 90 (MMYY) → (month, full_year)."""
    s = normalize_code(value)
    if len(s) != 4 or not s.isdigit():
        return None, None
    month = int(s[:2])
    yy = int(s[2:])
    if month < 1 or month > 12:
        return None, None
    year = 2000 + yy if yy <= 50 else 1900 + yy
    return month, year


def inspect_sort_key(value: object) -> tuple[int, int]:
    month, year = parse_inspect_mmyy(value)
    if year is None or month is None:
        return (9999, 99)
    return (year, month)


def carry_forward_report(panel: pd.DataFrame) -> dict[str, Any]:
    p = panel.sort_values(["bridge_id", "inventory_year"]).copy()
    p["cur_inspect"] = p[COL_INSPECT_DATE].map(normalize_code)
    p["next_inspect"] = (
        p.groupby("bridge_id")[COL_INSPECT_DATE].shift(-1).map(normalize_code)
    )
    pairs = p[p["next_year"] == p["inventory_year"] + 1].copy()
    pairs["unchanged"] = pairs["cur_inspect"] == pairs["next_inspect"]
    valid = (pairs["cur_inspect"] != "") & (pairs["next_inspect"] != "")
    vu = pairs[valid]

    def _rate(df: pd.DataFrame) -> dict[str, Any]:
        d = df[df["y_drop"].notna()]
        return {
            "n_defined": int(len(d)),
            "n_positive": int(d["y_drop"].sum()) if len(d) else 0,
            "drop_rate": float(d["y_drop"].mean()) if len(d) else float("nan"),
        }

    return {
        "n_consecutive_inventory_pairs": int(len(pairs)),
        "n_with_both_dates": int(valid.sum()),
        "fraction_unchanged_inspect_date": (
            float(vu["unchanged"].mean()) if len(vu) else float("nan")
        ),
        "unchanged_inspect": _rate(vu[vu["unchanged"]]),
        "changed_inspect": _rate(vu[~vu["unchanged"]]),
        "interpretation": (
            "Unchanged DATE_OF_INSPECT_090 across consecutive inventory years "
            "indicates carried-forward inspection records rather than a new "
            "field assessment. Models trained on inventory-year targets may "
            "partly learn update/carry-forward patterns."
        ),
    }


def build_inspection_event_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per bridge per distinct inspection date (latest inventory snapshot)."""
    df = panel.copy()
    df["inspect_key"] = df[COL_INSPECT_DATE].map(normalize_code)
    df = df[df["inspect_key"] != ""].copy()
    df["_sort"] = df["inspect_key"].map(inspect_sort_key)
    df = df.sort_values(["bridge_id", "_sort", "inventory_year"])
    events = df.groupby(["bridge_id", "inspect_key"], as_index=False).tail(1)
    events = events.sort_values(["bridge_id", "_sort", "inventory_year"]).copy()

    g = events.groupby("bridge_id", sort=False)
    events["next_inspect_key"] = g["inspect_key"].shift(-1)
    events["next_min_rating"] = g["min_rating"].shift(-1)
    events["next_inventory_year"] = g["inventory_year"].shift(-1)

    def _elapsed(a: object, b: object) -> float:
        m0, y0 = parse_inspect_mmyy(a)
        m1, y1 = parse_inspect_mmyy(b)
        if None in (m0, y0, m1, y1):
            return float("nan")
        return (y1 + (m1 - 1) / 12.0) - (y0 + (m0 - 1) / 12.0)

    events["years_to_next_inspection"] = [
        _elapsed(a, b)
        for a, b in zip(events["inspect_key"], events["next_inspect_key"], strict=True)
    ]
    # Distinct consecutive inspections only (exclude identical keys; shift already next)
    events["y_insp_drop"] = [
        primary_drop_target(c, n)
        for c, n in zip(events["min_rating"], events["next_min_rating"], strict=True)
    ]
    return events


def run_inspection_to_inspection_study(
    panel: pd.DataFrame,
    *,
    train_max_inventory_year: int = 2021,
    test_min_inventory_year: int = 2023,
) -> dict[str, Any]:
    """Markov + unweighted HGB on inspection-event drops (supplementary)."""
    carry = carry_forward_report(panel)
    events = build_inspection_event_panel(panel)
    featured = add_features(events)
    featured["y_drop"] = featured["y_insp_drop"]
    labeled = featured[featured["y_drop"].notna()].copy()

    train = labeled[
        (labeled["inventory_year"] <= train_max_inventory_year)
        & (
            labeled["next_inventory_year"].isna()
            | (labeled["next_inventory_year"] < test_min_inventory_year)
        )
    ].copy()
    test = labeled[labeled["inventory_year"] >= test_min_inventory_year].copy()

    report: dict[str, Any] = {
        "study": "inspection_to_inspection",
        "carry_forward": carry,
        "n_inspection_events": int(len(events)),
        "n_labeled_transitions": int(len(labeled)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_positive_rate": float(train["y_drop"].mean()) if len(train) else float("nan"),
        "test_positive_rate": float(test["y_drop"].mean()) if len(test) else float("nan"),
        "mean_years_between_inspections_test": (
            float(test["years_to_next_inspection"].mean()) if len(test) else float("nan")
        ),
        "exclusion_note": (
            "Transitions require numeric min ratings at both inspection events. "
            "years_to_next_inspection is a descriptor only and is not used as a feature."
        ),
    }

    if train.empty or test.empty:
        report["skipped"] = "insufficient train/test after cutoffs"
        write_json(report, RESULTS / "inspection_study.json")
        return report

    y_test = test["y_drop"].astype(int).to_numpy()
    markov = MarkovDropModel().fit(train)
    p_m = markov.predict_proba(test)

    vmax = int(train["inventory_year"].quantile(0.8))
    tr_fit = train[train["inventory_year"] <= vmax]
    tr_val = train[train["inventory_year"] > vmax]
    if tr_fit.empty:
        tr_fit, tr_val = train, train.iloc[0:0]
    best_iter, grid_scores = select_hgb_max_iter(
        tr_fit, tr_val, seed=42, class_weight=None
    )
    fit_df = tr_fit if len(tr_val) else train
    hgb = fit_hgb(fit_df, 42, max_iter=best_iter, class_weight=None)
    p_h = predict_hgb(hgb, test)

    report["models"] = {
        "markov": {
            "metrics": binary_metrics(y_test, p_m),
            "topk": topk_metrics(y_test, p_m, aux=test),
        },
        "hgb_unweighted": {
            "selected_max_iter": best_iter,
            "val_ap_by_iter": grid_scores,
            "metrics": binary_metrics(y_test, p_h),
            "topk": topk_metrics(y_test, p_h, aux=test),
        },
    }
    write_json(report, RESULTS / "inspection_study.json")
    return report
