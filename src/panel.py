"""Build the analysis-ready bridge-year panel from raw NBI files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import (
    COL_ADT,
    COL_DECK,
    COL_INSPECT_DATE,
    COL_STATE,
    COL_STRUCTURE,
    COL_STRUCTURE_KIND,
    COL_STRUCTURE_TYPE,
    COL_SUB,
    COL_SUPER,
    COL_YEAR_BUILT,
    DATA_PROCESSED,
)
from src.features import add_features
from src.load_nbi import condition_summary, load_raw_panel
from src.targets import TargetBuildReport, attach_targets


KEEP_RAW = [
    "bridge_id",
    "inventory_year",
    COL_STATE,
    COL_STRUCTURE,
    COL_YEAR_BUILT,
    COL_ADT,
    COL_STRUCTURE_KIND,
    COL_STRUCTURE_TYPE,
    COL_DECK,
    COL_SUPER,
    COL_SUB,
    COL_INSPECT_DATE,
]


def build_panel(raw: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Load raw NBI (if needed), attach targets/features, write artifacts."""
    if raw is None:
        raw = load_raw_panel()

    # Deduplicate on bridge-year (rare duplicates if any).
    slim = raw[KEEP_RAW].drop_duplicates(["bridge_id", "inventory_year"], keep="first")
    cond_stats = condition_summary(slim)

    with_targets, report = attach_targets(slim)
    featured = add_features(with_targets)

    meta = {
        "condition_value_counts": cond_stats,
        "target_report": report.__dict__,
        "n_rows": int(len(featured)),
        "n_bridges": int(featured["bridge_id"].nunique()),
        "years": sorted(int(y) for y in featured["inventory_year"].unique()),
    }

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_parquet = DATA_PROCESSED / "panel.parquet"
    out_csv = DATA_PROCESSED / "panel_sample.csv"
    featured.to_parquet(out_parquet, index=False)
    featured.head(200).to_csv(out_csv, index=False)
    (DATA_PROCESSED / "panel_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(
        f"[panel] rows={meta['n_rows']:,} bridges={meta['n_bridges']:,} "
        f"primary_defined={report.n_primary_defined:,} "
        f"positives={report.n_primary_positive:,} "
        f"({report.n_primary_positive / max(report.n_primary_defined, 1):.2%})"
    )
    print(
        f"[panel] excluded (N/missing) current={report.n_excluded_current_N_or_missing:,} "
        f"next={report.n_excluded_next_N_or_missing:,}"
    )
    print(f"[panel] wrote {out_parquet}")
    return featured, meta


def load_panel(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA_PROCESSED / "panel.parquet")
    return pd.read_parquet(path)


def format_target_report(report: TargetBuildReport | dict) -> str:
    d = report if isinstance(report, dict) else report.__dict__
    return (
        f"bridge-years={d['n_bridge_years']}, with_next_year={d['n_with_next_year']}, "
        f"excluded_current_N/missing={d['n_excluded_current_N_or_missing']}, "
        f"excluded_next_N/missing={d['n_excluded_next_N_or_missing']}, "
        f"primary_defined={d['n_primary_defined']} "
        f"(pos={d['n_primary_positive']}, neg={d['n_primary_negative']})"
    )
