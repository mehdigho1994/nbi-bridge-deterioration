"""Load and lightly clean FHWA NBI delimited yearly files.

Column names are verified against the file header at load time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import (
    COL_RECORD_TYPE,
    COL_STATE,
    COL_STRUCTURE,
    CONDITION_COLUMNS,
    DATA_PROCESSED,
    DATA_RAW,
    NOT_APPLICABLE,
    RECORD_TYPE_ON_STRUCTURE,
    REQUIRED_COLUMNS,
    STATE_ABBR,
    STATE_CODE,
    VALID_NUMERIC_RATINGS,
    YEARS,
)
from src.download_data import expected_filename


def normalize_code(value: object) -> str:
    """Strip whitespace; empty -> ''."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


# Backwards-compatible alias used in older drafts / tests.
_normalize_code = normalize_code


def parse_condition(value: object) -> float | None:
    """Parse a condition code to float 0–9, or None for N / missing / invalid.

    Documented policy: 'N' (not applicable) and blanks are treated as missing
    and excluded from the primary min-rating target (see build_targets).
    """
    code = _normalize_code(value).upper()
    if code == "" or code == NOT_APPLICABLE:
        return None
    if code in VALID_NUMERIC_RATINGS:
        return float(code)
    # Rare non-standard tokens → missing
    return None


def verify_columns(columns: list[str], path: Path) -> None:
    """Raise if required columns are absent; print discovered key names."""
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise KeyError(
            f"{path.name}: missing required columns {missing}. "
            f"Found columns (first 20): {columns[:20]}"
        )


def load_year_file(path: Path, inventory_year: int) -> pd.DataFrame:
    """Load one delimited NBI file and keep on-structure highway records."""
    # quotechar="'" matches FHWA delimited convention.
    df = pd.read_csv(
        path,
        dtype=str,
        quotechar="'",
        low_memory=False,
    )
    verify_columns(list(df.columns), path)

    # Keep only primary on-structure records (avoid under-route duplicates).
    df = df[df[COL_RECORD_TYPE].map(_normalize_code) == RECORD_TYPE_ON_STRUCTURE].copy()
    df[COL_STATE] = df[COL_STATE].map(_normalize_code)
    df[COL_STRUCTURE] = df[COL_STRUCTURE].map(_normalize_code)
    # Restrict to configured state code when present.
    df = df[df[COL_STATE] == STATE_CODE].copy()

    df["inventory_year"] = inventory_year
    df["bridge_id"] = df[COL_STATE] + "|" + df[COL_STRUCTURE]
    return df


def load_raw_panel(
    years: tuple[int, ...] = YEARS,
    state: str = STATE_ABBR,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Concatenate yearly files into a long panel (one row per bridge-year)."""
    raw_dir = raw_dir or DATA_RAW
    frames: list[pd.DataFrame] = []
    schema_report: dict[str, list[str]] = {}

    for year in years:
        path = raw_dir / expected_filename(year, state)
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Missing raw file: {path}. Run: python -m src.download_data"
            )
        year_df = load_year_file(path, year)
        schema_report[str(year)] = list(year_df.columns)
        frames.append(year_df)
        print(
            f"[load] {path.name}: {len(year_df):,} on-structure rows "
            f"(state={STATE_CODE})"
        )

    panel = pd.concat(frames, ignore_index=True, sort=False)
    # Persist verified schema for transparency.
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    schema_path = DATA_PROCESSED / "verified_columns.json"
    schema_path.write_text(json.dumps(schema_report, indent=2), encoding="utf-8")
    print(f"[schema] wrote verified column names -> {schema_path}")
    return panel


def condition_summary(df: pd.DataFrame) -> dict[str, int]:
    """Count N / missing / numeric for condition columns (for documentation)."""
    stats: dict[str, int] = {}
    for col in CONDITION_COLUMNS:
        raw = df[col].map(_normalize_code).str.upper()
        stats[f"{col}_N"] = int((raw == NOT_APPLICABLE).sum())
        stats[f"{col}_blank"] = int((raw == "").sum())
        stats[f"{col}_numeric"] = int(raw.isin(VALID_NUMERIC_RATINGS).sum())
        stats[f"{col}_other"] = int(
            (~raw.isin(VALID_NUMERIC_RATINGS | {NOT_APPLICABLE, ""})).sum()
        )
    return stats
