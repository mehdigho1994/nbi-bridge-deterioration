"""Central configuration for the NBI deterioration research prototype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------
STATE_ABBR = "VA"
STATE_CODE = "51"
YEARS: tuple[int, ...] = tuple(range(2016, 2026))  # 2016–2025 inclusive

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
EXAMPLES = ROOT / "examples"

# ---------------------------------------------------------------------------
# Verified NBI delimited column names (from FHWA file headers)
# ---------------------------------------------------------------------------
COL_STATE = "STATE_CODE_001"
COL_STRUCTURE = "STRUCTURE_NUMBER_008"
COL_RECORD_TYPE = "RECORD_TYPE_005A"
COL_YEAR_BUILT = "YEAR_BUILT_027"
COL_ADT = "ADT_029"
COL_STRUCTURE_KIND = "STRUCTURE_KIND_043A"
COL_STRUCTURE_TYPE = "STRUCTURE_TYPE_043B"
COL_DECK = "DECK_COND_058"
COL_SUPER = "SUPERSTRUCTURE_COND_059"
COL_SUB = "SUBSTRUCTURE_COND_060"
COL_INSPECT_DATE = "DATE_OF_INSPECT_090"
COL_INSPECT_FREQ = "INSPECT_FREQ_MONTHS_091"

REQUIRED_COLUMNS: tuple[str, ...] = (
    COL_STATE,
    COL_STRUCTURE,
    COL_RECORD_TYPE,
    COL_YEAR_BUILT,
    COL_ADT,
    COL_STRUCTURE_KIND,
    COL_STRUCTURE_TYPE,
    COL_DECK,
    COL_SUPER,
    COL_SUB,
    COL_INSPECT_DATE,
)

RECORD_TYPE_ON_STRUCTURE = "1"
CONDITION_COLUMNS: tuple[str, ...] = (COL_DECK, COL_SUPER, COL_SUB)
VALID_NUMERIC_RATINGS = frozenset(str(i) for i in range(10))
NOT_APPLICABLE = "N"

# ---------------------------------------------------------------------------
# Seeds / thresholds
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
EVAL_SEEDS: tuple[int, ...] = (42, 123, 456)
# Preserved original reporting threshold (not recommended for balanced models).
LEGACY_DECISION_THRESHOLD = 0.10
MAX_LAG_YEARS = 3

# HGB iteration grid for chronological validation (early_stopping disabled).
HGB_MAX_ITER_GRID: tuple[int, ...] = (50, 100, 150, 200)

FHWA_DELIMITED_URL = (
    "https://www.fhwa.dot.gov/bridge/nbi/{year}/delimited/{state}{yy}.txt"
)


@dataclass(frozen=True)
class TemporalSplitSpec:
    """Chronological split with explicit feature / label / cutoff semantics.

    Definitions
    -----------
    feature_year:
        Inventory year of the covariates (ratings and attributes at time t).
    label_observation_year:
        Year whose ratings define the outcome for a row with feature_year t
        (here always t+1 for inventory tasks).
    train_max_feature_year:
        Maximum feature year allowed in the fitting set.
    validation_feature_year:
        Held-out chronological year used for model selection / calibration /
        threshold selection (never the final test year).
    test_feature_year:
        Final evaluation feature year.
    protocol:
        ``standard`` — one-step panel (train labels may use the calendar year
        that later appears as the test feature year).
        ``strict`` — every training label observation year is strictly before
        the test feature year.
    """

    name: str
    protocol: str  # "standard" | "strict"
    train_max_feature_year: int
    validation_feature_year: int | None
    test_feature_year: int
    target_col: str = "y_drop"


def split_slug(spec: TemporalSplitSpec) -> str:
    return spec.name


# Standard (baseline) protocol — retained for continuity with original study.
# Example: train features <= 2022 (labels through 2023); test features 2023
# (labels 2024). Training labels therefore use calendar 2023, which equals the
# test feature year.
STANDARD_SPLITS: tuple[TemporalSplitSpec, ...] = (
    TemporalSplitSpec(
        name="standard_train_to_2022_val_2022_test_2023",
        protocol="standard",
        train_max_feature_year=2021,  # fit on <=2021
        validation_feature_year=2022,  # select/calibrate on 2022
        test_feature_year=2023,
    ),
    TemporalSplitSpec(
        name="standard_train_to_2023_val_2023_test_2024",
        protocol="standard",
        train_max_feature_year=2022,
        validation_feature_year=2023,
        test_feature_year=2024,
    ),
)

# Strict protocol — max train label year < test feature year.
# Test feature 2023 ⇒ train features <= 2021 (labels through 2022).
STRICT_SPLITS: tuple[TemporalSplitSpec, ...] = (
    TemporalSplitSpec(
        name="strict_train_to_2020_val_2021_test_2023",
        protocol="strict",
        train_max_feature_year=2020,
        validation_feature_year=2021,
        test_feature_year=2023,
    ),
    TemporalSplitSpec(
        name="strict_train_to_2021_val_2022_test_2024",
        protocol="strict",
        train_max_feature_year=2021,
        validation_feature_year=2022,
        test_feature_year=2024,
    ),
)

# Primary + supplementary binary targets
PRIMARY_TARGET = "y_drop"
ALT_TARGETS: tuple[str, ...] = (
    "y_any_component_drop",
    "y_deck_drop",
    "y_super_drop",
    "y_sub_drop",
)

TOP_K_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.10)
