"""Project configuration: state, years, paths, seeds, evaluation settings."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Scope (chosen defaults — documented in README)
# ---------------------------------------------------------------------------
STATE_ABBR = "VA"
STATE_CODE = "51"  # Virginia FIPS / NBI state code
# Latest ~10 consecutive ASCII Coding-Guide years available from FHWA.
YEARS: tuple[int, ...] = tuple(range(2016, 2026))  # 2016–2025 inclusive

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
EXAMPLES = ROOT / "examples"

# ---------------------------------------------------------------------------
# Verified column names from FHWA delimited NBI files (header row present).
# Confirmed on VA25.txt / FL25.txt / IA18.txt downloads — do not invent names.
# ---------------------------------------------------------------------------
COL_STATE = "STATE_CODE_001"
COL_STRUCTURE = "STRUCTURE_NUMBER_008"
COL_RECORD_TYPE = "RECORD_TYPE_005A"
COL_YEAR_BUILT = "YEAR_BUILT_027"
COL_ADT = "ADT_029"
COL_STRUCTURE_KIND = "STRUCTURE_KIND_043A"  # material
COL_STRUCTURE_TYPE = "STRUCTURE_TYPE_043B"  # design/construction
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

# Keep only on-structure inventory records (one row per bridge).
RECORD_TYPE_ON_STRUCTURE = "1"

# Condition codes: 0–9 numeric; 'N' = not applicable.
CONDITION_COLUMNS: tuple[str, ...] = (COL_DECK, COL_SUPER, COL_SUB)
VALID_NUMERIC_RATINGS = frozenset(str(i) for i in range(10))
NOT_APPLICABLE = "N"

# ---------------------------------------------------------------------------
# Modelling / evaluation
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
EVAL_SEEDS: tuple[int, ...] = (42, 123, 456)
# Decision threshold for precision/recall / confusion matrix (stated explicitly).
DECISION_THRESHOLD = 0.10

# Rolling time splits: train on feature-years <= T, test on T+1.
# Feature years that can have a next-year target depend on data years.
# With YEARS=2016..2025, next-year targets exist for feature years 2016..2024.
TIME_SPLITS: tuple[tuple[int, int], ...] = (
    (2022, 2023),  # train feature-years <= 2022, test 2023
    (2023, 2024),  # train feature-years <= 2023, test 2024
)

# Windows-safe token for filenames (avoid '<' / '>' / '|').
def split_slug(train_max_year: int, test_year: int) -> str:
    return f"train_to_{train_max_year}_test_{test_year}"

# Minimum prior years needed for lag features (we still allow shorter history).
MAX_LAG_YEARS = 3

FHWA_DELIMITED_URL = (
    "https://www.fhwa.dot.gov/bridge/nbi/{year}/delimited/{state}{yy}.txt"
)
