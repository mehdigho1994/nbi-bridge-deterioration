# Field mapping: FHWA NBI → our bridge inspection system

This table maps the NBI Coding Guide items used by this prototype to fields in
**our** inspection system. Replace each `[OUR_FIELD_NAME]` placeholder with the
real column/API name once integration begins.

Verified NBI delimited column names come from FHWA yearly files (e.g. `VA25.txt`).

| Concept | NBI item | NBI delimited column | Our system field | Notes |
| --- | --- | --- | --- | --- |
| State / agency code | 1 | `STATE_CODE_001` | `[OUR_FIELD_NAME]` | Part of stable bridge ID |
| Structure number | 8 | `STRUCTURE_NUMBER_008` | `[OUR_FIELD_NAME]` | Part of stable bridge ID |
| Bridge ID | 1+8 | `state\|structure` | `[OUR_FIELD_NAME]` | Must be stable across inspections |
| Year built | 27 | `YEAR_BUILT_027` | `[OUR_FIELD_NAME]` | Used for bridge age |
| Average daily traffic | 29 | `ADT_029` | `[OUR_FIELD_NAME]` | Traffic exposure proxy |
| Structure material | 43A | `STRUCTURE_KIND_043A` | `[OUR_FIELD_NAME]` | Kind of material/design |
| Structure type | 43B | `STRUCTURE_TYPE_043B` | `[OUR_FIELD_NAME]` | Type of design/construction |
| Deck condition | 58 | `DECK_COND_058` | `[OUR_FIELD_NAME]` | Codes 0–9; `N` = not applicable |
| Superstructure condition | 59 | `SUPERSTRUCTURE_COND_059` | `[OUR_FIELD_NAME]` | Codes 0–9; `N` = not applicable |
| Substructure condition | 60 | `SUBSTRUCTURE_COND_060` | `[OUR_FIELD_NAME]` | Codes 0–9; `N` = not applicable |
| Inspection date | 90 | `DATE_OF_INSPECT_090` | `[OUR_FIELD_NAME]` | NBI is MMYY; we also use inventory year |
| Inspection frequency | 91 | `INSPECT_FREQ_MONTHS_091` | `[OUR_FIELD_NAME]` | Optional; often ~24 months |

## Target used by this module

For each bridge-year with a defined rating:

\[
y = \mathbf{1}\big[\min(58,59,60)_{t+1} \le \min(58,59,60)_t - 1\big]
\]

Rows where any of items 58–60 is `N` or missing on year \(t\) or \(t+1\) are
**excluded** from the primary binary target (counts are logged in
`data/processed/panel_meta.json`).

## Extra data our system would need to collect

For this predictive-maintenance style module to transfer beyond the US NBI demo:

1. **Repeated inspections over time** for the same bridge ID (a panel), not one-off surveys.
2. **A consistent ordinal rating scale** across years (same codes, same component definitions). If our scale differs from NBI 0–9, retrain — do not assume code equivalence.
3. **Stable asset identifiers** so history can be joined without silent identity breaks after renumbering/rebuilds.
4. **Inspection timestamps** (or at least inspection year) to build chronological features and time-based evaluation.
5. **Basic static attributes**: year built (or age), traffic volume, structure material/type.
6. Ideally: explicit flags when a rating is “not applicable” vs truly missing, and records of major rehab/replacement (to avoid treating rebuilds as mysterious rating jumps).

Sensor streams, images, or element-level quantities are **not** required for this prototype, but would be natural extensions once the tabular panel pipeline is wired in.

## Integration sketch

`src/predict_from_record.py` accepts a JSON history for one bridge (see
`examples/bridge_history_example.json`) and returns `p_drop_next_year`. Map our
API payload into that JSON using the table above, then call the scorer after each
new inspection is stored.
