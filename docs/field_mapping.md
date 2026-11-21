# Conceptual mapping: NBI fields ↔ BridgeWise inspection concepts

This mapping is **conceptual**. It does not expose the municipal database schema,
API routes, table names, or internal identifiers of the Karaj BridgeWise system.

The public repository contains **no municipal inspection records**. Retraining,
validation, and calibration on local data are required before any operational use.

| Concept | NBI item | NBI delimited column | BridgeWise conceptual field | Notes |
| --- | --- | --- | --- | --- |
| Jurisdiction / agency code | 1 | `STATE_CODE_001` | Agency or network code | Part of stable asset ID |
| Structure number | 8 | `STRUCTURE_NUMBER_008` | Bridge asset identifier | Must remain stable across inspections |
| Bridge ID | 1+8 | `state\|structure` | Canonical bridge key | Join key for longitudinal history |
| Year built | 27 | `YEAR_BUILT_027` | Construction / opening year | Used for age |
| Average daily traffic | 29 | `ADT_029` | Traffic volume estimate | Exposure proxy |
| Structure material | 43A | `STRUCTURE_KIND_043A` | Primary material class | Categorical |
| Structure type | 43B | `STRUCTURE_TYPE_043B` | Structural system / design type | Categorical |
| Deck condition | 58 | `DECK_COND_058` | Deck condition rating | Ordinal; define local scale explicitly |
| Superstructure condition | 59 | `SUPERSTRUCTURE_COND_059` | Superstructure condition rating | Ordinal |
| Substructure condition | 60 | `SUBSTRUCTURE_COND_060` | Substructure condition rating | Ordinal |
| Inspection date | 90 | `DATE_OF_INSPECT_090` | Field inspection timestamp | NBI is MMYY; BridgeWise should store full dates |
| Inspection frequency | 91 | `INSPECT_FREQ_MONTHS_091` | Planned inspection interval | Optional |

## Primary research target (NBI prototype)

\[
y_t = \mathbf{1}\big[\min(58,59,60)_{t+1} \le \min(58,59,60)_t - 1\big]
\]

Rows with `N` / missing components are excluded from this primary target.

## Integration prototype

`src/predict_from_record.py` accepts an illustrative JSON history and returns an
**uncalibrated ranking score** (and optionally a calibrated probability if a
calibrator bundle is provided). Deployment requires:

1. Local panel construction from BridgeWise histories  
2. Retraining and chronological validation  
3. Probability calibration on a held-out chronological period  
4. Policy decisions for top‑k inspection / maintenance budgets  

## Extra data BridgeWise should retain for this modelling pattern

1. Repeated inspections with a **stable bridge ID**  
2. A **consistent ordinal rating scale** over time  
3. Explicit not-applicable vs missing flags  
4. Inspection timestamps (not only inventory publish years)  
5. Basic attributes: age/year built, traffic, structure type/material  
6. Major rehab / replacement events (to avoid treating rebuilds as mystery jumps)
