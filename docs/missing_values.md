# Missing and `N` value policy

## Condition ratings (items 58, 59, 60)

| Raw value | Meaning | Handling |
| --- | --- | --- |
| `0`–`9` | Valid condition codes | Parsed as floats; used in min rating and features |
| `N` | Not applicable (e.g. culverts for deck/super/sub) | Treated as **missing** |
| blank / NaN | Not reported | Treated as **missing** |
| other tokens | Invalid / unexpected | Treated as **missing** |

### Primary binary target

`min_rating = min(deck, super, sub)` is defined only when **all three** components
are numeric 0–9.

If `min_rating` is undefined at year \(t\) **or** year \(t+1\`, the primary target
`y_drop` is undefined and the bridge-year is **excluded** from training/evaluation
for the binary task.

Exclusion counts are written to `data/processed/panel_meta.json`:

- `n_excluded_current_N_or_missing`
- `n_excluded_next_N_or_missing`

### Features

HistGradientBoosting natively supports NaN numeric features. Logistic regression
uses median / most-frequent imputation inside its pipeline. We do **not** impute
condition ratings with a sentinel like `-1` for the primary target path.

## ADT and year built

Non-numeric values become NaN. Implausible ages (&lt;0 or &gt;250) are set to NaN.
