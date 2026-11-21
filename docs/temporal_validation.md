# Temporal validation semantics

## Terms

| Term | Meaning in this project |
| --- | --- |
| Feature year \(t\) | `inventory_year` of the covariate row (ratings and attributes at time \(t\)) |
| Label observation year | Year whose ratings define the outcome; for inventory tasks this is \(t+1\) |
| Training cutoff | Maximum feature year used to **fit** models |
| Validation period | A later chronological feature year used only for hyperparameter selection, threshold selection, and probability calibration |
| Test period | Final chronological feature year used once for reporting |

## Standard protocol (baseline)

Example: test feature year 2023.

- Fit on feature years ≤ 2021 (labels observed through 2022)  
- Validate on feature year 2022 (labels observed in 2023)  
- Test on feature year 2023 (labels observed in 2024)  

Training/validation labels may use calendar years that equal the eventual test
**feature** year (here, validation labels use 2023). This is a common one-step
panel design but is **not** fully label-isolated from the test feature calendar year.

## Strict protocol

Example: test feature year 2023.

- Fit on feature years ≤ 2020 (labels through 2021)  
- Validate on feature year 2021 (labels through 2022)  
- Test on feature year 2023 (labels through 2024)  

Every training and validation **label observation year** is strictly before the
test feature year.

## What is fitted where

| Step | Data used |
| --- | --- |
| Lag features | Past/current rows of the same bridge only |
| Markov tables | Training fit set only |
| Imputer / scaler / one-hot | Training fit set only |
| HGB `max_iter` selection | Chronological validation Average Precision |
| Decision threshold selection | Chronological validation |
| Probability calibration | Chronological validation |
| Final metrics | Test set (once) |

Random `early_stopping` holdouts inside the training matrix are **disabled**.
