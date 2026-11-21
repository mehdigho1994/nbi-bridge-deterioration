"""Permutation importance and feature-group ablation (supplementary)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from src.config import EVAL_SEEDS, RESULTS
from src.evaluate import write_json
from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, FEATURE_GROUPS
from src.models import fit_hgb, predict_hgb, select_hgb_max_iter
from src.splits import TemporalSplitSpec, apply_temporal_split


def run_permutation_importance(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    seed: int = 42,
    max_iter: int = 150,
    target_col: str = "y_drop",
) -> dict[str, Any]:
    """Permutation importance on chronological validation (not final test)."""
    if val.empty:
        return {"skipped": "empty validation"}
    model = fit_hgb(
        train, seed, max_iter=max_iter, class_weight=None, target_col=target_col
    )
    x_cols: list[str] = getattr(model, "_x_cols_")
    # Evaluate on validation
    from src.models import _prep
    from src.features import CATEGORICAL_FEATURES as CATS

    cat = [c for c in CATS if c in x_cols]
    val_p = _prep(val, cat)
    y = val[target_col].astype(int).to_numpy()
    r = permutation_importance(
        model,
        val_p[x_cols],
        y,
        n_repeats=10,
        random_state=seed,
        scoring="average_precision",
    )
    rows = []
    for name, mean, std in zip(x_cols, r.importances_mean, r.importances_std, strict=True):
        rows.append({"feature": name, "importance_mean": float(mean), "importance_std": float(std)})
    rows.sort(key=lambda d: d["importance_mean"], reverse=True)
    return {
        "scoring": "average_precision",
        "evaluated_on": "chronological_validation",
        "features": rows,
        "caveat": "Importance reflects association under the fitted model, not causation.",
    }


def run_feature_ablation(
    panel: pd.DataFrame,
    spec: TemporalSplitSpec,
) -> dict[str, Any]:
    """Compare feature groups on a fixed temporal split (unweighted HGB)."""
    train, val, test = apply_temporal_split(panel, spec)
    if train.empty or test.empty:
        return {"skipped": "empty split"}

    groups = {
        "current_ratings": list(FEATURE_GROUPS["current_ratings"]),
        "current_plus_history": list(FEATURE_GROUPS["current_ratings"])
        + list(FEATURE_GROUPS["history"]),
        "structure_ops": [
            c for c in FEATURE_GROUPS["structure_ops"] if c in FEATURE_COLUMNS
        ],
        "structure_ops_with_cats": list(FEATURE_GROUPS["structure_ops"]),
        "all_features": list(FEATURE_COLUMNS) + list(CATEGORICAL_FEATURES),
    }

    results: dict[str, Any] = {"split": spec.name, "groups": {}}
    y_test = test["y_drop"].astype(int).to_numpy()
    for name, cols in groups.items():
        nums = [c for c in cols if c in FEATURE_COLUMNS]
        cats = [c for c in cols if c in CATEGORICAL_FEATURES]
        if not nums and not cats:
            continue
        best_iter, _ = select_hgb_max_iter(
            train,
            val,
            seed=42,
            class_weight=None,
            feature_columns=nums,
            categorical_features=cats,
        )
        model = fit_hgb(
            train,
            42,
            max_iter=best_iter,
            class_weight=None,
            feature_columns=nums,
            categorical_features=cats,
        )
        prob = predict_hgb(model, test)
        try:
            ap = float(average_precision_score(y_test, prob))
        except ValueError:
            ap = float("nan")
        from sklearn.metrics import roc_auc_score

        try:
            roc = float(roc_auc_score(y_test, prob))
        except ValueError:
            roc = float("nan")
        results["groups"][name] = {
            "n_features_numeric": len(nums),
            "n_features_categorical": len(cats),
            "selected_max_iter": best_iter,
            "test_roc_auc": roc,
            "test_average_precision": ap,
            "mean_predicted_probability": float(prob.mean()),
        }
    write_json(results, RESULTS / f"ablation_{spec.name}.json")
    return results
