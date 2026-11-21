"""Train baselines + HGB under time splits; write metrics, tables, figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.baselines import (
    MarkovDropModel,
    fit_logistic,
    persistence_drop_proba,
    predict_logistic,
)
from src.config import DECISION_THRESHOLD, EVAL_SEEDS, RESULTS
from src.evaluate import (
    binary_metrics,
    curves_for_model,
    plot_calibration,
    plot_roc_pr,
    summarize_seeds,
    write_json,
)
from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from src.models import fit_hgb, predict_hgb
from src.panel import build_panel, load_panel
from src.splits import assert_features_not_from_future, apply_time_split, default_splits


def _safe_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    return binary_metrics(y_true, y_prob, threshold=DECISION_THRESHOLD)


def evaluate_split(
    panel: pd.DataFrame,
    split_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    seeds: tuple[int, ...] = EVAL_SEEDS,
) -> dict[str, Any]:
    """Run all models on one time split; average stochastic models over seeds."""
    assert_features_not_from_future(
        train, list(FEATURE_COLUMNS) + list(CATEGORICAL_FEATURES)
    )
    y_test = test["y_drop"].astype(int).to_numpy()
    balance = {
        "train_positive_rate": float(train["y_drop"].mean()),
        "test_positive_rate": float(test["y_drop"].mean()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_train_pos": int(train["y_drop"].sum()),
        "n_test_pos": int(test["y_drop"].sum()),
    }
    print(
        f"[{split_name}] train={balance['n_train']:,} "
        f"(pos={balance['n_train_pos']:,}, rate={balance['train_positive_rate']:.3%}) | "
        f"test={balance['n_test']:,} "
        f"(pos={balance['n_test_pos']:,}, rate={balance['test_positive_rate']:.3%})"
    )

    results: dict[str, Any] = {
        "split": split_name,
        "threshold": DECISION_THRESHOLD,
        "class_balance": balance,
        "models": {},
    }

    # --- Deterministic baselines ---
    pers_prob = persistence_drop_proba(test)
    # Persistence always predicts 0 → ROC/PR undefined or degenerate; still report.
    results["models"]["persistence"] = {
        "deterministic": True,
        "metrics": _safe_metrics(y_test, pers_prob),
        "note": "P(drop)=0 for all rows (rating assumed unchanged).",
    }

    markov = MarkovDropModel().fit(train)
    markov_prob = markov.predict_proba(test)
    results["models"]["markov"] = {
        "deterministic": True,
        "metrics": _safe_metrics(y_test, markov_prob),
        "p_drop_given_min": {str(k): v for k, v in markov.p_drop_given_min.items()},
        "global_rate": markov.global_rate,
    }

    # --- Seeded models ---
    logit_rows: list[dict[str, Any]] = []
    hgb_rows: list[dict[str, Any]] = []
    logit_probs: list[np.ndarray] = []
    hgb_probs: list[np.ndarray] = []
    last_hgb = None

    for seed in seeds:
        logit_model, _ = fit_logistic(train, seed=seed)
        lp = predict_logistic(logit_model, test)
        logit_probs.append(lp)
        logit_rows.append(_safe_metrics(y_test, lp))

        hgb_model = fit_hgb(train, seed=seed)
        hp = predict_hgb(hgb_model, test)
        hgb_probs.append(hp)
        hgb_rows.append(_safe_metrics(y_test, hp))
        last_hgb = hgb_model
        print(
            f"  seed={seed}: logit ROC={logit_rows[-1]['roc_auc']:.4f} "
            f"PR={logit_rows[-1]['pr_auc']:.4f} | "
            f"hgb ROC={hgb_rows[-1]['roc_auc']:.4f} PR={hgb_rows[-1]['pr_auc']:.4f}"
        )

    logit_mean_prob = np.mean(np.vstack(logit_probs), axis=0)
    hgb_mean_prob = np.mean(np.vstack(hgb_probs), axis=0)

    results["models"]["logistic_regression"] = {
        "deterministic": False,
        "seeds": list(seeds),
        "per_seed": logit_rows,
        "summary": summarize_seeds(logit_rows),
        "metrics_mean_prob": _safe_metrics(y_test, logit_mean_prob),
    }
    results["models"]["hist_gradient_boosting"] = {
        "deterministic": False,
        "seeds": list(seeds),
        "per_seed": hgb_rows,
        "summary": summarize_seeds(hgb_rows),
        "metrics_mean_prob": _safe_metrics(y_test, hgb_mean_prob),
    }

    # Honest comparison vs strongest baseline (Markov; persistence is degenerate).
    hgb_pr = results["models"]["hist_gradient_boosting"]["summary"]["pr_auc_mean"]
    hgb_roc = results["models"]["hist_gradient_boosting"]["summary"]["roc_auc_mean"]
    mk_pr = results["models"]["markov"]["metrics"]["pr_auc"]
    mk_roc = results["models"]["markov"]["metrics"]["roc_auc"]
    results["comparison"] = {
        "hgb_beats_markov_pr_auc": bool(hgb_pr > mk_pr) if mk_pr == mk_pr else False,
        "hgb_beats_markov_roc_auc": bool(hgb_roc > mk_roc) if mk_roc == mk_roc else False,
        "hgb_pr_auc_mean": hgb_pr,
        "markov_pr_auc": mk_pr,
        "hgb_roc_auc_mean": hgb_roc,
        "markov_roc_auc": mk_roc,
    }

    # Figures (use mean-seed probabilities for stochastic models).
    roc_curves = {}
    pr_curves = {}
    calib = {}
    for name, prob in [
        ("markov", markov_prob),
        ("logistic", logit_mean_prob),
        ("hgb", hgb_mean_prob),
    ]:
        if len(np.unique(y_test)) >= 2:
            roc, pr = curves_for_model(y_test, prob)
            roc_curves[name] = roc
            pr_curves[name] = pr
            calib[name] = prob

    fig_dir = RESULTS / "figures"
    if roc_curves:
        plot_roc_pr(
            roc_curves,
            "roc",
            fig_dir / f"roc_{split_name}.png",
            title=f"ROC — {split_name}",
        )
        plot_roc_pr(
            pr_curves,
            "pr",
            fig_dir / f"pr_{split_name}.png",
            title=f"Precision–Recall — {split_name}",
        )
        plot_calibration(
            y_test,
            calib,
            fig_dir / f"calibration_{split_name}.png",
            title=f"Calibration — {split_name}",
        )

    # Persist last HGB for predict_from_record demos (second split overwrites).
    if last_hgb is not None:
        model_path = RESULTS / "hgb_model.joblib"
        joblib.dump(
            {
                "model": last_hgb,
                "feature_columns": list(FEATURE_COLUMNS),
                "categorical_features": list(CATEGORICAL_FEATURES),
                "split": split_name,
                "threshold": DECISION_THRESHOLD,
            },
            model_path,
        )

    # Save per-split predictions for inspection.
    pred_df = test[
        ["bridge_id", "inventory_year", "min_rating", "y_drop"]
    ].copy()
    pred_df["p_persistence"] = pers_prob
    pred_df["p_markov"] = markov_prob
    pred_df["p_logistic"] = logit_mean_prob
    pred_df["p_hgb"] = hgb_mean_prob
    pred_df.to_csv(RESULTS / f"predictions_{split_name}.csv", index=False)

    return results


def run_evaluation(rebuild_panel: bool = True) -> dict[str, Any]:
    """Build panel if needed and evaluate all configured time splits."""
    from src.config import DATA_PROCESSED

    RESULTS.mkdir(parents=True, exist_ok=True)
    panel_path = DATA_PROCESSED / "panel.parquet"
    meta_path = DATA_PROCESSED / "panel_meta.json"
    if rebuild_panel or not panel_path.exists():
        panel, meta = build_panel()
    else:
        panel = load_panel(panel_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    all_results: dict[str, Any] = {
        "panel_meta": meta,
        "decision_threshold": DECISION_THRESHOLD,
        "seeds": list(EVAL_SEEDS),
        "splits": {},
    }

    table_rows: list[dict[str, Any]] = []
    for split in default_splits():
        train, test = apply_time_split(panel, split)
        if train.empty or test.empty:
            print(f"[warn] empty split {split.name}; skipping")
            continue
        split_result = evaluate_split(panel, split.name, train, test)
        all_results["splits"][split.name] = split_result

        for model_name, payload in split_result["models"].items():
            if payload.get("deterministic"):
                m = payload["metrics"]
                table_rows.append(
                    {
                        "split": split.name,
                        "model": model_name,
                        "roc_auc": m.get("roc_auc"),
                        "pr_auc": m.get("pr_auc"),
                        "precision": m.get("precision"),
                        "recall": m.get("recall"),
                        "positive_rate_test": m.get("positive_rate"),
                    }
                )
            else:
                s = payload["summary"]
                table_rows.append(
                    {
                        "split": split.name,
                        "model": model_name,
                        "roc_auc": f"{s['roc_auc_mean']:.4f} ± {s['roc_auc_std']:.4f}",
                        "pr_auc": f"{s['pr_auc_mean']:.4f} ± {s['pr_auc_std']:.4f}",
                        "precision": f"{s['precision_mean']:.4f} ± {s['precision_std']:.4f}",
                        "recall": f"{s['recall_mean']:.4f} ± {s['recall_std']:.4f}",
                        "positive_rate_test": split_result["class_balance"][
                            "test_positive_rate"
                        ],
                    }
                )

    table = pd.DataFrame(table_rows)
    table_path = RESULTS / "metrics_summary.csv"
    table.to_csv(table_path, index=False)
    write_json(all_results, RESULTS / "metrics_full.json")
    print(f"\n=== Results summary ===\n{table.to_string(index=False)}")
    print(f"\nWrote {table_path}")
    print(f"Wrote {RESULTS / 'metrics_full.json'}")
    return all_results


def main() -> None:
    run_evaluation(rebuild_panel=True)


if __name__ == "__main__":
    main()
