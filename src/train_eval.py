"""Core and supplementary experiment runner."""

from __future__ import annotations

import argparse
import json
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
from src.calibration import ProbabilityCalibrator
from src.config import (
    ALT_TARGETS,
    EVAL_SEEDS,
    LEGACY_DECISION_THRESHOLD,
    PRIMARY_TARGET,
    RESULTS,
    TemporalSplitSpec,
)
from src.evaluate import (
    binary_metrics,
    curves_for_model,
    plot_calibration,
    plot_roc_pr,
    plot_topk_bars,
    select_threshold_on_validation,
    summarize_seeds,
    topk_metrics,
    write_json,
)
from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from src.inspection_study import run_inspection_to_inspection_study
from src.interpretability import run_feature_ablation, run_permutation_importance
from src.models import fit_hgb, predict_hgb, select_hgb_max_iter
from src.panel import build_panel, load_panel
from src.splits import (
    assert_features_not_from_future,
    apply_temporal_split,
    default_standard_splits,
    default_strict_splits,
)


def _stack_mean(probs: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.vstack(probs), axis=0)


def _calibrate_block(
    y_val: np.ndarray,
    p_val: np.ndarray,
    p_test: np.ndarray,
    y_test: np.ndarray,
    test_aux: pd.DataFrame,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in ("sigmoid", "isotonic"):
        cal = ProbabilityCalibrator(method).fit(y_val, p_val)
        p_cal = cal.transform(p_test)
        out[method] = {
            "metrics_legacy_threshold": binary_metrics(
                y_test, p_cal, threshold=LEGACY_DECISION_THRESHOLD
            ),
            "topk": topk_metrics(y_test, p_cal, aux=test_aux),
            "mean_predicted_probability": float(p_cal.mean()),
            "note": "Calibrator fitted on chronological validation only.",
        }
    return out


def evaluate_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    split: TemporalSplitSpec,
    target_col: str = PRIMARY_TARGET,
    seeds: tuple[int, ...] = EVAL_SEEDS,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    assert_features_not_from_future(
        train, list(FEATURE_COLUMNS) + list(CATEGORICAL_FEATURES)
    )
    y_test = test[target_col].astype(int).to_numpy()
    has_val = len(val) > 0
    y_val = val[target_col].astype(int).to_numpy() if has_val else None

    result: dict[str, Any] = {
        "split": split.name,
        "protocol": split.protocol,
        "target": target_col,
        "temporal_semantics": {
            "feature_year": "inventory_year of the covariate row",
            "label_observation_year": "next inventory year (t+1) ratings",
            "train_max_feature_year": split.train_max_feature_year,
            "validation_feature_year": split.validation_feature_year,
            "test_feature_year": split.test_feature_year,
            "standard_note": (
                "Under the standard protocol, training labels may use ratings "
                "from the same calendar year that later appears as a test feature year."
            ),
            "strict_note": (
                "Under the strict protocol, every training (and validation) label "
                "observation year is strictly before the test feature year."
            ),
        },
        "class_balance": {
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "train_positive_rate": float(train[target_col].mean()),
            "val_positive_rate": float(val[target_col].mean()) if has_val else None,
            "test_positive_rate": float(test[target_col].mean()),
            "n_train_pos": int(train[target_col].sum()),
            "n_test_pos": int(test[target_col].sum()),
        },
        "models": {},
    }
    b = result["class_balance"]
    print(
        f"[{split.name}|{target_col}] train={b['n_train']:,} "
        f"(pos={b['n_train_pos']:,}) val={b['n_val']:,} "
        f"test={b['n_test']:,} (pos={b['n_test_pos']:,}, "
        f"rate={b['test_positive_rate']:.3%})"
    )

    probs_out: dict[str, np.ndarray] = {}

    # Persistence
    p = persistence_drop_proba(test)
    probs_out["persistence"] = p
    result["models"]["persistence"] = {
        "deterministic": True,
        "metrics_legacy_threshold": binary_metrics(
            y_test, p, threshold=LEGACY_DECISION_THRESHOLD
        ),
        "topk": topk_metrics(y_test, p, aux=test),
    }

    # Markov
    markov = MarkovDropModel().fit(train, target_col=target_col)
    p = markov.predict_proba(test)
    probs_out["markov"] = p
    entry: dict[str, Any] = {
        "deterministic": True,
        "p_drop_given_min": {str(k): v for k, v in markov.p_drop_given_min.items()},
        "global_rate": markov.global_rate,
        "metrics_legacy_threshold": binary_metrics(
            y_test, p, threshold=LEGACY_DECISION_THRESHOLD
        ),
        "topk": topk_metrics(y_test, p, aux=test),
    }
    if has_val:
        pv = markov.predict_proba(val)
        thr = select_threshold_on_validation(y_val, pv)["threshold"]
        entry["val_selected_threshold"] = thr
        entry["metrics_val_selected_threshold"] = binary_metrics(
            y_test, p, threshold=thr
        )
    result["models"]["markov"] = entry

    # Logistic variants
    for name, cw in (
        ("logistic_unweighted", None),
        ("logistic_balanced", "balanced"),
    ):
        seed_metrics = []
        te_list, va_list = [], []
        for seed in seeds:
            model = fit_logistic(
                train, seed, class_weight=cw, target_col=target_col
            )
            pt = predict_logistic(model, test)
            te_list.append(pt)
            if has_val:
                va_list.append(predict_logistic(model, val))
            m = binary_metrics(y_test, pt, threshold=LEGACY_DECISION_THRESHOLD)
            seed_metrics.append(m)
            print(
                f"  {name} seed={seed}: ROC={m['roc_auc']:.4f} "
                f"AP={m['average_precision']:.4f} "
                f"mean_p={m['mean_predicted_probability']:.4f}"
            )
        mean_te = _stack_mean(te_list)
        probs_out[name] = mean_te
        entry = {
            "class_weight": cw,
            "seeds": list(seeds),
            "per_seed": seed_metrics,
            "summary": summarize_seeds(seed_metrics),
            "metrics_mean_prob_legacy_threshold": binary_metrics(
                y_test, mean_te, threshold=LEGACY_DECISION_THRESHOLD
            ),
            "topk": topk_metrics(y_test, mean_te, aux=test),
        }
        if has_val and va_list:
            mean_va = _stack_mean(va_list)
            thr = select_threshold_on_validation(y_val, mean_va)["threshold"]
            entry["val_selected_threshold"] = thr
            entry["metrics_val_selected_threshold"] = binary_metrics(
                y_test, mean_te, threshold=thr
            )
            entry["calibration"] = _calibrate_block(
                y_val, mean_va, mean_te, y_test, test
            )
        result["models"][name] = entry

    # HGB variants
    for name, cw in (("hgb_unweighted", None), ("hgb_balanced", "balanced")):
        seed_metrics = []
        te_list, va_list = [], []
        iters = []
        last_model = None
        for seed in seeds:
            best_iter, grid_scores = select_hgb_max_iter(
                train, val, seed, class_weight=cw, target_col=target_col
            )
            iters.append(best_iter)
            model = fit_hgb(
                train,
                seed,
                max_iter=best_iter,
                class_weight=cw,
                target_col=target_col,
            )
            last_model = model
            pt = predict_hgb(model, test)
            te_list.append(pt)
            if has_val:
                va_list.append(predict_hgb(model, val))
            m = binary_metrics(y_test, pt, threshold=LEGACY_DECISION_THRESHOLD)
            m["selected_max_iter"] = best_iter
            m["val_ap_by_iter"] = grid_scores
            seed_metrics.append(m)
            print(
                f"  {name} seed={seed}: iter={best_iter} "
                f"ROC={m['roc_auc']:.4f} AP={m['average_precision']:.4f} "
                f"mean_p={m['mean_predicted_probability']:.4f}"
            )
        mean_te = _stack_mean(te_list)
        probs_out[name] = mean_te
        entry = {
            "class_weight": cw,
            "seeds": list(seeds),
            "selected_max_iter_per_seed": iters,
            "hyperparameter_selection": (
                "max_iter chosen by Average Precision on chronological "
                "validation; early_stopping disabled (no random holdout)."
            ),
            "per_seed": seed_metrics,
            "summary": summarize_seeds(seed_metrics),
            "metrics_mean_prob_legacy_threshold": binary_metrics(
                y_test, mean_te, threshold=LEGACY_DECISION_THRESHOLD
            ),
            "topk": topk_metrics(y_test, mean_te, aux=test),
        }
        if has_val and va_list:
            mean_va = _stack_mean(va_list)
            thr = select_threshold_on_validation(y_val, mean_va)["threshold"]
            entry["val_selected_threshold"] = thr
            entry["metrics_val_selected_threshold"] = binary_metrics(
                y_test, mean_te, threshold=thr
            )
            entry["calibration"] = _calibrate_block(
                y_val, mean_va, mean_te, y_test, test
            )
            # Permutation importance on validation for primary unweighted HGB only
            if (
                name == "hgb_unweighted"
                and last_model is not None
                and target_col == PRIMARY_TARGET
            ):
                entry["permutation_importance_validation"] = run_permutation_importance(
                    train,
                    val,
                    test,
                    seed=seeds[-1],
                    max_iter=iters[-1],
                    target_col=target_col,
                )
        result["models"][name] = entry

        if (
            name == "hgb_unweighted"
            and last_model is not None
            and target_col == PRIMARY_TARGET
        ):
            joblib.dump(
                {
                    "model": last_model,
                    "feature_columns": list(FEATURE_COLUMNS),
                    "categorical_features": list(CATEGORICAL_FEATURES),
                    "split": split.name,
                    "target": target_col,
                    "score_type": "uncalibrated_ranking_score",
                    "disclaimer": (
                        "Scores are for ranking/prioritisation. With class weighting "
                        "or without calibration they are not reliable event probabilities."
                    ),
                },
                RESULTS / "hgb_model.joblib",
            )

    # Plots
    fig_dir = RESULTS / "figures"
    roc_curves, pr_curves, calib_map = {}, {}, {}
    for label in (
        "markov",
        "logistic_unweighted",
        "logistic_balanced",
        "hgb_unweighted",
        "hgb_balanced",
    ):
        if label in probs_out and len(np.unique(y_test)) >= 2:
            roc, pr = curves_for_model(y_test, probs_out[label])
            roc_curves[label] = roc
            pr_curves[label] = pr
            calib_map[label] = probs_out[label]
    if roc_curves:
        plot_roc_pr(
            roc_curves, "roc", fig_dir / f"roc_{split.name}.png", f"ROC — {split.name}"
        )
        plot_roc_pr(
            pr_curves, "pr", fig_dir / f"pr_{split.name}.png", f"Precision–Recall — {split.name}"
        )
        plot_calibration(
            y_test,
            calib_map,
            fig_dir / f"calibration_{split.name}.png",
            f"Calibration — {split.name}",
        )
        # calibrated HGB if present
        cal = result["models"]["hgb_unweighted"].get("calibration", {})
        if "sigmoid" in cal:
            # rebuild from stored mean_predicted only not enough; skip overlay
            pass
        topk_rows = []
        for mname, parr in probs_out.items():
            tk = topk_metrics(y_test, parr, aux=test)
            for frac, vals in tk["fractions"].items():
                topk_rows.append(
                    {
                        "model": mname,
                        "fraction": float(frac),
                        "precision": vals["precision"],
                        "lift": vals["lift"],
                    }
                )
        plot_topk_bars(
            topk_rows,
            fig_dir / f"topk_{split.name}.png",
            f"Top-k prioritisation — {split.name}",
        )

    # Predictions CSV
    pred = test[["bridge_id", "inventory_year", "min_rating"]].copy()
    pred["y_true"] = y_test
    for k, v in probs_out.items():
        pred[f"p_{k}"] = v
    pred.to_csv(RESULTS / f"predictions_{split.name}.csv", index=False)

    return result, probs_out


def _summary_rows(all_results: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for split_name, payload in all_results.get("splits", {}).items():
        for model_name, model in payload.get("models", {}).items():
            if "summary" in model:
                s = model["summary"]
                rows.append(
                    {
                        "split": split_name,
                        "protocol": payload.get("protocol"),
                        "target": payload.get("target"),
                        "model": model_name,
                        "roc_auc": f"{s['roc_auc_mean']:.4f} ± {s['roc_auc_std']:.4f}",
                        "average_precision": (
                            f"{s['average_precision_mean']:.4f} ± "
                            f"{s['average_precision_std']:.4f}"
                        ),
                        "mean_pred": (
                            f"{s['mean_predicted_probability_mean']:.4f} ± "
                            f"{s['mean_predicted_probability_std']:.4f}"
                        ),
                        "brier": f"{s['brier_mean']:.4f} ± {s['brier_std']:.4f}",
                    }
                )
            elif "metrics_legacy_threshold" in model:
                m = model["metrics_legacy_threshold"]
                rows.append(
                    {
                        "split": split_name,
                        "protocol": payload.get("protocol"),
                        "target": payload.get("target"),
                        "model": model_name,
                        "roc_auc": m.get("roc_auc"),
                        "average_precision": m.get("average_precision"),
                        "mean_pred": m.get("mean_predicted_probability"),
                        "brier": m.get("brier"),
                    }
                )
    return pd.DataFrame(rows)


def run_core(rebuild_panel: bool = True) -> dict[str, Any]:
    """Primary target under standard + strict temporal protocols."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    from src.config import DATA_PROCESSED

    if rebuild_panel or not (DATA_PROCESSED / "panel.parquet").exists():
        panel, meta = build_panel()
    else:
        panel = load_panel()
        meta = json.loads((DATA_PROCESSED / "panel_meta.json").read_text(encoding="utf-8"))

    all_results: dict[str, Any] = {
        "panel_meta": meta,
        "legacy_decision_threshold": LEGACY_DECISION_THRESHOLD,
        "seeds": list(EVAL_SEEDS),
        "splits": {},
        "result_labels": {
            "original_study_note": (
                "Earlier repository versions reported a simpler train<=T / test T+1 "
                "design without a chronological validation year and with random "
                "early_stopping. This run supersedes those metrics."
            )
        },
    }

    for split in list(default_standard_splits()) + list(default_strict_splits()):
        train, val, test = apply_temporal_split(panel, split)
        if train.empty or test.empty:
            print(f"[warn] skipping empty split {split.name}")
            continue
        split_result, _ = evaluate_split(train, val, test, split)
        all_results["splits"][split.name] = split_result

    table = _summary_rows(all_results)
    table.to_csv(RESULTS / "metrics_summary.csv", index=False)
    write_json(all_results, RESULTS / "metrics_full.json")
    print("\n=== Core summary ===")
    print(table.to_string(index=False))
    return all_results


def run_supplementary(rebuild_panel: bool = False) -> dict[str, Any]:
    """Alternative targets, inspection study, ablation."""
    from src.config import DATA_PROCESSED

    if rebuild_panel or not (DATA_PROCESSED / "panel.parquet").exists():
        panel, _ = build_panel()
    else:
        panel = load_panel()

    out: dict[str, Any] = {"alternative_targets": {}, "ablation": {}, "inspection": {}}

    # One standard + one strict split for each alternative target (faster)
    focus_splits = [default_standard_splits()[-1], default_strict_splits()[-1]]
    for target in ALT_TARGETS:
        out["alternative_targets"][target] = {}
        for split in focus_splits:
            spec = TemporalSplitSpec(
                name=f"{split.name}__{target}",
                protocol=split.protocol,
                train_max_feature_year=split.train_max_feature_year,
                validation_feature_year=split.validation_feature_year,
                test_feature_year=split.test_feature_year,
                target_col=target,
            )
            train, val, test = apply_temporal_split(panel, spec, target_col=target)
            if train.empty or test.empty:
                continue
            # Lightweight: markov + unweighted logit/HGB only (1 seed)
            res, _ = evaluate_split(
                train, val, test, spec, target_col=target, seeds=(42,)
            )
            out["alternative_targets"][target][spec.name] = {
                "class_balance": res["class_balance"],
                "markov_ap": res["models"]["markov"]["metrics_legacy_threshold"].get(
                    "average_precision"
                ),
                "hgb_unweighted_ap": res["models"]["hgb_unweighted"]["summary"].get(
                    "average_precision_mean"
                ),
                "hgb_unweighted_mean_pred": res["models"]["hgb_unweighted"]["summary"].get(
                    "mean_predicted_probability_mean"
                ),
            }

    out["inspection"] = run_inspection_to_inspection_study(panel)
    out["ablation"] = run_feature_ablation(panel, default_standard_splits()[-1])
    write_json(out, RESULTS / "supplementary_results.json")
    return out


def run_evaluation(rebuild_panel: bool = True, supplementary: bool = True) -> dict[str, Any]:
    core = run_core(rebuild_panel=rebuild_panel)
    supp = run_supplementary(rebuild_panel=False) if supplementary else {}
    return {"core": core, "supplementary": supp}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run NBI deterioration experiments.")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Skip supplementary experiments (faster).",
    )
    parser.add_argument(
        "--supplementary-only",
        action="store_true",
        help="Run only supplementary experiments (requires existing panel).",
    )
    parser.add_argument(
        "--reuse-panel",
        action="store_true",
        help="Reuse existing processed panel if present.",
    )
    args = parser.parse_args(argv)
    if args.supplementary_only:
        run_supplementary(rebuild_panel=False)
        return
    run_evaluation(
        rebuild_panel=not args.reuse_panel, supplementary=not args.core_only
    )


if __name__ == "__main__":
    main()
