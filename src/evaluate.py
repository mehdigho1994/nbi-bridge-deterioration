"""Metrics for imbalanced deterioration prediction.

Average Precision (AP) is sklearn's average_precision_score (area under the
precision-recall curve via the step-function / interpolation used by sklearn).
We label it ``average_precision`` in machine-readable outputs; prose may say
PR-AUC only when clearly equated to AP.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import LEGACY_DECISION_THRESHOLD, TOP_K_FRACTIONS


def class_balance(y_true: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    n = len(y)
    n_pos = int(y.sum())
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n - n_pos,
        "positive_rate": float(n_pos / n) if n else float("nan"),
    }


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    """ECE with equal-width probability bins on [0, 1]."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = []
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        w = float(mask.mean())
        ece += w * abs(acc - conf)
        bin_stats.append(
            {"bin": i, "lo": lo, "hi": hi, "n": int(mask.sum()), "acc": acc, "conf": conf}
        )
    return {"ece": float(ece), "n_bins": n_bins, "binning": "equal_width_[0,1]", "bins": bin_stats, "n": n}


def binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = LEGACY_DECISION_THRESHOLD,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    y_hat = (y_prob >= threshold).astype(int)
    out: dict[str, Any] = {
        "threshold": threshold,
        **class_balance(y_true),
        "mean_predicted_probability": float(y_prob.mean()),
    }

    if len(np.unique(y_true)) < 2:
        out.update(
            {
                "roc_auc": float("nan"),
                "average_precision": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "brier": float("nan"),
                "log_loss": float("nan"),
                "confusion_matrix": [[0, 0], [0, 0]],
                "note": "Only one class present in y_true.",
            }
        )
        return out

    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["average_precision"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        out["average_precision"] = float("nan")
    # Back-compat alias (AP, not a separate trapezoidal PR-AUC).
    out["pr_auc"] = out["average_precision"]

    out["precision"] = float(precision_score(y_true, y_hat, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_hat, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_hat, zero_division=0))
    out["brier"] = float(brier_score_loss(y_true, y_prob))
    # clip for log_loss numerical safety
    eps = 1e-15
    out["log_loss"] = float(
        log_loss(y_true, np.clip(y_prob, eps, 1 - eps), labels=[0, 1])
    )
    out["confusion_matrix"] = confusion_matrix(y_true, y_hat, labels=[0, 1]).tolist()
    out["accuracy_secondary"] = float((y_hat == y_true).mean())
    out["ece"] = expected_calibration_error(y_true, y_prob)["ece"]
    return out


def topk_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fractions: tuple[float, ...] = TOP_K_FRACTIONS,
    tie_policy: str = "min_rating_desc_then_bridge_id",
    aux: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Precision/recall/lift at top fractions with a deterministic tie policy.

    tie_policy ``min_rating_desc_then_bridge_id``:
      sort by (-score, -min_rating if available, bridge_id).
    This makes Markov ties (identical scores within a rating) reproducible.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    base = float(y_true.mean()) if n else float("nan")
    order_keys: dict[str, Any] = {"score": -y_prob}
    if aux is not None and "min_rating" in aux.columns:
        order_keys["min_rating"] = -pd.to_numeric(aux["min_rating"], errors="coerce").fillna(-1).to_numpy()
    if aux is not None and "bridge_id" in aux.columns:
        # lexsort last key is primary; build stable order
        bridge = aux["bridge_id"].astype(str).to_numpy()
    else:
        bridge = np.arange(n).astype(str)

    # numpy lexsort: last key is primary
    keys = [bridge]
    if "min_rating" in order_keys:
        keys.append(order_keys["min_rating"])
    keys.append(order_keys["score"])
    order = np.lexsort(tuple(keys))

    out: dict[str, Any] = {
        "base_rate": base,
        "tie_policy": tie_policy,
        "fractions": {},
    }
    for f in fractions:
        k = max(1, int(round(n * f)))
        sel = order[:k]
        prec = float(y_true[sel].mean())
        rec = float(y_true[sel].sum() / max(y_true.sum(), 1))
        lift = float(prec / base) if base > 0 else float("nan")
        out["fractions"][str(f)] = {
            "k": k,
            "precision": prec,
            "recall": rec,
            "lift": lift,
            "true_positives_in_budget": int(y_true[sel].sum()),
        }
    return out


def select_threshold_on_validation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    grid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Choose a decision threshold on validation only (never on test)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if grid is None:
        grid = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), y_prob]))
    best_t, best_v = 0.5, -1.0
    for t in grid:
        hat = (y_prob >= t).astype(int)
        if metric == "f1":
            v = f1_score(y_true, hat, zero_division=0)
        elif metric == "precision":
            v = precision_score(y_true, hat, zero_division=0)
        else:
            raise ValueError(metric)
        if v > best_v:
            best_v, best_t = float(v), float(t)
    return {"threshold": best_t, "validation_metric": metric, "validation_value": best_v}


def summarize_seeds(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "f1",
        "brier",
        "log_loss",
        "mean_predicted_probability",
    ]
    summary: dict[str, Any] = {"n_seeds": len(metric_rows)}
    for k in keys:
        vals = np.array([r.get(k, np.nan) for r in metric_rows], dtype=float)
        summary[f"{k}_mean"] = float(np.nanmean(vals))
        summary[f"{k}_std"] = float(np.nanstd(vals, ddof=0))
    return summary


def curves_for_model(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return (fpr, tpr), (rec, prec)


def plot_roc_pr(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    kind: str,
    out_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(6.2, 4.8))
    for name, (x, y) in curves.items():
        plt.plot(x, y, label=name, linewidth=1.6)
    if kind == "roc":
        plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
    else:
        plt.xlabel("Recall")
        plt.ylabel("Precision")
    plt.title(title)
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_calibration(
    y_true: np.ndarray,
    prob_by_model: dict[str, np.ndarray],
    out_path: Path,
    title: str,
    n_bins: int = 10,
) -> None:
    plt.figure(figsize=(6.2, 4.8))
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
    for name, prob in prob_by_model.items():
        if len(np.unique(y_true)) < 2:
            continue
        frac_pos, mean_pred = calibration_curve(
            y_true, np.clip(prob, 0, 1), n_bins=n_bins, strategy="quantile"
        )
        plt.plot(mean_pred, frac_pos, marker="o", markersize=4, label=name)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_topk_bars(
    rows: list[dict[str, Any]], out_path: Path, title: str
) -> None:
    """rows: list of {model, fraction, precision, lift}."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, metric, ylab in zip(
        axes, ("precision", "lift"), ("Precision@k", "Lift@k"), strict=True
    ):
        for model, g in df.groupby("model"):
            ax.plot(g["fraction"] * 100, g[metric], marker="o", label=model)
        ax.set_xlabel("Top fraction (%)")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
