"""Metrics and plotting for imbalanced drop prediction.

Accuracy is computed but never headlined — drops are rare.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive; required for headless / Windows runs
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import DECISION_THRESHOLD, RESULTS


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


def binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, Any]:
    """ROC-AUC, PR-AUC, precision/recall at threshold, confusion matrix."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    # Clip numerical noise.
    y_prob = np.clip(y_prob, 0.0, 1.0)
    y_hat = (y_prob >= threshold).astype(int)

    out: dict[str, Any] = {"threshold": threshold, **class_balance(y_true)}

    if len(np.unique(y_true)) < 2:
        out.update(
            {
                "roc_auc": float("nan"),
                "pr_auc": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "confusion_matrix": [[0, 0], [0, 0]],
                "note": "Only one class present in y_true.",
            }
        )
        return out

    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = float("nan")
        out["roc_auc_note"] = "undefined (constant scores or degenerate)"
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        out["pr_auc"] = float("nan")
    out["precision"] = float(precision_score(y_true, y_hat, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_hat, zero_division=0))
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    out["confusion_matrix"] = cm.tolist()
    # Accuracy reported only as a secondary diagnostic (not headlined).
    out["accuracy_secondary"] = float((y_hat == y_true).mean())
    return out


def summarize_seeds(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and std across seeds for key metrics."""
    keys = ["roc_auc", "pr_auc", "precision", "recall"]
    summary: dict[str, Any] = {"n_seeds": len(metric_rows)}
    for k in keys:
        vals = np.array([r[k] for r in metric_rows], dtype=float)
        summary[f"{k}_mean"] = float(np.nanmean(vals))
        summary[f"{k}_std"] = float(np.nanstd(vals, ddof=0))
    return summary


def plot_roc_pr(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    kind: str,
    out_path: Path,
    title: str,
) -> None:
    """Plot ROC or PR curves for multiple models."""
    plt.figure(figsize=(6.5, 5))
    for name, (x, y) in curves.items():
        plt.plot(x, y, label=name)
    if kind == "roc":
        plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
    else:
        plt.xlabel("Recall")
        plt.ylabel("Precision")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_calibration(
    y_true: np.ndarray,
    prob_by_model: dict[str, np.ndarray],
    out_path: Path,
    title: str,
    n_bins: int = 10,
) -> None:
    plt.figure(figsize=(6.5, 5))
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
    for name, prob in prob_by_model.items():
        if len(np.unique(y_true)) < 2:
            continue
        frac_pos, mean_pred = calibration_curve(
            y_true, np.clip(prob, 0, 1), n_bins=n_bins, strategy="quantile"
        )
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def curves_for_model(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return (fpr, tpr), (rec, prec)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def metrics_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
