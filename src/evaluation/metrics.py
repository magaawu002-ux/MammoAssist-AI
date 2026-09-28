"""Binary metrics with MALIGNANT=1 and explicit undefined quantities."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.majority import binary_labels


def prediction_arrays(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = binary_labels(labels)
    p = np.asarray(probabilities, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must match labels and be finite in [0,1]")
    return y, p


def evaluate_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    """PR-AUC denotes average precision (step-weighted PR area), not trapezoidal area."""
    y, p = prediction_arrays(labels, probabilities)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite in [0,1]")
    predicted = p >= threshold
    tn = int(np.sum((y == 0) & ~predicted))
    fp = int(np.sum((y == 0) & predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    tp = int(np.sum((y == 1) & predicted))
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    sensitivity, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    both = len(np.unique(y)) == 2
    result = {"threshold": float(threshold), "n": len(y), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tn + tp) / len(y), "sensitivity": sensitivity, "specificity": specificity,
        "precision": ratio(tp, tp + fp), "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (sensitivity + specificity) / 2 if both else None,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "pr_auc_definition": "average_precision", "brier_score": float(np.mean((p - y) ** 2))}
    result["undefined_metrics"] = [key for key, value in result.items() if value is None]
    result["edge_case_note"] = "One-class labels: ROC/PR AUC and balanced accuracy undefined" if not both else ""
    return result
