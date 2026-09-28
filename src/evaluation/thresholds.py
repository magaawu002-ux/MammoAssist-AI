"""Validation-only operating-point selection with documented deterministic ties."""

import numpy as np

from src.evaluation.metrics import prediction_arrays


def select_threshold(labels: np.ndarray, probabilities: np.ndarray, *, partition: str,
                     method: str = "youden_j") -> dict:
    """Maximise sensitivity + specificity - 1; ties nearest 0.5, then lower value."""
    if partition != "validation":
        raise ValueError("Threshold selection accepts validation predictions only")
    y, p = prediction_arrays(labels, probabilities)
    if method not in ("youden_j", "fixed_0.5"):
        raise ValueError("Unsupported threshold criterion")
    if method == "fixed_0.5":
        return {"threshold": 0.5, "criterion": method, "partition": partition, "score": None}
    if len(np.unique(y)) != 2:
        raise ValueError("Youden J requires both classes in validation; no silent fallback")
    # Include all decision boundaries under >=, plus defaults/endpoints.
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], p, np.nextafter(p[p < 1], 1))))
    scores = []
    for threshold in candidates:
        predicted = p >= threshold
        scores.append(float(predicted[y == 1].mean() + (~predicted[y == 0]).mean() - 1))
    best = max(scores)
    tied = [float(t) for t, score in zip(candidates, scores) if np.isclose(score, best, rtol=0, atol=1e-12)]
    threshold = min(tied, key=lambda t: (abs(t - 0.5), t))
    return {"threshold": threshold, "criterion": method, "partition": partition, "score": best,
            "tie_break": "nearest_0.5_then_lower", "decision_rule": "probability >= threshold"}
