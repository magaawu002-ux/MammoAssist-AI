"""Descriptive calibration assessment; no post-hoc calibration fitting."""

import numpy as np

from src.evaluation.metrics import prediction_arrays


def calibration_curve(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> dict:
    """Equal-width bins including p=1; empty bins have null means and zero count."""
    y, p = prediction_arrays(labels, probabilities)
    if type(bins) is not int or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    indices = np.minimum((p * bins).astype(int), bins - 1)
    records = []
    for i in range(bins):
        chosen = indices == i
        records.append({"lower": i / bins, "upper": (i + 1) / bins, "count": int(chosen.sum()),
                        "mean_probability": float(p[chosen].mean()) if chosen.any() else None,
                        "positive_fraction": float(y[chosen].mean()) if chosen.any() else None})
    return {"brier_score": float(np.mean((p - y) ** 2)), "bins": records}
