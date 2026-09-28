"""Optional participant-cluster percentile bootstrap, keeping a fixed threshold."""

import numpy as np

from src.evaluation.metrics import evaluate_metrics, prediction_arrays


def bootstrap_intervals(labels: np.ndarray, probabilities: np.ndarray, participants: list[str],
                        threshold: float, *, seed: int, iterations: int = 2000, confidence: float = 0.95) -> dict:
    """Resample participants, including all their images; report invalid replicate counts."""
    y, p = prediction_arrays(labels, probabilities)
    ids = np.asarray(participants)
    if ids.shape != y.shape or any(not str(v) for v in ids) or iterations < 2 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap inputs")
    evaluate_metrics(y, p, threshold)
    unique = np.unique(ids)
    if len(unique) < 2:
        raise ValueError("Cluster bootstrap requires at least two participants")
    groups = {v: np.flatnonzero(ids == v) for v in unique}
    rng = np.random.default_rng(seed)
    values = {key: [] for key in ("sensitivity", "specificity", "balanced_accuracy", "roc_auc")}
    for _ in range(iterations):
        indices = np.concatenate([groups[v] for v in rng.choice(unique, len(unique), replace=True)])
        metrics = evaluate_metrics(y[indices], p[indices], threshold)
        for key in values:
            if metrics[key] is not None:
                values[key].append(metrics[key])
    alpha = (1 - confidence) / 2
    return {"method": "participant_cluster_percentile", "seed": seed, "iterations": iterations,
        "confidence": confidence, "threshold_fixed": threshold,
        "intervals": {key: {"lower": float(np.quantile(v, alpha)) if len(v) >= 2 else None,
                            "upper": float(np.quantile(v, 1 - alpha)) if len(v) >= 2 else None,
                            "valid_replicates": len(v), "invalid_replicates": iterations - len(v)}
                      for key, v in values.items()}}
