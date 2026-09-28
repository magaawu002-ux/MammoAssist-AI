"""Traceable predictions and malignant false-negative review infrastructure."""

import numpy as np

from src.evaluation.metrics import evaluate_metrics, prediction_arrays


def prediction_records(rows: list[dict], probabilities: np.ndarray, threshold: float) -> list[dict]:
    y, p = prediction_arrays(np.array([int(r["binary_label"]) for r in rows]), probabilities)
    evaluate_metrics(y, p, threshold)  # Validate threshold even when no positive predictions.
    result = []
    names = {(0, 0): "true_negative", (0, 1): "false_positive", (1, 0): "false_negative", (1, 1): "true_positive"}
    for row, label, probability in zip(rows, y, p):
        predicted = int(probability >= threshold)
        result.append(dict(row, true_label=int(label), predicted_probability=float(probability), threshold=threshold,
                           predicted_class=predicted, correct=bool(predicted == label), error_type=names[int(label), predicted]))
    return result


def false_negatives(records: list[dict]) -> list[dict]:
    """Rank most confidently missed malignant images first; identifiers stay intact."""
    return sorted((dict(r) for r in records if r["error_type"] == "false_negative"),
                  key=lambda r: (float(r["predicted_probability"]), r["canonical_image_id"]))


def error_summary(records: list[dict]) -> dict:
    malignant = sum(int(r["true_label"]) == 1 for r in records)
    count = len(false_negatives(records))
    return {"false_negatives": count, "malignant_images": malignant,
            "false_negative_rate": count / malignant if malignant else None}


def compare_errors(first: list[dict], second: list[dict]) -> dict:
    """Compare only identical labelled cohorts, without discarding unmatched images."""
    a = {r["canonical_image_id"]: int(r["true_label"]) for r in first}
    b = {r["canonical_image_id"]: int(r["true_label"]) for r in second}
    if a != b or len(a) != len(first) or len(b) != len(second):
        raise ValueError("Error comparison requires identical unique labelled image cohorts")
    fa = {r["canonical_image_id"] for r in false_negatives(first)}
    fb = {r["canonical_image_id"] for r in false_negatives(second)}
    return {"both_false_negative": sorted(fa & fb), "first_only": sorted(fa - fb), "second_only": sorted(fb - fa)}
