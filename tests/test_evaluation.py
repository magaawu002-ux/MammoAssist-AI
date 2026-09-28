"""Known numerical examples and deterministic validation-only decisions."""

import numpy as np
import pytest

from src.evaluation.bootstrap import bootstrap_intervals
from src.evaluation.calibration import calibration_curve
from src.evaluation.errors import compare_errors, error_summary, false_negatives, prediction_records
from src.evaluation.metrics import evaluate_metrics
from src.evaluation.thresholds import select_threshold


def test_known_confusion_and_metrics():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.2, 0.7, 0.8, 0.9])
    m = evaluate_metrics(y, p)
    assert [m[k] for k in ("tn", "fp", "fn", "tp")] == [3, 1, 1, 3]
    for key in ("accuracy", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy"):
        assert m[key] == 0.75
    # Pairwise AUC: 12 wins (ties count half) / 16 positive-negative pairs.
    assert m["roc_auc"] == 0.75
    # AP: mean precision at each recall increment, grouping tied scores.
    assert m["pr_auc"] == pytest.approx((1 + 2/3 + 3/4 + 4/7) / 4)
    assert m["brier_score"] == pytest.approx(np.mean((p-y)**2))


@pytest.mark.parametrize("label", [0, 1])
def test_one_class_metrics_explicit(label):
    m = evaluate_metrics(np.array([label, label]), np.array([0.1, 0.9]))
    assert m["roc_auc"] is None and m["pr_auc"] is None and m["balanced_accuracy"] is None
    assert m["edge_case_note"]
    assert m["sensitivity" if label == 0 else "specificity"] is None


def test_zero_positive_predictions():
    m = evaluate_metrics(np.array([0, 1]), np.array([0.1, 0.2]))
    assert m["precision"] is None and m["f1"] == 0 and m["sensitivity"] == 0


def test_threshold_validation_only_deterministic():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.8])
    selected = select_threshold(y, p, partition="validation")
    assert selected == select_threshold(y, p, partition="validation")
    assert selected["threshold"] == 0.5 and selected["score"] == 1
    for partition in ("test", "train"):
        with pytest.raises(ValueError, match="validation predictions only"):
            select_threshold(y, p, partition=partition)
    with pytest.raises(ValueError, match="both classes"):
        select_threshold(np.array([1, 1]), np.array([0.2, 0.9]), partition="validation")


def test_prediction_errors_and_fn_ranking():
    rows = [{"participant_id": f"P_{i:05}", "canonical_image_id": str(i), "binary_label": y}
            for i, y in enumerate([0, 0, 1, 1, 1])]
    records = prediction_records(rows, np.array([0.1, 0.8, 0.3, 0.9, 0.1]), 0.5)
    assert [r["error_type"] for r in records] == ["true_negative", "false_positive", "false_negative", "true_positive", "false_negative"]
    assert [r["canonical_image_id"] for r in false_negatives(records)] == ["4", "2"]
    assert error_summary(records)["false_negative_rate"] == 2/3
    assert compare_errors(records, records)["both_false_negative"] == ["2", "4"]


def test_calibration_bins_and_brier():
    curve = calibration_curve(np.array([0, 1]), np.array([0.25, 1]), bins=4)
    assert curve["brier_score"] == 0.03125
    assert sum(r["count"] for r in curve["bins"]) == 2
    assert curve["bins"][-1]["positive_fraction"] == 1
    assert curve["bins"][0]["mean_probability"] is None


def test_cluster_bootstrap_reproducible_with_invalid_replicates():
    args = (np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), ["a", "a", "b", "b"], 0.5)
    first = bootstrap_intervals(*args, seed=42, iterations=40)
    assert first == bootstrap_intervals(*args, seed=42, iterations=40)
    assert first["intervals"]["roc_auc"]["invalid_replicates"] > 0
    assert first["intervals"]["roc_auc"]["lower"] == 1


@pytest.mark.parametrize("y,p", [([0, 2], [0.1, 0.2]), ([0, 1], [np.nan, 0.1]), ([0, 1], [0.1]), ([], [])])
def test_invalid_metrics_rejected(y, p):
    with pytest.raises(ValueError):
        evaluate_metrics(np.array(y), np.array(p))
