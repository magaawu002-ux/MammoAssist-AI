"""Descriptive tables from saved results; no fitting, resampling or inference."""

import math

from src.reporting.artifacts import METRICS, close


def metric_row(model: str, run: dict) -> dict:
    metrics = run["metrics"]
    malignant = metrics["tp"] + metrics["fn"]
    return {"model": model, **{k: metrics[k] for k in METRICS}, "false_negatives": metrics["fn"],
            "false_negative_rate": metrics["fn"]/malignant if malignant else None,
            "threshold": metrics["threshold"], "run": run["run"]}


def false_negative_tables(predictions: list[dict], threshold: float) -> tuple[dict, list[dict], list[dict]]:
    """FNR denominators are malignant images; proximity bands are cumulative."""
    malignant = [r for r in predictions if int(r["true_label"]) == 1]
    false_negatives = [r for r in malignant if r["error_type"] == "false_negative"]
    summary = {"malignant_images": len(malignant), "false_negative_images": len(false_negatives),
               "false_negative_participants": len({r["participant_id"] for r in false_negatives}),
               "false_negative_rate": len(false_negatives)/len(malignant) if malignant else None}
    groups = []
    for field, categories in (("view", ("CC", "MLO")), ("laterality", ("LEFT", "RIGHT"))):
        if any(r[field] not in categories for r in predictions):
            raise ValueError(f"Unexpected subgroup: {field}")
        for category in categories:
            subset = [r for r in malignant if r[field] == category]
            n = sum(r["error_type"] == "false_negative" for r in subset)
            groups.append(dict(group=field, category=category, malignant_images=len(subset),
                               false_negative_images=n, false_negative_rate=n/len(subset) if subset else None))
    proximity = []
    for band in (.01, .025, .05, .10):
        for scope, rows in (("all_images", predictions), ("malignant_images", malignant), ("false_negative_images", false_negatives)):
            n = sum(abs(float(r["predicted_probability"])-threshold) <= band for r in rows)
            proximity.append(dict(scope=scope, absolute_distance_at_most=band, count=n, denominator=len(rows),
                                  fraction=n/len(rows) if rows else None))
    return summary, groups, proximity


def confidence_rows(metrics: dict, bootstrap: dict) -> list[dict]:
    rows = []
    for key in ("sensitivity", "specificity", "balanced_accuracy", "roc_auc"):
        interval = bootstrap["intervals"][key]
        low, high = interval["lower"], interval["upper"]
        if (not math.isfinite(low) or not math.isfinite(high) or not 0 <= low <= high <= 1
                or interval["valid_replicates"] <= 0 or interval["invalid_replicates"] < 0
                or interval["valid_replicates"] + interval["invalid_replicates"] != bootstrap["iterations"]):
            raise ValueError("Invalid saved bootstrap interval")
        rows.append(dict(metric=key, point_estimate=metrics[key], lower_95=low, upper_95=high,
                         valid_replicates=interval["valid_replicates"], invalid_replicates=interval["invalid_replicates"]))
    return rows


def calibration_row(partition: str, calibration: dict, metrics: dict) -> dict:
    bins = calibration["bins"]
    if sum(b["count"] for b in bins) != metrics["n"] or any(b["count"] < 0 for b in bins):
        raise ValueError("Calibration bin counts disagree with metrics")
    occupied = [b for b in bins if b["count"]]
    for b in occupied:
        if any(not math.isfinite(b[k]) or not 0 <= b[k] <= 1 for k in ("mean_probability", "positive_fraction")):
            raise ValueError("Invalid saved calibration bin")
    mean_probability = math.fsum(b["count"]*b["mean_probability"] for b in occupied)/metrics["n"]
    prevalence = math.fsum(b["count"]*b["positive_fraction"] for b in occupied)/metrics["n"]
    close(prevalence, (metrics["tp"]+metrics["fn"])/metrics["n"], "calibration prevalence")
    return dict(partition=partition, brier_score=calibration["brier_score"],
                mean_probability=mean_probability, observed_malignant_fraction=prevalence,
                mean_probability_minus_observed=mean_probability-prevalence)


def history_values(rows: list[dict]) -> tuple[list[int], list[float], list[float], int | None]:
    epochs = [int(r["epoch"]) for r in rows]
    losses = [float(r["loss"]) for r in rows]
    validation = [float(r["val_loss"]) for r in rows]
    if epochs != list(range(1, len(rows)+1)) or any(not math.isfinite(v) or v < 0 for v in losses+validation):
        raise ValueError("Invalid training history")
    stages = [r.get("stage", "") for r in rows]
    boundary = None
    if all(stages):
        if set(stages) - {"head", "fine_tuning"}:
            raise ValueError("Unknown transfer-learning history phase")
        if "fine_tuning" in stages:
            boundary = stages.index("fine_tuning")
            if boundary == 0 or stages != ["head"]*boundary + ["fine_tuning"]*(len(rows)-boundary):
                raise ValueError("Invalid transfer-learning stage order")
    return epochs, losses, validation, boundary
