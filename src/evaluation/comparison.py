"""Validation-only model comparison from completed, integrity-checked runs."""

import json
from pathlib import Path

from src.data.cbis_ddsm_metadata import write_csv
from src.evaluation.evaluate import completed_run


def compare_runs(root: Path, runs: list[str], destination: Path | None = None) -> list[dict]:
    """Reject mixed cohorts and test metrics; never fabricate missing runs."""
    if not runs or len(runs) != len(set(runs)):
        raise ValueError("Provide unique completed experiment runs")
    rows, cohort = [], None
    for run in runs:
        directory, metadata, threshold, _ = completed_run(root, run)
        current = (metadata["processed_manifest_sha256"], metadata["cache_fingerprint"])
        if cohort is not None and cohort != current:
            raise ValueError("Comparison requires identical train/validation cohorts and canonical caches")
        cohort = current
        metrics = json.loads((directory / "validation/metrics.json").read_text(encoding="utf-8"))
        if metrics.get("partition") != "validation" or metrics["threshold"] != threshold["threshold"]:
            raise ValueError("Model selection comparison accepts matching validation metrics only")
        architecture = metadata["architecture"]
        parameters = architecture.get("total_parameters", architecture.get("stage_a", {}).get("total_parameters"))
        row = {"model": metadata["model"], "run": run, "partition": "validation", "threshold": threshold["threshold"],
            **{k: metrics[k] for k in ("accuracy", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy", "roc_auc", "pr_auc")},
            "false_negatives": metrics["fn"], "parameters": parameters, "training_time": metadata["training_seconds"]}
        rows.append(row)
    write_csv(destination or root / "results/metrics/model_comparison.csv", rows)
    return rows
