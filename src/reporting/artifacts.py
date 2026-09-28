"""Validate saved artifacts and their lock/hash chain without loading models."""

import csv
import json
import math
from pathlib import Path

from src.data.provenance import sha256_file
from src.preprocessing.manifests import local_path

MODELS = ("majority_reference", "mammocnn_baseline", "mammocnn_regularised",
          "vgg16", "resnet50", "efficientnetb0")
METRICS = ("accuracy", "sensitivity", "specificity", "precision", "f1",
           "balanced_accuracy", "roc_auc", "pr_auc", "brier_score")


class Sources:
    """Record exact source bytes; detect changes during report generation."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.hashes: dict[str, str] = {}

    def path(self, relative: str) -> Path:
        path = local_path(self.root, relative, ())
        if not path.is_file():
            raise ValueError(f"Required reporting artifact missing: {relative}")
        value = sha256_file(path)
        if relative in self.hashes and self.hashes[relative] != value:
            raise ValueError(f"Source changed during reporting: {relative}")
        self.hashes[relative] = value
        return path

    def json(self, relative: str) -> dict:
        return json.loads(self.path(relative).read_text(encoding="utf-8"))

    def csv(self, relative: str) -> list[dict]:
        with self.path(relative).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            if (not rows or not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames)
                    or any(None in r or None in r.values() for r in rows)):
                raise ValueError(f"Empty/malformed artifact: {relative}")
            return rows

    def check(self, relative: str, expected: str) -> None:
        self.path(relative)
        if self.hashes[relative] != expected:
            raise ValueError(f"Artifact hash mismatch: {relative}")

    def unchanged(self) -> None:
        for relative in list(self.hashes):
            self.path(relative)


def close(actual: float | None, expected: float | None, description: str) -> None:
    if actual is None or expected is None:
        valid = actual is expected
    else:
        valid = math.isfinite(actual) and math.isfinite(expected) and math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-10)
    if not valid:
        raise ValueError(f"Inconsistent {description}: {actual} vs {expected}")


def verify_predictions(rows: list[dict], metrics: dict, split: list[dict], partition: str) -> None:
    """Check saved decisions/denominators; never infer or select a new threshold."""
    source = {r["canonical_image_id"]: r for r in split}
    ids = [r["canonical_image_id"] for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(source) or len(rows) != metrics["n"]:
        raise ValueError("Prediction image cohort mismatch")
    if metrics["partition"] != partition or metrics["pr_auc_definition"] != "average_precision":
        raise ValueError("Unexpected metric partition or PR-AUC definition")
    counts = dict(tn=0, fp=0, fn=0, tp=0)
    errors = {(0, 0): ("tn", "true_negative"), (0, 1): ("fp", "false_positive"),
              (1, 0): ("fn", "false_negative"), (1, 1): ("tp", "true_positive")}
    squared_errors = []
    for row in rows:
        original = source[row["canonical_image_id"]]
        for key in ("participant_id", "binary_label", "laterality", "view", "final_partition", "official_partition"):
            if row[key] != original[key]:
                raise ValueError(f"Prediction traceability mismatch: {key}")
        y, predicted, p = int(row["true_label"]), int(row["predicted_class"]), float(row["predicted_probability"])
        if (y not in (0, 1) or str(y) != row["binary_label"] or not math.isfinite(p) or not 0 <= p <= 1
                or float(row["threshold"]) != metrics["threshold"] or predicted != int(p >= metrics["threshold"])):
            raise ValueError("Invalid saved prediction/threshold")
        key, error = errors[y, predicted]
        if row["error_type"] != error or row["correct"] != str(y == predicted):
            raise ValueError("Saved prediction error label mismatch")
        counts[key] += 1
        squared_errors.append((p-y)**2)
    if any(metrics[k] != v for k, v in counts.items()):
        raise ValueError("Saved confusion matrix disagrees with predictions")
    tn, fp, fn, tp = (counts[k] for k in ("tn", "fp", "fn", "tp"))
    ratio = lambda n, d: n/d if d else None
    sensitivity, specificity = ratio(tp, tp+fn), ratio(tn, tn+fp)
    expected = {"accuracy": (tn+tp)/len(rows), "sensitivity": sensitivity, "specificity": specificity,
                "precision": ratio(tp, tp+fp), "f1": ratio(2*tp, 2*tp+fp+fn),
                "balanced_accuracy": (sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
                "brier_score": math.fsum(squared_errors)/len(rows)}
    for key in METRICS:
        value = metrics[key]
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"Invalid saved metric: {key}")
        if key in expected:
            close(value, expected[key], key)


def load_evidence(root: Path) -> dict:
    """Require exactly six unambiguous runs and one completed locked test receipt."""
    sources = Sources(root)
    base = "results/experiments"
    lock = sources.json(f"{base}/final_selection.json")
    receipt = sources.json(f"{base}/final_test_attempt.json")
    selected = lock["run"]
    local_path(root, selected, ("results", "experiments", "resnet50"))
    if lock["status"] != "LOCKED_FOR_FINAL_TEST" or receipt["status"] != "COMPLETE" or receipt["run"] != selected:
        raise ValueError("Require completed official test and locked ResNet50")
    tests = {p.parent.parent.relative_to(root).as_posix() for p in (root/base).glob("**/test/metrics.json")}
    if tests != {selected}:
        raise ValueError("Official test metrics found outside the locked final model, or missing")
    splits, participants = {}, {}
    for partition, expected_hash in lock["split_hashes"].items():
        path = f"data/splits/{partition}.csv"
        sources.check(path, expected_hash)
        rows = sources.csv(path)
        if len({r["canonical_image_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate split image")
        if any(r["final_partition"] != partition or r["official_partition"] != ("test" if partition == "test" else "train") for r in rows):
            raise ValueError("Official split partition mismatch")
        splits[partition] = rows
        participants[partition] = {r["participant_id"] for r in rows}
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Incomplete split definitions")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if participants[a] & participants[b] or {r["canonical_image_id"] for r in splits[a]} & {r["canonical_image_id"] for r in splits[b]}:
            raise ValueError("Participant/image split leakage")
    runs = {}
    for path in sorted((root/base).glob("*/*/run_metadata.json")):
        relative = path.parent.relative_to(root).as_posix()
        metadata = sources.json(f"{relative}/run_metadata.json")
        model = metadata["model"]
        if model not in MODELS or model in runs or metadata["status"] != "COMPLETE" or metadata["selection_only"] is not True:
            raise ValueError("Incomplete/ambiguous model list: require one complete validation-only run per model")
        if metadata["split_hashes"] != lock["split_hashes"] or metadata["cache_fingerprint"] != lock["cache_fingerprint"]:
            raise ValueError("Experiment cohorts/cache differ")
        for file, key in (("threshold.json", "threshold_sha256"), ("config_used.yaml", "config_sha256"),
                          ("validation/metrics.json", "validation_metrics_sha256"), ("validation/predictions.csv", "validation_predictions_sha256")):
            sources.check(f"{relative}/{file}", metadata[key])
        for partition in ("train", "validation"):
            sources.check(f"data/processed/manifests/{partition}_processed.csv", metadata["processed_manifest_sha256"][partition])
        threshold = sources.json(f"{relative}/threshold.json")
        metrics = sources.json(f"{relative}/validation/metrics.json")
        if (threshold["partition"] != "validation" or threshold["threshold"] != metrics["threshold"]
                or threshold["checkpoint_sha256"] != metadata["checkpoint_sha256"]
                or threshold["validation_manifest_sha256"] != metadata["processed_manifest_sha256"]["validation"]):
            raise ValueError("Validation threshold binding mismatch")
        predictions = sources.csv(f"{relative}/validation/predictions.csv")
        verify_predictions(predictions, metrics, splits["validation"], "validation")
        calibration = sources.json(f"{relative}/validation/calibration.json")
        close(calibration["brier_score"], metrics["brier_score"], "calibration Brier")
        runs[model] = dict(run=relative, metadata=metadata, metrics=metrics, calibration=calibration)
    if set(runs) != set(MODELS) or runs["resnet50"]["run"] != selected:
        raise ValueError("Incomplete model list or selected ResNet50 mismatch")
    metadata = runs["resnet50"]["metadata"]
    sources.check(f"{selected}/run_metadata.json", lock["run_metadata_sha256"])
    for key in ("checkpoint_sha256", "threshold_sha256", "config_sha256", "selection_participants"):
        if metadata[key] != lock[key]:
            raise ValueError(f"Selected ResNet50 differs from final_selection.json: {key}")
    if set(lock["selection_participants"]) != participants["train"] | participants["validation"]:
        raise ValueError("Locked selection participant mismatch")
    sources.check(f"{selected}/test/metrics.json", receipt["metrics_sha256"])
    sources.check("data/processed/manifests/test_processed.csv", receipt["test_manifest_sha256"])
    test = sources.json(f"{selected}/test/metrics.json")
    if test["threshold"] != lock["threshold"] or runs["resnet50"]["metrics"]["threshold"] != lock["threshold"]:
        raise ValueError("Test/validation threshold differs from locked threshold")
    predictions = sources.csv(f"{selected}/test/predictions.csv")
    verify_predictions(predictions, test, splits["test"], "test")
    calibration = sources.json(f"{selected}/test/calibration.json")
    close(calibration["brier_score"], test["brier_score"], "test calibration Brier")
    bootstrap = sources.json(f"{selected}/test/bootstrap_intervals.json")
    if (bootstrap["method"] != "participant_cluster_percentile" or bootstrap["iterations"] != 2000
            or bootstrap["seed"] != 42 or bootstrap["confidence"] != .95 or bootstrap["threshold_fixed"] != lock["threshold"]):
        raise ValueError("Unexpected saved bootstrap protocol")
    histories = {model: sources.csv(runs[model]["run"] + "/training_history.csv") for model in ("vgg16", "resnet50", "efficientnetb0")}
    return dict(sources=sources, runs=runs, lock=lock, test=test, predictions=predictions,
                calibration=calibration, bootstrap=bootstrap, splits=splits, histories=histories)
