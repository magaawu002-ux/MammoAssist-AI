"""Explicit final selection and one-time holdout evaluation; no training shortcut."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from src.data.provenance import sha256_file
from src.evaluation.bootstrap import bootstrap_intervals
from src.evaluation.outputs import save_evaluation, write_json
from src.preprocessing.manifests import local_path
from src.training.data import load_final_test, load_selection_data, make_dataset
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT


def completed_run(root: Path, run: str) -> tuple[Path, dict, dict, dict]:
    directory = local_path(root, run, ("results", "experiments"))
    metadata_path = directory / "run_metadata.json"
    if not metadata_path.is_file():
        raise ValueError("Completed experiment does not exist")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "COMPLETE" or metadata.get("selection_only") is not True:
        raise ValueError("Run must be complete and selected without test evaluation")
    checkpoint_name = metadata["checkpoint"]
    if Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("Invalid checkpoint filename")
    for file, key in ((checkpoint_name, "checkpoint_sha256"), ("threshold.json", "threshold_sha256"),
                      ("config_used.yaml", "config_sha256"), ("validation/predictions.csv", "validation_predictions_sha256"),
                      ("validation/metrics.json", "validation_metrics_sha256")):
        if sha256_file(directory / file) != metadata[key]:
            raise ValueError(f"Completed artifact changed: {file}")
    threshold = json.loads((directory / "threshold.json").read_text(encoding="utf-8"))
    if (threshold.get("partition") != "validation" or threshold.get("checkpoint_sha256") != metadata["checkpoint_sha256"]
            or threshold.get("validation_manifest_sha256") != metadata["processed_manifest_sha256"]["validation"]):
        raise ValueError("Threshold is not bound to this checkpoint and validation cohort")
    return directory, metadata, threshold, load_config(directory / "config_used.yaml")


def lock_final(root: Path, run: str, rationale: str) -> dict:
    """Freeze one globally selected run; selection rationale must be validation-only."""
    if not rationale.strip():
        raise ValueError("Record the validation-based final selection rationale")
    directory, metadata, threshold, config = completed_run(root, run)
    train, validation = load_selection_data(root, config["project"])
    if {"train": train.manifest_sha256, "validation": validation.manifest_sha256} != metadata["processed_manifest_sha256"]:
        raise ValueError("Selection manifests changed since training")
    lock = {"status": "LOCKED_FOR_FINAL_TEST", "run": directory.relative_to(root).as_posix(),
        "rationale": rationale, "checkpoint_sha256": metadata["checkpoint_sha256"],
        "threshold_sha256": metadata["threshold_sha256"], "config_sha256": metadata["config_sha256"],
        "run_metadata_sha256": sha256_file(directory / "run_metadata.json"),
        "threshold": threshold["threshold"], "split_hashes": metadata["split_hashes"],
        "cache_fingerprint": metadata["cache_fingerprint"], "selection_participants": metadata["selection_participants"]}
    destination = root / "results/experiments/final_selection.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, allow_nan=False)
    return lock


def evaluate_final(root: Path, run: str, *, bootstrap_iterations: int = 0) -> dict:
    """Require explicit frozen selection; block repeated test attempts globally."""
    directory, metadata, threshold, config = completed_run(root, run)
    lock_path = root / "results/experiments/final_selection.json"
    if not lock_path.is_file():
        raise ValueError("Final evaluation requires explicit --lock-final first")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock.get("run") != directory.relative_to(root).as_posix() or
            lock.get("run_metadata_sha256") != sha256_file(directory / "run_metadata.json") or
            lock.get("checkpoint_sha256") != metadata["checkpoint_sha256"] or
            lock.get("threshold_sha256") != metadata["threshold_sha256"] or
            lock.get("config_sha256") != metadata["config_sha256"] or lock.get("threshold") != threshold["threshold"]):
        raise ValueError("Locked model, config or threshold changed")
    receipt = root / "results/experiments/final_test_attempt.json"
    if receipt.exists():
        raise ValueError("Final test attempt already recorded; repeated test-driven selection is prohibited")
    if bootstrap_iterations != 0 and bootstrap_iterations < 2:
        raise ValueError("Bootstrap iterations must be zero (disabled) or >= 2")
    data = load_final_test(root, config["project"], lock)
    with receipt.open("x", encoding="utf-8") as handle:
        json.dump({"status": "STARTED", "run": run, "test_manifest_sha256": data.manifest_sha256}, handle)
    start = perf_counter()
    if metadata["model"] == "majority_reference":
        from src.models.majority import MajorityReference
        model = MajorityReference()
        model.majority_class = json.loads((directory / metadata["checkpoint"]).read_text())["majority_class"]
        probabilities = model.predict_probability(len(data.labels))
    else:
        import tensorflow as tf
        model = tf.keras.models.load_model(directory / metadata["checkpoint"], compile=False)
        dataset = make_dataset(data, metadata["model"], config["experiment"]["batch_size"], config["project"]["random_seed"])
        probabilities = model.predict(dataset, verbose=0).reshape(-1)
    elapsed = perf_counter() - start
    metrics = save_evaluation(directory / "test", data.rows, probabilities, threshold["threshold"], partition="test")
    if bootstrap_iterations:
        intervals = bootstrap_intervals(data.labels, probabilities, [r["participant_id"] for r in data.rows],
            threshold["threshold"], seed=config["project"]["random_seed"], iterations=bootstrap_iterations)
        write_json(directory / "test/bootstrap_intervals.json", intervals)
    write_json(directory / "test/timing.json", {"seconds": elapsed, "seconds_per_image": elapsed / len(data.labels),
        "scope": "includes checkpoint load, cold inference, adapters and batching"})
    write_json(receipt, {"status": "COMPLETE", "run": run, "test_manifest_sha256": data.manifest_sha256,
                        "metrics_sha256": sha256_file(directory / "test/metrics.json")})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Repository-relative completed run directory")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--lock-final", action="store_true")
    actions.add_argument("--evaluate-final", action="store_true")
    actions.add_argument("--dry-run", action="store_true", help="Check completed selection artifacts only; never open test")
    parser.add_argument("--rationale", default="")
    parser.add_argument("--bootstrap-iterations", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.lock_final:
            print(lock_final(PROJECT_ROOT, args.experiment, args.rationale))
        elif args.evaluate_final:
            print(evaluate_final(PROJECT_ROOT, args.experiment, bootstrap_iterations=args.bootstrap_iterations))
        else:
            completed_run(PROJECT_ROOT, args.experiment)
            print("Completed selection artifacts verified; test not accessed")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Evaluation refused: {exc}\n")


if __name__ == "__main__":
    main()
