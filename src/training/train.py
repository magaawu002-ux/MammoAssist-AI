"""Train and select on train/validation only; never automatically evaluate test."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import yaml

from src.data.provenance import sha256_file
from src.evaluation.outputs import save_evaluation, save_history, write_json
from src.evaluation.thresholds import select_threshold
from src.models.builders import build_model, model_description, unfreeze_upper_layers
from src.models.majority import MajorityReference
from src.training.config import load_experiment, TRANSFER_MODELS
from src.training.data import load_selection_data, make_dataset, training_class_weights
from src.training.runtime import compile_model, environment, seed_runtime, training_callbacks
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT


def fit_neural(config: dict, project: dict, train, validation, directory: Path) -> tuple[object, list[dict], dict]:
    """Save per-stage best models and select the lowest validation loss across stages."""
    import tensorflow as tf
    name = config["model"]
    model = build_model(name, project["input_size"], weights=config["transfer"]["weights"],
                        dropout=config.get("dropout", 0.3), l2=config.get("l2", 0.0001))
    summary = []
    model.summary(print_fn=lambda line: summary.append(line))
    (directory / "model_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    description = {"stage_a": model_description(model)}
    training = make_dataset(train, name, config["batch_size"], project["random_seed"], shuffle=True)
    validating = make_dataset(validation, name, config["batch_size"], project["random_seed"])
    weights = training_class_weights(train.labels, partition="train") if config["class_weight"] else None
    histories, candidates = [], []
    stages = [("head" if name in TRANSFER_MODELS else "custom", config["epochs"], config["learning_rate"])]
    if name in TRANSFER_MODELS and config["transfer"]["fine_tuning_epochs"]:
        stages.append(("fine_tuning", config["transfer"]["fine_tuning_epochs"], config["transfer"]["fine_tuning_learning_rate"]))
    for stage, epochs, lr in stages:
        if stage == "fine_tuning":
            model = tf.keras.models.load_model(candidates[0][1], compile=False)
            description["unfrozen_layers"] = unfreeze_upper_layers(model, config["transfer"]["unfreeze_from"])
            description["stage_b"] = model_description(model)
        compile_model(model, lr)  # Mandatory after changing trainable layers.
        checkpoint = directory / f"best_{stage}.keras"
        result = model.fit(training, validation_data=validating, epochs=epochs, class_weight=weights,
                           callbacks=training_callbacks(config, checkpoint), verbose=2, shuffle=False)
        if not result.history.get("val_loss") or not np.isfinite(result.history["val_loss"]).all():
            raise ValueError("Training did not produce finite validation losses")
        offset = len(histories)
        for i in range(len(result.history["loss"])):
            histories.append({"epoch": offset + i + 1, "stage": stage,
                "loss": float(result.history["loss"][i]), "val_loss": float(result.history["val_loss"][i]),
                "learning_rate": float(result.history.get("learning_rate", [lr] * len(result.history["loss"]))[i])})
        candidates.append((min(result.history["val_loss"]), checkpoint))
    best_loss, checkpoint = min(candidates, key=lambda entry: entry[0])
    shutil.copyfile(checkpoint, directory / "best_model.keras")
    description["best_validation_loss"] = float(best_loss)
    description["selected_stage_checkpoint"] = checkpoint.name
    return tf.keras.models.load_model(directory / "best_model.keras", compile=False), histories, description


def run_training(experiment: str, root: Path = PROJECT_ROOT, *, dry_run: bool = False) -> dict:
    if (root / "results/experiments/final_selection.json").exists() and not dry_run:
        raise ValueError("Final model selection is locked; further model-selection training is prohibited")
    config = load_experiment(experiment, root)
    project = load_config(root / "config/project.yaml")
    train, validation = load_selection_data(root, project)  # Fail before model construction or downloads.
    if dry_run:
        return {"status": "VALIDATED_DRY_RUN", "train_images": len(train.labels), "validation_images": len(validation.labels)}
    if len(np.unique(train.labels)) != 2 and experiment != "majority_reference":
        raise ValueError("Neural training requires both training classes")
    if config["threshold_method"] == "youden_j" and len(np.unique(validation.labels)) != 2:
        raise ValueError("Youden threshold selection requires both validation classes")
    seed_info = seed_runtime(project["random_seed"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = root / "results/experiments" / experiment / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config_used.yaml").write_text(yaml.safe_dump({"project": project, "experiment": config}, sort_keys=True), encoding="utf-8")
    metadata = {"status": "RUNNING", "model": experiment, "environment": environment(root), "reproducibility": seed_info,
        "training_class_counts": {str(i): int(np.sum(train.labels == i)) for i in (0, 1)},
        "class_weights": training_class_weights(train.labels, partition="train") if config["class_weight"] else None,
        "implementation_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
            for package in ("models", "training", "evaluation")
            for p in sorted((PROJECT_ROOT / "src" / package).glob("*.py"))},
        "processed_manifest_sha256": {"train": train.manifest_sha256, "validation": validation.manifest_sha256},
        "split_hashes": train.split_hashes, "cache_fingerprint": train.cache_fingerprint,
        "selection_participants": sorted({r["participant_id"] for r in train.rows + validation.rows}),
        "selection_only": True}
    write_json(directory / "run_metadata.json", metadata)
    start = perf_counter()
    try:
        if experiment == "majority_reference":
            model = MajorityReference().fit(train.labels, partition="train")
            checkpoint = directory / "majority.json"
            write_json(checkpoint, {"majority_class": model.majority_class, "fitted_partition": "train"})
            description = {"total_parameters": 0, "trainable_parameters": 0}
            (directory / "model_summary.txt").write_text("Training-label majority reference; tie -> benign.\n", encoding="utf-8")
            history = []
        else:
            model, history, description = fit_neural(config, project, train, validation, directory)
            checkpoint = directory / "best_model.keras"
        training_seconds = perf_counter() - start
        inference_start = perf_counter()
        if experiment == "majority_reference":
            probabilities = model.predict_probability(len(validation.labels))
        else:
            dataset = make_dataset(validation, experiment, config["batch_size"], project["random_seed"])
            probabilities = model.predict(dataset, verbose=0).reshape(-1)
        inference_seconds = perf_counter() - inference_start
        threshold = select_threshold(validation.labels, probabilities, partition="validation", method=config["threshold_method"])
        threshold.update(checkpoint_sha256=sha256_file(checkpoint), validation_manifest_sha256=validation.manifest_sha256)
        write_json(directory / "threshold.json", threshold)
        save_evaluation(directory / "validation", validation.rows, probabilities, threshold["threshold"], partition="validation")
        save_history(directory, history)
        metadata.update(status="COMPLETE", architecture=description, checkpoint=checkpoint.name,
            checkpoint_sha256=sha256_file(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
            training_seconds=training_seconds, validation_inference_seconds=inference_seconds,
            validation_inference_seconds_per_image=inference_seconds / len(validation.labels),
            timing_scope="Training includes build/save; inference includes adapter/batching and cold first-call overhead",
            threshold_sha256=sha256_file(directory / "threshold.json"),
            config_sha256=sha256_file(directory / "config_used.yaml"),
            validation_metrics_sha256=sha256_file(directory / "validation/metrics.json"),
            validation_predictions_sha256=sha256_file(directory / "validation/predictions.csv"))
        write_json(directory / "run_metadata.json", metadata)
    except (ValueError, OSError, RuntimeError) as exc:
        metadata.update(status="FAILED", failure=f"{type(exc).__name__}: {exc}")
        write_json(directory / "run_metadata.json", metadata)
        raise
    return {"status": "COMPLETE", "run_directory": directory.relative_to(root).as_posix(), "test_evaluated": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        print(run_training(args.experiment, dry_run=args.dry_run))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Training refused: {exc}\n")


if __name__ == "__main__":
    main()
