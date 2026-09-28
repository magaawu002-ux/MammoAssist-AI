"""Synthetic cache-to-run integration; all generated artifacts stay in pytest temp."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from test_preprocessing import cohort  # Reuse the existing fully audited synthetic DICOM fixture.
from src.evaluation.comparison import compare_runs
from src.evaluation.evaluate import completed_run, evaluate_final, lock_final
from src.preprocessing.cache import build_cache
from src.training.config import load_experiment, MODELS
from src.training.data import ProcessedSplit, load_canonical, load_selection_data, make_dataset, training_class_weights
from src.training.train import fit_neural, run_training
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT


@pytest.fixture
def cached_cohort(cohort):
    shutil.copytree(PROJECT_ROOT / "config/experiments", cohort / "config/experiments")
    build_cache("train", cohort)
    build_cache("validation", cohort)
    return cohort


@pytest.mark.parametrize("name", MODELS)
def test_experiment_configuration(name):
    config = load_experiment(name)
    assert config["model"] == name and "input_size" not in config and "random_seed" not in config
    assert config["augmentation"] == "none"


def test_loader_never_reads_test_or_raw_dicoms(cached_cohort, monkeypatch):
    import src.training.data as module
    original = module.read_rows
    def guarded(path):
        assert "test" not in path.name
        return original(path)
    monkeypatch.setattr(module, "read_rows", guarded)
    def forbidden(*args, **kwargs):
        pytest.fail("Training loader must never open raw DICOMs")
    monkeypatch.setattr("pydicom.dcmread", forbidden)
    train, validation = load_selection_data(cached_cohort, load_config())
    assert train.partition == "train" and validation.partition == "validation"
    assert train.images.shape == (1, 224, 224)
    assert not (cached_cohort / "data/processed/manifests/test_processed.csv").exists()


@pytest.mark.parametrize("variant", ["shape", "nan", "inf", "range", "dtype"])
def test_cache_array_validation(tmp_path, variant):
    array = np.zeros((224, 224), dtype=np.float32)
    if variant == "shape": array = array[:100]
    elif variant == "nan": array[0, 0] = np.nan
    elif variant == "inf": array[0, 0] = np.inf
    elif variant == "range": array[0, 0] = 1.1
    else: array = array.astype(np.float64)
    path = tmp_path / "synthetic.npy"
    np.save(path, array)
    with pytest.raises(ValueError):
        load_canonical(path, 224)


def test_training_class_weights_training_only():
    assert training_class_weights(np.array([0, 0, 0, 1]), partition="train") == {0: 2/3, 1: 2}
    with pytest.raises(ValueError):
        training_class_weights(np.array([0, 2]), partition="train")
    with pytest.raises(ValueError, match="training labels only"):
        training_class_weights(np.array([0, 1]), partition="test")


def synthetic_split(partition):
    labels = np.array([0, 1, 0, 1])
    images = np.stack([np.full((32, 32), value / 4, dtype=np.float32) for value in range(4)])
    return ProcessedSplit(partition, [], images, labels, "synthetic", "synthetic", {})


def test_deterministic_training_shuffle_and_validation_order():
    train = synthetic_split("train")
    def sequence():
        return np.concatenate([x.numpy()[:, 0, 0, 0] for x, _ in make_dataset(train, "mammocnn_baseline", 2, 42, shuffle=True)])
    np.testing.assert_array_equal(sequence(), sequence())
    validation = synthetic_split("validation")
    order = np.concatenate([x.numpy()[:, 0, 0, 0] for x, _ in make_dataset(validation, "mammocnn_baseline", 2, 42)])
    np.testing.assert_array_equal(order, [0, 0.25, 0.5, 0.75])
    with pytest.raises(ValueError):
        make_dataset(validation, "mammocnn_baseline", 2, 42, shuffle=True)


def test_tiny_synthetic_training_callbacks_checkpoint(tmp_path):
    """One tiny CPU smoke epoch, never a real model result or repository checkpoint."""
    from src.training.runtime import seed_runtime
    seed_runtime(42)
    config = load_experiment("mammocnn_baseline")
    config["epochs"] = 1
    project = load_config() | {"input_size": 32}
    model, history, description = fit_neural(config, project, synthetic_split("train"), synthetic_split("validation"), tmp_path)
    assert len(history) == 1 and np.isfinite(history[0]["val_loss"])
    assert (tmp_path / "best_model.keras").is_file()
    assert model.output_shape == (None, 1) and description["stage_a"]["total_parameters"] > 0


def test_majority_run_comparison_and_explicit_final_safety(cached_cohort):
    root = cached_cohort
    first = run_training("majority_reference", root)["run_directory"]
    second = run_training("majority_reference", root)["run_directory"]
    assert len(compare_runs(root, [first, second])) == 2
    assert not (root / first / "test").exists()
    with pytest.raises(ValueError, match="lock-final"):
        evaluate_final(root, first)
    lock = lock_final(root, first, "Synthetic validation-only integration test")
    assert lock["status"] == "LOCKED_FOR_FINAL_TEST"
    assert not (root / "data/processed/manifests/test_processed.csv").exists()
    with pytest.raises(ValueError, match="locked"):
        run_training("majority_reference", root)
    with pytest.raises(ValueError, match="manifest missing"):
        evaluate_final(root, first)
    # Explicitly exercise the final path only on the same synthetic fixture.
    build_cache("test", root)
    assert evaluate_final(root, first)["partition"] == "test"
    with pytest.raises(ValueError, match="already recorded"):
        evaluate_final(root, first)
    with (root / first / "threshold.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="artifact changed"):
        completed_run(root, first)


@pytest.fixture
def empty_project(tmp_path):
    """Missing-artifact checks must not depend on or read the real project data."""
    shutil.copytree(PROJECT_ROOT / "config", tmp_path / "config")
    return tmp_path


def test_missing_processed_manifests_refusal(empty_project):
    with pytest.raises(ValueError, match="manifest missing"):
        run_training("mammocnn_baseline", root=empty_project, dry_run=True)


@pytest.mark.parametrize("module,args,message", [
    ("src.training.train", ["--experiment", "mammocnn_baseline", "--dry-run"], "Training refused"),
    ("src.evaluation.evaluate", ["--experiment", "results/experiments/missing", "--evaluate-final"], "Evaluation refused")])
def test_cli_refuses_without_real_artifacts(module, args, message, empty_project):
    # Override the CLI root in the child process, not production code or assertions.
    script = ("import importlib,sys; from pathlib import Path; "
              "m=importlib.import_module(sys.argv[1]); root=Path(sys.argv[2]); "
              "sys.argv=[sys.argv[1],*sys.argv[3:]]; "
              "m.PROJECT_ROOT=root; "
              "original=getattr(m,'run_training',None); "
              "m.run_training=lambda experiment,**kw: original(experiment,root=root,**kw); m.main()")
    result = subprocess.run([sys.executable, "-c", script, module, str(empty_project), *args],
                            cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and message in result.stderr


def test_evaluation_cli_requires_explicit_action():
    result = subprocess.run([sys.executable, "-m", "src.evaluation.evaluate", "--experiment", "missing"],
                             cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and "required" in result.stderr
