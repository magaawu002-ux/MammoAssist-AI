"""Setup checks only: no datasets, model construction or training."""

import os
from pathlib import Path
import random

import numpy as np
import pytest

from src.utils import paths
from src.utils.config import load_config
from src.utils.reproducibility import set_reproducibility


REQUIRED_DIRS = (
    "config", "data", "data/raw", "data/interim", "data/processed",
    "data/splits", "models", "models/checkpoints", "results",
    "results/figures", "results/metrics", "results/predictions",
    "reports", "reports/evidence", "src", "src/data", "src/preprocessing",
    "src/models", "src/training", "src/evaluation", "src/explainability",
    "src/app", "src/utils", "tests",
)


@pytest.mark.parametrize("relative_path", REQUIRED_DIRS)
def test_required_directory_exists(relative_path):
    assert (paths.PROJECT_ROOT / relative_path).is_dir()


def test_project_configuration():
    assert load_config() == {
        "project_name": "MammoAssist AI",
        "dataset": "CBIS-DDSM",
        "subset": "mass",
        "task": "binary_classification",
        "image_type": "full_mammogram",
        "positive_class": "MALIGNANT",
        "input_size": 224,
        "random_seed": 42,
        "preprocessing": {
            "version": "canonical_v1",
            "crop_background": True,
            "crop_margin_fraction": 0.02,
            "cache_directory": "data/processed/cache",
            "cache_dtype": "float32",
        },
    }


def test_all_project_paths_are_under_root():
    assert paths.PROJECT_ROOT == Path(__file__).resolve().parents[1]
    defined_paths = {
        name: value for name, value in vars(paths).items()
        if isinstance(value, Path) and name != "PROJECT_ROOT"
    }
    assert len(defined_paths) == 14
    for name, value in defined_paths.items():
        assert value.is_absolute(), name
        assert value.resolve().is_relative_to(paths.PROJECT_ROOT), name
        assert value.resolve() != paths.PROJECT_ROOT, name


def test_source_packages():
    for relative_path in REQUIRED_DIRS:
        if relative_path == "src" or relative_path.startswith("src/"):
            assert (paths.PROJECT_ROOT / relative_path / "__init__.py").is_file()


def test_reproducibility_executes_and_repeats():
    import tensorflow as tf

    expected_seed = load_config()["random_seed"]
    assert set_reproducibility() == expected_seed
    assert os.environ["PYTHONHASHSEED"] == str(expected_seed)
    first = (random.random(), np.random.random(4), tf.random.uniform((4,)).numpy())
    set_reproducibility()
    second = (random.random(), np.random.random(4), tf.random.uniform((4,)).numpy())
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])


def test_config_loading_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config()["project_name"] == "MammoAssist AI"


def test_config_rejects_non_mapping(tmp_path):
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("- a list is not a configuration mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(config_file)


def test_config_rejects_executable_yaml(tmp_path):
    import yaml

    config_file = tmp_path / "unsafe.yaml"
    config_file.write_text("!!python/object:builtins.object {}\n", encoding="utf-8")
    with pytest.raises(yaml.constructor.ConstructorError):
        load_config(config_file)
