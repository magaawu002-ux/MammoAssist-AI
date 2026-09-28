"""Explicit experiment defaults, without copying project-wide constants."""

import math
from pathlib import Path

from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT

MODELS = ("majority_reference", "mammocnn_baseline", "mammocnn_regularised", "vgg16", "resnet50", "efficientnetb0")
TRANSFER_MODELS = MODELS[3:]


def load_experiment(name: str, root: Path = PROJECT_ROOT) -> dict:
    """Merge one level of named defaults and reject unsupported protocol settings."""
    if name not in MODELS:
        raise ValueError(f"Unknown experiment: {name}")
    directory = root / "config/experiments"
    config = load_config(directory / "defaults.yaml")
    override = load_config(directory / f"{name}.yaml")
    allowed = set(config) | {"model", "dropout", "l2"}
    if set(override) - allowed:
        raise ValueError("Unknown experiment settings")
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            if set(value) - set(config[key]):
                raise ValueError(f"Unknown {key} settings")
            config[key] = config[key] | value
        else:
            config[key] = value
    if config["model"] != name or config["optimizer"] != "adam" or config["loss"] != "binary_crossentropy":
        raise ValueError("Only named models, Adam and binary cross entropy are supported")
    if config["threshold_method"] not in ("youden_j", "fixed_0.5") or config["augmentation"] != "none":
        raise ValueError("Unsupported threshold/augmentation protocol")
    for key in ("batch_size", "epochs", "early_stopping_patience"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["class_weight"]) is not bool:
        raise ValueError("class_weight must be boolean")
    lr = config["learning_rate"]
    transfer, reduce = config["transfer"], config["reduce_lr"]
    if not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if (type(reduce["enabled"]) is not bool or not 0 < reduce["factor"] < 1 or
            type(reduce["patience"]) is not int or reduce["patience"] < 1 or
            not 0 < reduce["min_learning_rate"] <= lr):
        raise ValueError("Invalid learning-rate reduction settings")
    if type(transfer["fine_tuning_epochs"]) is not int or transfer["fine_tuning_epochs"] < 0:
        raise ValueError("Invalid fine-tuning epochs")
    if transfer["weights"] not in (None, "imagenet"):
        raise ValueError("weights must be null or imagenet")
    if transfer["fine_tuning_epochs"] and (name not in TRANSFER_MODELS or
            not 0 < transfer["fine_tuning_learning_rate"] <= lr / 10 or not transfer["unfreeze_from"]):
        raise ValueError("Fine-tuning requires an upper-layer boundary and at least 10x lower LR")
    if name == "mammocnn_regularised" and (not 0 <= config["dropout"] < 1 or not 0 <= config["l2"] < 1):
        raise ValueError("Invalid regularisation settings")
    return config
