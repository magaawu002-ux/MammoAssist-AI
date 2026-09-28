"""Training callbacks, explicit seeding and computational provenance."""

import platform
import subprocess
from pathlib import Path

import numpy as np


def seed_runtime(seed: int) -> dict:
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    return {"seed": seed, "deterministic_ops": True,
            "note": "Reproducibility still depends on hardware and library versions"}


def environment(root: Path) -> dict:
    import tensorflow as tf
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
            "tensorflow": tf.__version__, "keras": tf.keras.__version__, "numpy": np.__version__,
            "devices": [str(d) for d in tf.config.list_physical_devices()], "git_commit": commit}


def compile_model(model, learning_rate: float) -> None:
    import tensorflow as tf
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="binary_crossentropy")


def training_callbacks(config: dict, checkpoint: Path) -> list:
    import tensorflow as tf
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                    patience=config["early_stopping_patience"], restore_best_weights=True),
                 tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", mode="min", save_best_only=True),
                 tf.keras.callbacks.TerminateOnNaN()]
    reduce = config["reduce_lr"]
    if reduce["enabled"]:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=reduce["factor"],
            patience=reduce["patience"], min_lr=reduce["min_learning_rate"]))
    return callbacks
