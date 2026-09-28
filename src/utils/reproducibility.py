"""Seed supported random generators using the central project configuration."""

import os
import random

from src.utils.config import load_config


def set_reproducibility() -> int:
    """Set seeds and return the configured seed; do not construct or train models.

    PYTHONHASHSEED affects hash randomisation only when Python starts. Setting
    it here configures inherited environments, not this process's hash seed.
    Seeds alone do not guarantee deterministic TensorFlow operations or
    identical outputs across hardware and library versions.
    """
    seed = load_config()["random_seed"]
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("random_seed must be an integer in [0, 2**32 - 1].")

    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return seed
