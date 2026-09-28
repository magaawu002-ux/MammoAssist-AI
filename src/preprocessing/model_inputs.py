"""Adapters only: never instantiate models, download weights or mutate caches."""

import numpy as np


def canonical_array(image: np.ndarray) -> np.ndarray:
    """Accept a single cached HxW grayscale array in [0,1]."""
    array = np.asarray(image)
    if array.ndim != 2 or array.size == 0 or array.dtype.kind not in "uif":
        raise ValueError("Expected a nonempty numeric HxW canonical grayscale array")
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError("Canonical input must be finite and in [0,1]")
    return array.astype(np.float32, copy=True)


def custom_cnn_input(image: np.ndarray) -> np.ndarray:
    """Return HxWx1 float32, still in [0,1]."""
    return canonical_array(image)[..., None]


def replicated_rgb(image: np.ndarray) -> np.ndarray:
    """Return three identical channels in [0,255] before family preprocessing."""
    return np.repeat(custom_cnn_input(image), 3, axis=-1) * np.float32(255)


def transfer_input(image: np.ndarray, family: str) -> np.ndarray:
    """Use Keras VGG16/ResNet50 transforms; EfficientNetB0 keeps [0,255].

    Keras EfficientNetB0 includes its own rescaling layer. Do not apply another
    division by 255 to this adapter output when using that application model.
    """
    if family not in ("vgg16", "resnet50", "efficientnetb0"):
        raise ValueError(f"Unsupported model family: {family}")
    from tensorflow.keras.applications import efficientnet, resnet50, vgg16
    functions = {"vgg16": vgg16.preprocess_input, "resnet50": resnet50.preprocess_input,
                 "efficientnetb0": efficientnet.preprocess_input}
    return np.asarray(functions[family](replicated_rgb(image), data_format="channels_last"), dtype=np.float32)
