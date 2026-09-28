"""Compact related CNNs and two-stage Keras application builders."""

from typing import Any

from src.training.config import TRANSFER_MODELS


def build_model(name: str, input_size: int, *, weights: str | None = None,
                dropout: float = 0.3, l2: float = 0.0001) -> Any:
    """Default to no downloads; real runners explicitly pass configured weights."""
    import tensorflow as tf
    layers = tf.keras.layers
    if input_size < 32:
        raise ValueError("Model input_size must be at least 32")
    if name in ("mammocnn_baseline", "mammocnn_regularised"):
        regularised = name == "mammocnn_regularised"
        inputs = layers.Input((input_size, input_size, 1))
        x = inputs
        for filters in (16, 32, 64):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=not regularised,
                kernel_regularizer=tf.keras.regularizers.l2(l2) if regularised else None)(x)
            if regularised:
                x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D()(x)
        x = layers.GlobalAveragePooling2D()(x)
        if regularised:
            x = layers.Dropout(dropout)(x)
        return tf.keras.Model(inputs, layers.Dense(1, activation="sigmoid")(x), name=name)
    if name not in TRANSFER_MODELS:
        raise ValueError(f"Unknown neural architecture: {name}")
    applications = {"vgg16": tf.keras.applications.VGG16, "resnet50": tf.keras.applications.ResNet50,
                    "efficientnetb0": tf.keras.applications.EfficientNetB0}
    base = applications[name](include_top=False, weights=weights, input_shape=(input_size, input_size, 3))
    base.trainable = False
    inputs = layers.Input((input_size, input_size, 3))
    # Keep pretrained BN running statistics fixed during both stages.
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    return tf.keras.Model(inputs, layers.Dense(1, activation="sigmoid")(x), name=name)


def unfreeze_upper_layers(model: Any, boundary: str) -> list[str]:
    """Unfreeze from a named upper layer; keep all base BatchNorm layers frozen."""
    import tensorflow as tf
    bases = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)]
    if len(bases) != 1:
        raise ValueError("Expected one transfer-learning base")
    base = bases[0]
    names = [layer.name for layer in base.layers]
    if boundary not in names:
        raise ValueError(f"Fine-tuning boundary absent: {boundary}")
    index = names.index(boundary)
    base.trainable = True
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= index and not isinstance(layer, tf.keras.layers.BatchNormalization)
    return [layer.name for layer in base.layers if layer.trainable]


def model_description(model: Any) -> dict:
    """Measure parameter counts and declared shapes without training."""
    import numpy as np
    return {"total_parameters": model.count_params(),
            "trainable_parameters": sum(int(np.prod(w.shape)) for w in model.trainable_weights),
            "input_shape": list(model.input_shape), "output_shape": list(model.output_shape)}
