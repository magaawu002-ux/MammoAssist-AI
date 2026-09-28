"""Probability-targeted Grad-CAM for the selected nested ResNet50 classifier."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.preprocessing.model_inputs import transfer_input
from src.utils.config import load_config

TARGET_LAYER = "conv5_block3_out"


def validate_image(image: np.ndarray, size: int) -> None:
    """Require the canonical single-mammogram contract without silent conversion."""
    if not isinstance(image, np.ndarray) or image.shape != (size, size):
        raise ValueError(f"Expected a canonical ({size}, {size}) grayscale mammogram")
    if image.dtype != np.float32:
        raise ValueError("Canonical mammogram dtype must be float32")
    if not np.isfinite(image).all() or np.any((image < 0) | (image > 1)):
        raise ValueError("Canonical mammogram must be finite and within [0,1]")


@dataclass(frozen=True)
class GradCAMResult:
    malignant_probability: float
    heatmap: np.ndarray
    target_layer: str
    feature_map_shape: tuple[int, ...]
    gradient_l1: float
    raw_positive_max: float
    zero_heatmap: bool


class ResNet50GradCAM:
    """Explain the terminal nested feature tensor through the original head.

    Deliberately supports the repository's Input -> ResNet50 -> GAP -> Dropout ->
    sigmoid Dense architecture only. No arbitrary intermediate-layer replay or
    intensity fallback is allowed. Model weights, trainability and activations
    are never modified. The gradient target is P(MALIGNANT), not a logit.
    """

    def __init__(self, model: tf.keras.Model, target_layer: str = TARGET_LAYER,
                 input_size: int | None = None) -> None:
        self.input_size = load_config()["input_size"] if input_size is None else input_size
        if type(self.input_size) is not int or self.input_size < 32:
            raise ValueError("input_size must be an integer >= 32")
        if target_layer != TARGET_LAYER:
            raise ValueError(f"This validated architecture requires target layer {TARGET_LAYER}")
        if model.input_shape != (None, self.input_size, self.input_size, 3) or model.output_shape != (None, 1):
            raise ValueError("Expected one RGB input and one binary probability output")
        layers = model.layers
        if (len(layers) != 5 or not isinstance(layers[0], tf.keras.layers.InputLayer)
                or not isinstance(layers[1], tf.keras.Model) or layers[1].name != "resnet50"
                or not isinstance(layers[2], tf.keras.layers.GlobalAveragePooling2D)
                or not isinstance(layers[3], tf.keras.layers.Dropout)
                or not isinstance(layers[4], tf.keras.layers.Dense)
                or layers[4].units != 1 or layers[4].activation != tf.keras.activations.sigmoid
                or layers[2].data_format != "channels_last"):
            raise ValueError("Unsupported classifier topology; expected nested ResNet50/GAP/Dropout/sigmoid")
        backbone = layers[1]
        target = backbone.get_layer(target_layer)
        if target.output is not backbone.output or len(target.output.shape) != 4:
            raise ValueError("Target must be the backbone's terminal spatial feature tensor")
        # Prove the head consumes the backbone output in the expected sequence.
        # These are public layer input/output tensors, not internal inbound-node APIs.
        for previous, current in zip(layers[2:], layers[3:]):
            if current.input is not previous.output:
                raise ValueError("Classifier head must form a single connected sequence")
        head_probe = tf.keras.Model(layers[2].input, model.output)
        self.feature_model = tf.keras.Model(backbone.inputs[0], target.output, name="gradcam_resnet50_features")
        self.head = head_probe
        self.target_layer = target_layer
        self.model = model

    @classmethod
    def from_checkpoint(cls, checkpoint: Path, *, input_size: int | None = None) -> "ResNet50GradCAM":
        """Load inference-only; never compile, train, download weights or save a model."""
        return cls(tf.keras.models.load_model(checkpoint, compile=False, safe_mode=True), input_size=input_size)

    def explain(self, image: np.ndarray) -> GradCAMResult:
        """Compute spatially averaged score gradients, weighted maps, ReLU and scaling."""
        validate_image(image, self.input_size)
        inputs = tf.convert_to_tensor(transfer_input(image, "resnet50")[None, ...])
        features = self.feature_model(inputs, training=False)
        # Explicit watching works even when the entire backbone/head is frozen.
        # Only the head needs differentiation for this terminal target tensor.
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(features)
            probability = self.head(features, training=False)
            score = probability[0, 0]
        gradients = tape.gradient(score, features)
        if gradients is None:
            raise ValueError("Disconnected Grad-CAM gradient; no explanation can be produced")
        if not all(bool(tf.reduce_all(tf.math.is_finite(v))) for v in (features, gradients, probability)):
            raise ValueError("Nonfinite features, gradients or probability")
        value = float(score.numpy())
        if not 0 <= value <= 1:
            raise ValueError("Model output is not a probability")
        weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
        positive = tf.nn.relu(tf.reduce_sum(features[0] * weights, axis=-1))
        maximum = tf.reduce_max(positive)
        normalized = tf.math.divide_no_nan(positive, maximum)
        resized = tf.image.resize(normalized[..., None], (self.input_size, self.input_size), method="bilinear")
        heatmap = np.clip(resized[..., 0].numpy(), 0, 1).astype(np.float32)
        if not np.isfinite(heatmap).all():
            raise ValueError("Nonfinite resized heatmap")
        return GradCAMResult(value, heatmap, self.target_layer, tuple(int(v) for v in features.shape[1:]),
                             float(tf.reduce_sum(tf.abs(gradients)).numpy()), float(maximum.numpy()),
                             bool(float(maximum.numpy()) == 0))
