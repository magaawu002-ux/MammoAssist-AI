"""Offline Grad-CAM tests, including an analytical gradient and a real ResNet graph."""

import socket

import numpy as np
import pytest
import tensorflow as tf

from src.data.provenance import sha256_file
from src.explainability.gradcam import ResNet50GradCAM, TARGET_LAYER
from src.explainability.overlay import overlay_heatmap
from src.models.builders import build_model
from src.preprocessing.model_inputs import transfer_input


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Grad-CAM tests must not access the network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    yield
    tf.keras.backend.clear_session()


def toy_classifier(kernel=0.001):
    """Known linear feature/head geometry with the same nested terminal-layer contract."""
    inputs = tf.keras.Input((224, 224, 3))
    x = tf.keras.layers.AveragePooling2D(32)(inputs)
    x = tf.keras.layers.Conv2D(2, 1, activation="relu", kernel_initializer="ones", bias_initializer="zeros", name=TARGET_LAYER)(x)
    base = tf.keras.Model(inputs, x, name="resnet50")
    base.trainable = False
    inputs = tf.keras.Input((224, 224, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.8)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer=tf.keras.initializers.Constant(kernel))(x)
    return tf.keras.Model(inputs, output)


@pytest.fixture
def explainer():
    return ResNet50GradCAM(toy_classifier())


@pytest.mark.parametrize("image", [np.zeros((224, 224, 1), np.float32), np.zeros((223, 224), np.float32),
    np.zeros((1, 224, 224), np.float32), np.zeros((224, 224), np.float64), np.zeros((224, 224), np.uint8),
    np.full((224, 224), np.nan, np.float32), np.full((224, 224), np.inf, np.float32),
    np.full((224, 224), -0.01, np.float32), np.full((224, 224), 1.01, np.float32)])
def test_invalid_canonical_images(explainer, image):
    with pytest.raises(ValueError):
        explainer.explain(image)


def test_analytical_gradient_probability_and_determinism(explainer):
    image = np.linspace(0.6, 0.9, 224 * 224, dtype=np.float32).reshape(224, 224)
    before = [v.numpy().copy() for v in explainer.model.weights]
    flags = [l.trainable for l in explainer.model.layers]
    result = explainer.explain(image)
    ordinary = float(explainer.model(transfer_input(image, "resnet50")[None], training=False).numpy()[0, 0])
    assert result.malignant_probability == pytest.approx(ordinary, abs=1e-6)
    assert result.target_layer == TARGET_LAYER
    assert result.heatmap.shape == (224, 224) and result.heatmap.dtype == np.float32
    assert np.isfinite(result.heatmap).all() and 0 <= result.heatmap.min() <= result.heatmap.max() <= 1
    assert result.gradient_l1 > 0 and not result.zero_heatmap
    features = explainer.feature_model(transfer_input(image, "resnet50")[None], training=False).numpy()[0]
    # dp/dA_ijk = p(1-p)*w_k/49 for the known GAP + sigmoid head.
    expected = np.sum(features * (ordinary * (1-ordinary) * 0.001 / 49), axis=-1)
    expected /= expected.max()
    expected = tf.image.resize(expected[..., None], (224, 224)).numpy()[..., 0]
    np.testing.assert_allclose(result.heatmap, expected, atol=1e-6)
    np.testing.assert_array_equal(result.heatmap, explainer.explain(image).heatmap)
    for a, b in zip(before, explainer.model.weights):
        np.testing.assert_array_equal(a, b.numpy())
    assert flags == [l.trainable for l in explainer.model.layers]


@pytest.mark.parametrize("kernel", [0.0, -0.001])
def test_zero_or_negative_class_evidence_has_zero_heatmap(kernel):
    explainer = ResNet50GradCAM(toy_classifier(kernel))
    result = explainer.explain(np.full((224, 224), 0.8, np.float32))
    assert result.zero_heatmap
    np.testing.assert_array_equal(result.heatmap, 0)
    assert np.isfinite(result.heatmap).all()


def test_resnet50_nested_graph_frozen_checkpoint(tmp_path):
    tf.keras.utils.set_random_seed(42)
    model = build_model("resnet50", 224, weights=None)
    model.trainable = False
    path = tmp_path / "synthetic.keras"
    model.save(path)
    original_hash = sha256_file(path)
    explainer = ResNet50GradCAM.from_checkpoint(path)
    image = np.full((224, 224), 0.5, np.float32)
    result = explainer.explain(image)
    ordinary = float(explainer.model(transfer_input(image, "resnet50")[None], training=False).numpy()[0, 0])
    assert result.feature_map_shape == (7, 7, 2048)
    assert result.target_layer == "conv5_block3_out"
    assert result.malignant_probability == pytest.approx(ordinary, abs=1e-6)
    assert sha256_file(path) == original_hash


def test_unsupported_target_and_architecture():
    with pytest.raises(ValueError, match="target layer"):
        ResNet50GradCAM(toy_classifier(), target_layer="other")
    with pytest.raises(ValueError):
        ResNet50GradCAM(build_model("mammocnn_baseline", 224))


def test_overlay_shape_dtype_and_base_preservation():
    base = np.full((224, 224), 0.6, np.float32)
    heatmap = np.zeros_like(base)
    heatmap[20:40, 20:40] = 1
    output = overlay_heatmap(base, heatmap)
    assert output.shape == (224, 224, 3) and output.dtype == np.float32
    assert np.isfinite(output).all() and output.min() >= 0 and output.max() <= 1
    np.testing.assert_array_equal(output[0, 0], [np.float32(0.6)] * 3)
    np.testing.assert_array_equal(overlay_heatmap(base, heatmap, alpha=0), np.repeat(base[..., None], 3, axis=-1))
    assert output[25, 25, 0] > output[25, 25, 2]


@pytest.mark.parametrize("alpha", [-1, 1.1, np.nan])
def test_overlay_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        overlay_heatmap(np.zeros((4, 4)), np.zeros((4, 4)), alpha)
