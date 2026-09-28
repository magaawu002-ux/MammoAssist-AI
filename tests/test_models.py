"""Offline architecture checks only; no ImageNet downloads or real training."""

import gc
import socket

import numpy as np
import pytest
import tensorflow as tf

from src.models.builders import build_model, model_description, unfreeze_upper_layers
from src.models.majority import MajorityReference


@pytest.fixture(autouse=True)
def offline_models(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Model unit tests must not access the network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(tf.keras.utils, "get_file", forbidden)
    yield
    tf.keras.backend.clear_session()
    gc.collect()


def test_majority_training_only_and_ties():
    model = MajorityReference().fit(np.array([0, 0, 1]), partition="train")
    np.testing.assert_array_equal(model.predict_probability(5), 0)
    assert MajorityReference().fit(np.array([0, 1]), partition="train").majority_class == 0
    for partition in ("validation", "test"):
        with pytest.raises(ValueError, match="training labels only"):
            model.fit(np.array([1, 1, 1]), partition=partition)


@pytest.mark.parametrize("name", ["mammocnn_baseline", "mammocnn_regularised", "vgg16", "resnet50", "efficientnetb0"])
def test_build_binary_models_without_weights(name):
    model = build_model(name, 32, weights=None)
    channels = 1 if name.startswith("mammocnn") else 3
    assert model.input_shape == (None, 32, 32, channels)
    assert model.output_shape == (None, 1)
    assert model.layers[-1].activation.__name__ == "sigmoid"
    probability = model(np.zeros((1, 32, 32, channels), dtype=np.float32), training=False).numpy()
    assert np.isfinite(probability).all() and 0 <= probability.item() <= 1
    assert model_description(model)["total_parameters"] > 0


def test_related_but_distinct_custom_models():
    baseline = build_model("mammocnn_baseline", 224)
    regularised = build_model("mammocnn_regularised", 224)
    convs = lambda m: [l.filters for l in m.layers if isinstance(l, tf.keras.layers.Conv2D)]
    assert convs(baseline) == convs(regularised) == [16, 32, 64]
    assert not any(isinstance(l, (tf.keras.layers.BatchNormalization, tf.keras.layers.Dropout)) for l in baseline.layers)
    assert sum(isinstance(l, tf.keras.layers.BatchNormalization) for l in regularised.layers) == 3
    assert any(isinstance(l, tf.keras.layers.Dropout) for l in regularised.layers)
    assert len(regularised.losses) == 3 and not baseline.losses


@pytest.mark.parametrize("name,boundary", [("vgg16", "block5_conv1"), ("resnet50", "conv5_block1_1_conv"),
    ("efficientnetb0", "block7a_expand_conv")])
def test_selective_fine_tuning(name, boundary):
    model = build_model(name, 32, weights=None)
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    assert not base.trainable
    changed = unfreeze_upper_layers(model, boundary)
    assert boundary in changed
    assert all(not l.trainable for l in base.layers if isinstance(l, tf.keras.layers.BatchNormalization))
    assert not base.layers[1].trainable

