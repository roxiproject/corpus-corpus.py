import numpy as np
import pytest

from corpus_corpus.mlp import MLP, Dense


def test_dense_forward_shape():
    layer = Dense(4, 6, activation="relu", rng=np.random.default_rng(0))
    x = np.random.default_rng(0).normal(size=(10, 4))
    out = layer.forward(x)
    assert out.shape == (10, 6)


def test_dense_rejects_unknown_activation():
    with pytest.raises(ValueError):
        Dense(3, 3, activation="swish")


def test_mlp_rejects_too_few_layer_sizes():
    with pytest.raises(ValueError):
        MLP([5])


def test_mlp_rejects_mismatched_activations_length():
    with pytest.raises(ValueError):
        MLP([2, 3, 2], activations=["relu"])


def test_mlp_forward_output_shape():
    mlp = MLP([3, 8, 8, 5], seed=0)
    x = np.random.default_rng(0).normal(size=(7, 3))
    out = mlp.forward(x)
    assert out.shape == (7, 5)


def test_mlp_predict_returns_valid_class_indices():
    mlp = MLP([2, 4, 3], seed=1)
    x = np.random.default_rng(1).normal(size=(20, 2))
    preds = mlp.predict(x)
    assert preds.shape == (20,)
    assert np.all((preds >= 0) & (preds < 3))


def test_mlp_is_deterministic_given_seed():
    mlp1 = MLP([2, 5, 2], seed=42)
    mlp2 = MLP([2, 5, 2], seed=42)
    for l1, l2 in zip(mlp1.layers, mlp2.layers):
        assert np.array_equal(l1.W, l2.W)
        assert np.array_equal(l1.b, l2.b)


def test_mlp_different_seeds_give_different_weights():
    mlp1 = MLP([2, 5, 2], seed=1)
    mlp2 = MLP([2, 5, 2], seed=2)
    assert not np.array_equal(mlp1.layers[0].W, mlp2.layers[0].W)


def test_mlp_num_params_matches_manual_count():
    mlp = MLP([3, 4, 2], seed=0)
    expected = (3 * 4 + 4) + (4 * 2 + 2)
    assert mlp.num_params() == expected


def test_mlp_get_flat_params_length():
    mlp = MLP([3, 4, 5, 2], seed=0)
    flat = mlp.get_flat_params()
    assert len(flat) == 3 * 2  # W and b per layer, 3 layers


def test_dense_backward_shapes():
    layer = Dense(4, 6, activation="relu", rng=np.random.default_rng(0))
    x = np.random.default_rng(0).normal(size=(10, 4))
    layer.forward(x)
    grad_output = np.random.default_rng(1).normal(size=(10, 6))
    grad_x, grad_W, grad_b = layer.backward(grad_output)
    assert grad_x.shape == (10, 4)
    assert grad_W.shape == (4, 6)
    assert grad_b.shape == (6,)
