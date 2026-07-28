"""Mandatory gradient checking: analytic backprop through the whole MLP
(dense layers + activations + softmax cross-entropy) verified against
central-difference numerical gradients on the full loss.
"""

import numpy as np
import pytest

from corpus_corpus.mlp import MLP
from tests.gradcheck_utils import numerical_gradient, relative_error

CONFIGS = [
    ([2, 3, 2], ["relu", "linear"]),
    ([2, 4, 4, 2], ["relu", "relu", "linear"]),
    ([3, 5, 3], ["gelu", "linear"]),
    ([4, 6, 6, 3], ["gelu", "gelu", "linear"]),
    ([2, 3, 4, 2], ["relu", "gelu", "linear"]),
    ([5, 8, 2], ["relu", "linear"]),
]


@pytest.mark.parametrize("layer_sizes,activations", CONFIGS)
def test_mlp_param_gradients_match_numerical(layer_sizes, activations):
    rng = np.random.default_rng(0)
    n_samples = 6
    in_dim = layer_sizes[0]
    out_dim = layer_sizes[-1]
    x = rng.normal(size=(n_samples, in_dim))
    labels = rng.integers(0, out_dim, size=n_samples)

    mlp = MLP(layer_sizes, activations=activations, seed=1)

    def loss_only():
        loss, _, _ = mlp.loss_and_grads(x, labels)
        return loss

    _, _, analytic_grads = mlp.loss_and_grads(x, labels)

    for i, layer in enumerate(mlp.layers):
        for name in ("W", "b"):
            arr = getattr(layer, name)

            def f(_arr, layer=layer, name=name):
                # arr is mutated in place by numerical_gradient, and
                # `layer`'s attribute *is* that same array, so no
                # reassignment is needed here.
                loss, _, _ = mlp.loss_and_grads(x, labels)
                return loss

            numeric = numerical_gradient(f, arr, eps=1e-5)
            analytic = analytic_grads[i][name]
            err = relative_error(numeric, analytic)
            assert np.max(err) < 1e-4, (
                f"grad mismatch layer={i} param={name} config={layer_sizes}/{activations} "
                f"max_err={np.max(err)}"
            )


@pytest.mark.parametrize("layer_sizes,activations", CONFIGS)
def test_mlp_input_gradient_matches_numerical(layer_sizes, activations):
    rng = np.random.default_rng(2)
    n_samples = 4
    in_dim = layer_sizes[0]
    out_dim = layer_sizes[-1]
    x = rng.normal(size=(n_samples, in_dim))
    labels = rng.integers(0, out_dim, size=n_samples)
    mlp = MLP(layer_sizes, activations=activations, seed=3)

    def f(x_):
        loss, _, _ = mlp.loss_and_grads(x_, labels)
        return loss

    numeric = numerical_gradient(f, x.copy(), eps=1e-5)

    # analytic input gradient: rerun backward manually to capture grad_x
    logits = mlp.forward(x)
    from corpus_corpus.losses import softmax_cross_entropy

    _, grad_logits = softmax_cross_entropy(logits, labels)
    grad = grad_logits
    for layer in reversed(mlp.layers):
        grad, _, _ = layer.backward(grad)
    analytic = grad

    err = relative_error(numeric, analytic)
    assert np.max(err) < 1e-4


def test_deep_mlp_gradients_are_finite_and_correct():
    """A deeper net (5 hidden layers) to exercise chain-rule composition."""
    layer_sizes = [3, 6, 6, 6, 6, 2]
    activations = ["relu", "gelu", "relu", "gelu", "linear"]
    rng = np.random.default_rng(7)
    x = rng.normal(size=(5, 3)) * 0.5
    labels = rng.integers(0, 2, size=5)
    mlp = MLP(layer_sizes, activations=activations, seed=7)
    loss, logits, grads = mlp.loss_and_grads(x, labels)
    assert np.isfinite(loss)
    for g in grads:
        assert np.all(np.isfinite(g["W"]))
        assert np.all(np.isfinite(g["b"]))

    def f(arr, layer=mlp.layers[2], name="W"):
        loss, _, _ = mlp.loss_and_grads(x, labels)
        return loss

    arr = mlp.layers[2].W
    numeric = numerical_gradient(f, arr, eps=1e-5)
    analytic = grads[2]["W"]
    err = relative_error(numeric, analytic)
    assert np.max(err) < 1e-4
