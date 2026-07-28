import numpy as np
import pytest

from corpus_corpus.activations import gelu_backward, gelu_forward, relu_backward, relu_forward
from tests.gradcheck_utils import numerical_gradient, relative_error


def _scalar_sum_after(fwd, x, weights):
    return np.sum(fwd(x) * weights)


@pytest.mark.parametrize("shape", [(1,), (5,), (3, 4), (7, 2)])
def test_relu_forward_matches_definition(shape, rng):
    x = rng.normal(size=shape)
    out = relu_forward(x)
    assert np.all(out >= 0)
    assert np.allclose(out, np.where(x > 0, x, 0.0))


@pytest.mark.parametrize("shape", [(1,), (5,), (3, 4), (6, 6)])
def test_relu_gradient_matches_numerical(shape, rng):
    x = rng.normal(size=shape)
    # avoid points exactly at 0 where relu is non-differentiable
    x = np.where(np.abs(x) < 1e-3, x + 0.1, x)
    weights = rng.normal(size=shape)

    def f(x_):
        return _scalar_sum_after(relu_forward, x_, weights)

    numeric = numerical_gradient(f, x.copy())
    analytic = relu_backward(x, weights)
    err = relative_error(numeric, analytic)
    assert np.max(err) < 1e-6


@pytest.mark.parametrize("shape", [(1,), (5,), (3, 4), (6, 6)])
def test_gelu_gradient_matches_numerical(shape, rng):
    x = rng.normal(size=shape)
    weights = rng.normal(size=shape)

    def f(x_):
        return _scalar_sum_after(gelu_forward, x_, weights)

    numeric = numerical_gradient(f, x.copy())
    analytic = gelu_backward(x, weights)
    err = relative_error(numeric, analytic)
    assert np.max(err) < 1e-5


def test_gelu_forward_is_close_to_x_for_large_positive_x():
    x = np.array([10.0, 20.0])
    out = gelu_forward(x)
    assert np.allclose(out, x, atol=1e-3)


def test_gelu_forward_is_close_to_zero_for_large_negative_x():
    x = np.array([-10.0, -20.0])
    out = gelu_forward(x)
    assert np.allclose(out, 0.0, atol=1e-3)


def test_relu_zero_at_origin():
    x = np.array([0.0])
    assert relu_forward(x)[0] == 0.0
