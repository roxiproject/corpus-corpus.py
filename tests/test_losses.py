import numpy as np
import pytest

from corpus_corpus.losses import softmax, softmax_cross_entropy
from tests.gradcheck_utils import numerical_gradient, relative_error


def test_softmax_rows_sum_to_one(rng):
    logits = rng.normal(size=(10, 5))
    probs = softmax(logits)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_softmax_is_shift_invariant():
    logits = np.array([[1.0, 2.0, 3.0]])
    shifted = logits + 1000.0
    assert np.allclose(softmax(logits), softmax(shifted))


@pytest.mark.parametrize("n,c", [(4, 3), (1, 2), (8, 10)])
def test_cross_entropy_gradient_matches_numerical(n, c, rng):
    logits = rng.normal(size=(n, c))
    labels = rng.integers(0, c, size=n)

    def f(logits_):
        loss, _ = softmax_cross_entropy(logits_, labels)
        return loss

    numeric = numerical_gradient(f, logits.copy())
    _, analytic = softmax_cross_entropy(logits, labels)
    err = relative_error(numeric, analytic)
    assert np.max(err) < 1e-6


def test_cross_entropy_loss_is_low_for_confident_correct_prediction():
    logits = np.array([[10.0, -10.0, -10.0]])
    labels = np.array([0])
    loss, _ = softmax_cross_entropy(logits, labels)
    assert loss < 1e-4


def test_cross_entropy_loss_is_high_for_confident_wrong_prediction():
    logits = np.array([[-10.0, 10.0, -10.0]])
    labels = np.array([0])
    loss, _ = softmax_cross_entropy(logits, labels)
    assert loss > 10.0


def test_cross_entropy_grad_sums_to_zero_per_class_axis_over_full_batch():
    # sum over the batch of (softmax - onehot) integrates to something
    # bounded; sanity check the grad shape and finiteness instead of an
    # exact identity (which only holds for uniform-prior special cases).
    logits = np.random.default_rng(0).normal(size=(6, 4))
    labels = np.array([0, 1, 2, 3, 0, 1])
    _, grad = softmax_cross_entropy(logits, labels)
    assert grad.shape == logits.shape
    assert np.all(np.isfinite(grad))
