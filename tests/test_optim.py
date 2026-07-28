"""Optimizer tests: both SGD-momentum and Adam must converge to the known
closed-form optimum of a convex linear regression problem.
"""

import numpy as np
import pytest

from corpus_corpus.optim import SGD, Adam, build_optimizer


def _make_linear_regression_problem(seed=0, n=200, d=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    true_w = rng.normal(size=(d, 1))
    true_b = rng.normal()
    y = x @ true_w + true_b

    # closed form: augment with bias column, solve normal equations
    x_aug = np.concatenate([x, np.ones((n, 1))], axis=1)
    theta_star, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    return x, y, theta_star


def _mse_grad(w, b, x, y):
    n = x.shape[0]
    pred = x @ w + b
    err = pred - y
    grad_w = (2.0 / n) * (x.T @ err)
    grad_b = (2.0 / n) * np.sum(err)
    return grad_w, grad_b


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sgd_momentum_converges_to_closed_form_optimum(seed):
    x, y, theta_star = _make_linear_regression_problem(seed=seed)
    d = x.shape[1]
    rng = np.random.default_rng(seed + 100)
    w = rng.normal(size=(d, 1)) * 0.1
    b = np.array([0.0])

    opt = SGD([w, b], lr=0.05, momentum=0.9)
    for _ in range(2000):
        grad_w, grad_b = _mse_grad(w, b, x, y)
        opt.step([grad_w, np.array([grad_b])])

    w_star = theta_star[:d, 0:1]
    b_star = theta_star[d, 0]
    assert np.allclose(w, w_star, atol=1e-2)
    assert np.allclose(b[0], b_star, atol=1e-2)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adam_converges_to_closed_form_optimum(seed):
    x, y, theta_star = _make_linear_regression_problem(seed=seed)
    d = x.shape[1]
    rng = np.random.default_rng(seed + 100)
    w = rng.normal(size=(d, 1)) * 0.1
    b = np.array([0.0])

    opt = Adam([w, b], lr=0.05)
    for _ in range(2000):
        grad_w, grad_b = _mse_grad(w, b, x, y)
        opt.step([grad_w, np.array([grad_b])])

    w_star = theta_star[:d, 0:1]
    b_star = theta_star[d, 0]
    assert np.allclose(w, w_star, atol=1e-2)
    assert np.allclose(b[0], b_star, atol=1e-2)


def test_adam_bias_correction_matches_hand_derivation():
    """First-step Adam update should exactly equal the textbook formula
    with bias correction (m_hat = g, v_hat = g^2 at t=1, since m0=v0=0).
    """
    w = np.array([1.0])
    opt = Adam([w], lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
    grad = np.array([2.0])
    opt.step([grad])

    # m1 = (1-b1)*g = 0.1*2 = 0.2 ; m_hat = m1 / (1-b1^1) = 0.1*2/0.1 = 2 = g
    # v1 = (1-b2)*g^2 = 0.001*4 = 0.004 ; v_hat = v1/(1-b2^1) = g^2 = 4
    expected_update = 0.1 * 2.0 / (np.sqrt(4.0) + 1e-8)
    expected_w = 1.0 - expected_update
    assert np.allclose(w[0], expected_w, atol=1e-9)


def test_sgd_momentum_accumulates_velocity():
    w = np.array([0.0])
    opt = SGD([w], lr=1.0, momentum=0.5)
    opt.step([np.array([1.0])])
    assert np.allclose(w[0], -1.0)
    opt.step([np.array([1.0])])
    # v2 = 0.5*(-1.0) - 1.0*1.0 = -1.5 ; w = -1.0 + (-1.5) = -2.5
    assert np.allclose(w[0], -2.5)


def test_build_optimizer_factory():
    w = np.array([0.0])
    sgd = build_optimizer("sgd", [w], lr=0.1, momentum=0.9)
    adam = build_optimizer("adam", [w], lr=0.1)
    assert isinstance(sgd, SGD)
    assert isinstance(adam, Adam)


def test_build_optimizer_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_optimizer("rmsprop", [np.array([0.0])])
