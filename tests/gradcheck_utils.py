"""Shared central-difference numerical gradient checking utility."""

import numpy as np


def numerical_gradient(f, x, eps=1e-5):
    """Central-difference numerical gradient of scalar-valued f at x."""
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = f(x)
        x[idx] = orig - eps
        f_minus = f(x)
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a, b):
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-12)
    return np.abs(a - b) / denom
