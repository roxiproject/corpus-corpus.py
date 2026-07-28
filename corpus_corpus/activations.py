"""Activation functions with hand-derived forward and backward passes.

Each activation exposes a `forward(x)` and a `backward(x, grad_output)`
function. The backward pass takes the *pre-activation input* `x` (not the
output) so it can recompute whatever it needs of the local derivative,
and returns dL/dx given dL/d(output).
"""

import numpy as np


def relu_forward(x):
    return np.maximum(0.0, x)


def relu_backward(x, grad_output):
    mask = (x > 0.0).astype(x.dtype)
    return grad_output * mask


_GELU_C = np.sqrt(2.0 / np.pi)


def gelu_forward(x):
    """Tanh-approximation GELU (same form used in GPT-2 etc.)."""
    inner = _GELU_C * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def gelu_backward(x, grad_output):
    """Analytic derivative of the tanh-approximation GELU."""
    inner = _GELU_C * (x + 0.044715 * x**3)
    t = np.tanh(inner)
    d_inner = _GELU_C * (1.0 + 3.0 * 0.044715 * x**2)
    sech2 = 1.0 - t**2
    local_grad = 0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner
    return grad_output * local_grad


ACTIVATIONS = {
    "relu": (relu_forward, relu_backward),
    "gelu": (gelu_forward, gelu_backward),
    "linear": (lambda x: x, lambda x, g: g),
}
