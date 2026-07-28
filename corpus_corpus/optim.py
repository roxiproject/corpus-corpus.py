"""Optimizers implemented from scratch: SGD with momentum, and Adam.

Both operate on a list of (param, grad) numpy-array pairs and mutate the
parameter arrays in place. Kept deliberately framework-free.
"""

import numpy as np


class SGD:
    """SGD with classical (heavy-ball) momentum.

    v <- momentum * v - lr * grad
    param <- param + v
    """

    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p) for p in self.params]

    def step(self, grads):
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * g
            p += self.velocity[i]

    def set_lr(self, lr):
        self.lr = lr


class Adam:
    """Adam with correct bias-corrected first/second moment estimates."""

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bias_correction1 = 1.0 - b1**self.t
        bias_correction2 = 1.0 - b2**self.t
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * (g * g)
            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def set_lr(self, lr):
        self.lr = lr


def build_optimizer(name, params, **kwargs):
    name = name.lower()
    if name == "sgd":
        return SGD(params, **kwargs)
    if name == "adam":
        return Adam(params, **kwargs)
    raise ValueError(f"unknown optimizer {name!r}")
