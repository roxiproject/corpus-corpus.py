"""Loss functions with hand-derived gradients."""

import numpy as np


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def softmax_cross_entropy(logits, labels):
    """Numerically stable softmax + cross entropy.

    logits: (N, C) raw scores.
    labels: (N,) integer class indices.

    Returns (mean_loss, grad_logits) where grad_logits is dL/d(logits)
    already averaged over the batch, i.e. (softmax(logits) - one_hot) / N.
    """
    n = logits.shape[0]
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    log_probs = shifted - log_sum_exp
    loss = -np.mean(log_probs[np.arange(n), labels])

    probs = np.exp(log_probs)
    grad = probs.copy()
    grad[np.arange(n), labels] -= 1.0
    grad /= n
    return loss, grad
