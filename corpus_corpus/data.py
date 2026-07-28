"""Minibatching, per-epoch shuffling, and train/val splitting."""

import numpy as np


def train_val_split(x, y, val_fraction=0.2, seed=0):
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def iterate_minibatches(x, y, batch_size, shuffle=True, seed=None, rng=None):
    """Yield (x_batch, y_batch) pairs covering the full dataset once.

    If `rng` is given it is used (and advanced) so callers can control the
    exact sequence across epochs with a single seeded generator. Otherwise
    a fresh `np.random.default_rng(seed)` is created for this call only.
    """
    n = x.shape[0]
    order = np.arange(n)
    if shuffle:
        gen = rng if rng is not None else np.random.default_rng(seed)
        gen.shuffle(order)
    for start in range(0, n, batch_size):
        batch_idx = order[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]


def num_batches(n_samples, batch_size):
    return (n_samples + batch_size - 1) // batch_size
