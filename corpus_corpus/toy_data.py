"""Toy classification tasks used to exercise the training harness end to end.

These are small, honestly-labeled toy tasks (XOR, and a 2D two-spirals-ish
task built from real trigonometric geometry, not random labels) used only
to prove the harness trains a real model to a real, measured accuracy.
"""

import numpy as np


def make_xor(n_per_class=64, noise=0.1, seed=0):
    """The classic non-linearly-separable XOR problem, with a little
    Gaussian jitter around the four corners so there's more than 4 points.
    """
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    labels_per_center = [0, 1, 1, 0]  # XOR truth table

    xs, ys = [], []
    for center, label in zip(centers, labels_per_center):
        pts = center + rng.normal(0.0, noise, size=(n_per_class, 2))
        xs.append(pts)
        ys.append(np.full(n_per_class, label, dtype=np.int64))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)

    idx = rng.permutation(len(y))
    return x[idx], y[idx]


def make_two_spirals(n_per_class=200, n_turns=1.5, noise=0.05, seed=0):
    """A two-spirals toy classification task: geometrically generated,
    non-linearly-separable, real coordinates (not random labels).
    """
    rng = np.random.default_rng(seed)

    def spiral(n, delta_angle):
        t = np.linspace(0.1, 1.0, n)
        angle = t * n_turns * 2 * np.pi + delta_angle
        radius = t
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        pts = np.stack([x, y], axis=1)
        pts += rng.normal(0.0, noise, size=pts.shape)
        return pts

    class0 = spiral(n_per_class, 0.0)
    class1 = spiral(n_per_class, np.pi)
    x = np.concatenate([class0, class1], axis=0)
    y = np.concatenate(
        [np.zeros(n_per_class, dtype=np.int64), np.ones(n_per_class, dtype=np.int64)]
    )
    idx = rng.permutation(len(y))
    return x[idx], y[idx]
