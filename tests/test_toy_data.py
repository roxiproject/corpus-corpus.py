import numpy as np

from corpus_corpus.toy_data import make_two_spirals, make_xor


def test_make_xor_shapes():
    x, y = make_xor(n_per_class=10, seed=0)
    assert x.shape == (40, 2)
    assert y.shape == (40,)


def test_make_xor_labels_are_binary():
    x, y = make_xor(n_per_class=10, seed=0)
    assert set(np.unique(y).tolist()) <= {0, 1}


def test_make_xor_is_deterministic_with_seed():
    x1, y1 = make_xor(seed=5)
    x2, y2 = make_xor(seed=5)
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)


def test_make_xor_class_balance():
    x, y = make_xor(n_per_class=50, seed=0)
    counts = np.bincount(y)
    assert counts[0] == 100
    assert counts[1] == 100


def test_make_two_spirals_shapes():
    x, y = make_two_spirals(n_per_class=30, seed=0)
    assert x.shape == (60, 2)
    assert y.shape == (60,)


def test_make_two_spirals_deterministic():
    x1, y1 = make_two_spirals(seed=3)
    x2, y2 = make_two_spirals(seed=3)
    assert np.array_equal(x1, x2)


def test_make_two_spirals_is_linearly_inseparable_by_a_trivial_rule():
    # sanity: labels aren't just sign(x) or sign(y) (which would make it
    # a trivial linearly separable task rather than a real toy task).
    x, y = make_two_spirals(n_per_class=100, seed=0)
    trivial_pred = (x[:, 0] > 0).astype(int)
    acc = np.mean(trivial_pred == y)
    assert acc < 0.9
