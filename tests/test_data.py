import numpy as np
import pytest

from corpus_corpus.data import iterate_minibatches, num_batches, train_val_split


def test_train_val_split_sizes():
    x = np.arange(100).reshape(100, 1)
    y = np.arange(100)
    x_tr, y_tr, x_val, y_val = train_val_split(x, y, val_fraction=0.2, seed=0)
    assert len(x_tr) == 80
    assert len(x_val) == 20
    assert len(y_tr) == 80
    assert len(y_val) == 20


def test_train_val_split_no_overlap():
    x = np.arange(50).reshape(50, 1)
    y = np.arange(50)
    x_tr, y_tr, x_val, y_val = train_val_split(x, y, val_fraction=0.3, seed=1)
    train_set = set(x_tr[:, 0].tolist())
    val_set = set(x_val[:, 0].tolist())
    assert train_set.isdisjoint(val_set)
    assert train_set | val_set == set(range(50))


def test_train_val_split_deterministic_with_seed():
    x = np.arange(50).reshape(50, 1)
    y = np.arange(50)
    a = train_val_split(x, y, seed=42)
    b = train_val_split(x, y, seed=42)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[2], b[2])


def test_train_val_split_different_seeds_differ():
    x = np.arange(50).reshape(50, 1)
    y = np.arange(50)
    a = train_val_split(x, y, seed=1)
    b = train_val_split(x, y, seed=2)
    assert not np.array_equal(a[0], b[0])


def test_num_batches():
    assert num_batches(10, 3) == 4
    assert num_batches(9, 3) == 3
    assert num_batches(1, 32) == 1
    assert num_batches(0, 3) == 0


def test_iterate_minibatches_covers_full_dataset():
    x = np.arange(23).reshape(23, 1)
    y = np.arange(23)
    seen = []
    for xb, yb in iterate_minibatches(x, y, batch_size=5, shuffle=False):
        seen.extend(yb.tolist())
    assert sorted(seen) == list(range(23))


def test_iterate_minibatches_batch_sizes():
    x = np.arange(23).reshape(23, 1)
    y = np.arange(23)
    sizes = [len(yb) for _, yb in iterate_minibatches(x, y, batch_size=5, shuffle=False)]
    assert sizes == [5, 5, 5, 5, 3]


def test_iterate_minibatches_shuffling_is_deterministic_with_seeded_rng():
    x = np.arange(20).reshape(20, 1)
    y = np.arange(20)
    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)
    order_a = [yb.tolist() for _, yb in iterate_minibatches(x, y, 4, shuffle=True, rng=rng_a)]
    order_b = [yb.tolist() for _, yb in iterate_minibatches(x, y, 4, shuffle=True, rng=rng_b)]
    assert order_a == order_b


def test_iterate_minibatches_shuffling_changes_order_vs_unshuffled():
    x = np.arange(50).reshape(50, 1)
    y = np.arange(50)
    unshuffled = [yb.tolist() for _, yb in iterate_minibatches(x, y, 5, shuffle=False)]
    shuffled = [yb.tolist() for _, yb in iterate_minibatches(x, y, 5, shuffle=True, seed=5)]
    assert unshuffled != shuffled


def test_iterate_minibatches_epochs_reshuffle_with_advancing_rng():
    x = np.arange(30).reshape(30, 1)
    y = np.arange(30)
    rng = np.random.default_rng(3)
    epoch1 = [yb.tolist() for _, yb in iterate_minibatches(x, y, 6, shuffle=True, rng=rng)]
    epoch2 = [yb.tolist() for _, yb in iterate_minibatches(x, y, 6, shuffle=True, rng=rng)]
    assert epoch1 != epoch2
