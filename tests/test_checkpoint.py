import numpy as np
import pytest

from corpus_corpus.checkpoint import load_checkpoint, load_extra, save_checkpoint
from corpus_corpus.mlp import MLP


def test_checkpoint_round_trip_identical_predictions(tmp_path):
    mlp = MLP([3, 5, 2], seed=0)
    x = np.random.default_rng(0).normal(size=(10, 3))
    preds_before = mlp.predict(x)
    logits_before = mlp.forward(x)

    path = tmp_path / "model.npz"
    save_checkpoint(str(path), mlp)

    mlp2 = MLP([3, 5, 2], seed=99)  # different init on purpose
    load_checkpoint(str(path), mlp2)

    preds_after = mlp2.predict(x)
    logits_after = mlp2.forward(x)

    assert np.array_equal(preds_before, preds_after)
    assert np.allclose(logits_before, logits_after)


def test_checkpoint_weights_are_byte_exact_after_reload(tmp_path):
    mlp = MLP([2, 4, 2], seed=5)
    path = tmp_path / "model.npz"
    save_checkpoint(str(path), mlp)

    mlp2 = MLP([2, 4, 2], seed=0)
    load_checkpoint(str(path), mlp2)

    for layer_a, layer_b in zip(mlp.layers, mlp2.layers):
        assert np.array_equal(layer_a.W, layer_b.W)
        assert np.array_equal(layer_a.b, layer_b.b)


def test_checkpoint_layer_count_mismatch_raises(tmp_path):
    mlp = MLP([2, 4, 2], seed=0)
    path = tmp_path / "model.npz"
    save_checkpoint(str(path), mlp)

    mlp_wrong = MLP([2, 4, 4, 2], seed=0)
    with pytest.raises(ValueError):
        load_checkpoint(str(path), mlp_wrong)


def test_checkpoint_extra_metadata_round_trips(tmp_path):
    mlp = MLP([2, 3, 2], seed=0)
    path = tmp_path / "model.npz"
    save_checkpoint(str(path), mlp, extra={"epoch": 17})
    extra = load_extra(str(path))
    assert extra["epoch"] == 17
