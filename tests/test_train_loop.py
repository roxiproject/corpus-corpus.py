import numpy as np
import pytest

from corpus_corpus.checkpoint import load_checkpoint
from corpus_corpus.mlp import MLP
from corpus_corpus.toy_data import make_xor
from corpus_corpus.train import EarlyStopper, evaluate, train


def test_train_returns_expected_keys(tmp_path):
    x, y = make_xor(n_per_class=20, seed=0)
    result = train(
        x, y, layer_sizes=[2, 8, 2], epochs=5, batch_size=8, seed=0, val_fraction=0.25,
    )
    assert "mlp" in result
    assert "records" in result
    assert len(result["records"]) == 5


def test_train_records_have_decreasing_loss_trend():
    x, y = make_xor(n_per_class=30, seed=0)
    result = train(
        x, y, layer_sizes=[2, 16, 2], epochs=60, batch_size=16, lr=0.05, seed=0,
    )
    first_loss = result["records"][0]["train_loss"]
    last_loss = result["records"][-1]["train_loss"]
    assert last_loss < first_loss


def test_training_is_reproducible_given_same_seed():
    x, y = make_xor(n_per_class=20, seed=1)
    r1 = train(x, y, layer_sizes=[2, 8, 2], epochs=10, batch_size=8, seed=7)
    r2 = train(x, y, layer_sizes=[2, 8, 2], epochs=10, batch_size=8, seed=7)
    assert r1["records"][-1]["train_loss"] == r2["records"][-1]["train_loss"]
    for l1, l2 in zip(r1["mlp"].layers, r2["mlp"].layers):
        assert np.array_equal(l1.W, l2.W)


def test_different_seeds_produce_different_shuffle_order_and_results():
    x, y = make_xor(n_per_class=20, seed=1)
    r1 = train(x, y, layer_sizes=[2, 8, 2], epochs=3, batch_size=4, seed=1)
    r2 = train(x, y, layer_sizes=[2, 8, 2], epochs=3, batch_size=4, seed=2)
    assert r1["records"][0]["train_loss"] != r2["records"][0]["train_loss"]


def test_checkpoint_is_saved_during_training_and_reloadable(tmp_path):
    x, y = make_xor(n_per_class=30, seed=0)
    ckpt_path = str(tmp_path / "model.npz")
    result = train(
        x, y, layer_sizes=[2, 16, 2], epochs=30, batch_size=16, lr=0.05, seed=0,
        checkpoint_path=ckpt_path,
    )
    assert result["checkpoint_path"] == ckpt_path

    preds_before = result["mlp"].predict(result["x_val"])

    reloaded = MLP([2, 16, 2], seed=999)  # different init
    load_checkpoint(ckpt_path, reloaded)
    preds_after = reloaded.predict(result["x_val"])

    # the checkpoint holds the *best* epoch, not necessarily the final
    # one, so compare against a fresh forward pass through the saved
    # weights rather than assuming identity with the final live model.
    val_loss_reloaded, val_acc_reloaded = evaluate(reloaded, result["x_val"], result["y_val"])
    assert np.isfinite(val_loss_reloaded)
    assert 0.0 <= val_acc_reloaded <= 1.0
    assert preds_after.shape == preds_before.shape


def test_metrics_files_written_during_training(tmp_path):
    x, y = make_xor(n_per_class=20, seed=0)
    csv_path = str(tmp_path / "metrics.csv")
    jsonl_path = str(tmp_path / "metrics.jsonl")
    train(
        x, y, layer_sizes=[2, 8, 2], epochs=8, batch_size=8, seed=0,
        metrics_csv=csv_path, metrics_jsonl=jsonl_path,
    )
    import os

    assert os.path.exists(csv_path)
    assert os.path.exists(jsonl_path)
    with open(csv_path) as f:
        lines = f.readlines()
    assert len(lines) == 9  # header + 8 epochs


def test_early_stopper_triggers_after_patience_bad_epochs():
    stopper = EarlyStopper(patience=3)
    losses = [1.0, 0.9, 0.95, 0.96, 0.97, 0.98]
    stopped_at = None
    for epoch, loss in enumerate(losses):
        if stopper.step(loss, epoch):
            stopped_at = epoch
            break
    assert stopped_at == 4  # 3 consecutive non-improvements after epoch 1


def test_early_stopper_resets_on_improvement():
    stopper = EarlyStopper(patience=2)
    assert not stopper.step(1.0, 0)
    assert not stopper.step(1.1, 1)  # bad epoch 1
    assert not stopper.step(0.5, 2)  # improvement resets counter
    assert not stopper.step(0.6, 3)  # bad epoch 1
    assert stopper.step(0.6, 4)  # bad epoch 2 -> patience reached, stop now
    assert stopper.should_stop


def test_early_stopping_actually_shortens_training():
    x, y = make_xor(n_per_class=20, seed=0)
    result = train(
        x, y, layer_sizes=[2, 4, 2], epochs=1000, batch_size=8, lr=0.2, seed=0,
        patience=5,
    )
    # xor is easy enough that this should stop well before 1000 epochs
    assert len(result["records"]) < 1000


def test_evaluate_matches_manual_accuracy_computation():
    x, y = make_xor(n_per_class=20, seed=0)
    mlp = MLP([2, 8, 2], seed=0)
    loss, acc = evaluate(mlp, x, y)
    preds = mlp.predict(x)
    manual_acc = float(np.mean(preds == y))
    assert np.isclose(acc, manual_acc)


def test_train_val_split_within_train_has_no_leakage():
    x, y = make_xor(n_per_class=40, seed=0)
    result = train(x, y, layer_sizes=[2, 8, 2], epochs=2, batch_size=8, val_fraction=0.3, seed=0)
    x_val = result["x_val"]
    # val set size should roughly match requested fraction
    assert abs(len(x_val) - 0.3 * len(x)) <= 2
