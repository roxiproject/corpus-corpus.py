"""End-to-end: train a real tiny model on a real toy task and check it
hits a real, measured accuracy threshold. No fabricated numbers -- this
test fails if the harness doesn't actually learn.
"""

import numpy as np

from corpus_corpus.toy_data import make_two_spirals, make_xor
from corpus_corpus.train import evaluate, train


def test_xor_end_to_end_reaches_perfect_accuracy():
    x, y = make_xor(n_per_class=64, seed=0)
    result = train(
        x, y,
        layer_sizes=[2, 16, 16, 2],
        activations=["relu", "relu", "linear"],
        optimizer_name="adam",
        lr=0.05,
        batch_size=16,
        epochs=300,
        val_fraction=0.2,
        patience=30,
        warmup_steps=10,
        seed=0,
    )
    val_loss, val_acc = evaluate(result["mlp"], result["x_val"], result["y_val"])
    assert val_acc >= 0.95, f"xor toy task val_acc too low: {val_acc}"


def test_two_spirals_end_to_end_reaches_high_accuracy():
    x, y = make_two_spirals(n_per_class=150, seed=0)
    result = train(
        x, y,
        layer_sizes=[2, 32, 32, 2],
        activations=["relu", "relu", "linear"],
        optimizer_name="adam",
        lr=0.02,
        batch_size=32,
        epochs=350,
        val_fraction=0.2,
        patience=35,
        warmup_steps=15,
        seed=0,
    )
    val_loss, val_acc = evaluate(result["mlp"], result["x_val"], result["y_val"])
    assert val_acc >= 0.9, f"two_spirals toy task val_acc too low: {val_acc}"


def test_sgd_can_also_solve_xor():
    x, y = make_xor(n_per_class=64, seed=1)
    result = train(
        x, y,
        layer_sizes=[2, 16, 16, 2],
        optimizer_name="sgd",
        lr=0.3,
        momentum=0.9,
        batch_size=16,
        epochs=500,
        patience=50,
        seed=1,
    )
    val_loss, val_acc = evaluate(result["mlp"], result["x_val"], result["y_val"])
    assert val_acc >= 0.9, f"sgd xor val_acc too low: {val_acc}"
