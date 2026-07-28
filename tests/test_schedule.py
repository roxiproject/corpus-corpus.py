import math

import pytest

from corpus_corpus.schedule import cosine_decay, linear_warmup, warmup_then_cosine


def test_linear_warmup_at_step_zero_is_zero():
    assert linear_warmup(0, 10, 1.0) == 0.0


def test_linear_warmup_reaches_base_lr_at_warmup_steps():
    assert linear_warmup(10, 10, 1.0) == 1.0


def test_linear_warmup_midpoint():
    assert math.isclose(linear_warmup(5, 10, 2.0), 1.0)


def test_linear_warmup_zero_steps_returns_base_lr_immediately():
    assert linear_warmup(0, 0, 3.0) == 3.0


def test_linear_warmup_past_warmup_returns_base_lr():
    assert linear_warmup(100, 10, 1.0) == 1.0


def test_cosine_decay_at_step_zero_is_base_lr():
    assert math.isclose(cosine_decay(0, 100, 1.0), 1.0)


def test_cosine_decay_at_final_step_is_min_lr():
    assert math.isclose(cosine_decay(100, 100, 1.0, min_lr=0.0), 0.0, abs_tol=1e-9)


def test_cosine_decay_at_halfway_point():
    # cos(pi/2) = 0, so value = min_lr + (base-min)*0.5
    val = cosine_decay(50, 100, 1.0, min_lr=0.0)
    assert math.isclose(val, 0.5, abs_tol=1e-9)


def test_cosine_decay_respects_min_lr_floor():
    val = cosine_decay(100, 100, 1.0, min_lr=0.1)
    assert math.isclose(val, 0.1, abs_tol=1e-9)


def test_cosine_decay_clamps_beyond_total_steps():
    val_at_total = cosine_decay(100, 100, 1.0)
    val_beyond = cosine_decay(500, 100, 1.0)
    assert math.isclose(val_at_total, val_beyond, abs_tol=1e-9)


def test_cosine_decay_is_monotonically_non_increasing():
    values = [cosine_decay(s, 100, 1.0) for s in range(101)]
    assert all(values[i] >= values[i + 1] - 1e-12 for i in range(len(values) - 1))


def test_warmup_then_cosine_during_warmup_matches_linear_warmup():
    v1 = warmup_then_cosine(3, 10, 100, 1.0)
    v2 = linear_warmup(3, 10, 1.0)
    assert math.isclose(v1, v2)


def test_warmup_then_cosine_after_warmup_is_at_base_lr_immediately():
    val = warmup_then_cosine(10, 10, 110, 1.0)
    assert math.isclose(val, 1.0, abs_tol=1e-9)


def test_warmup_then_cosine_at_final_step_reaches_min_lr():
    val = warmup_then_cosine(110, 10, 110, 1.0, min_lr=0.0)
    assert math.isclose(val, 0.0, abs_tol=1e-6)
