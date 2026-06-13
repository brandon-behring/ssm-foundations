r"""Tests for Chapter 15 §15.3 — the information-counting bound (P1′).

Pins the bound's content and the claim that the ch16 slot model's recall cliff sits
exactly at the counting threshold $n^\* = d$. The slot model itself is ch16's
(``rtol=0``-pinned there); here we pin the *bridge* — that the abstract budget $d\,b$
with $b = \log_2(\text{vocab})$ predicts the measured cliff load.
"""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from companions.ch15.jax import copying_bound as cb  # noqa: E402
from companions.ch16.jax import mqar  # noqa: E402


def test_state_capacity_bits_exact() -> None:
    assert cb.state_capacity_bits(16, 6.0) == 96.0
    assert cb.state_capacity_bits(1, 1.0) == 1.0
    assert cb.state_capacity_bits(8, 6.0) == 48.0


def test_min_lossless_state_bits_exact() -> None:
    # n * log2(vocab); pinned at rtol=0 against the math identity.
    np.testing.assert_allclose(cb.min_lossless_state_bits(16, 64), 16 * math.log2(64), rtol=0, atol=0)
    np.testing.assert_allclose(cb.min_lossless_state_bits(10, 2), 10.0, rtol=0, atol=0)
    np.testing.assert_allclose(cb.min_lossless_state_bits(5, 256), 40.0, rtol=0, atol=0)


def test_slot_bits_per_pair() -> None:
    assert cb.slot_bits_per_pair(64) == 6.0
    assert cb.slot_bits_per_pair(256) == 8.0


def test_threshold_meets_cliff_load() -> None:
    """With b = log2(vocab) the counting threshold equals d, the slot model's cliff."""
    d, vocab = 16, 64
    b = cb.slot_bits_per_pair(vocab)
    assert cb.max_recallable_length(d, b, vocab) == d
    assert cb.recall_cliff_load(d) == d
    # And the bits identity is exact at n = d: required == budget.
    np.testing.assert_allclose(
        cb.min_lossless_state_bits(d, vocab), cb.state_capacity_bits(d, b), rtol=0, atol=0
    )


@pytest.mark.parametrize("d,vocab", [(16, 64), (8, 256), (32, 2), (5, 50)])
def test_pigeonhole_inequality_brackets_threshold(d: int, vocab: int) -> None:
    """Below the threshold the budget suffices; one past it the requirement exceeds the budget."""
    b = cb.slot_bits_per_pair(vocab)
    nstar = cb.max_recallable_length(d, b, vocab)
    budget = cb.state_capacity_bits(d, b)
    assert cb.min_lossless_state_bits(nstar, vocab) <= budget + 1e-9
    assert cb.min_lossless_state_bits(nstar + 1, vocab) > budget + 1e-12


def test_slot_cliff_sits_at_threshold() -> None:
    """The measured ch16 slot reader is exact up to N=d and degrades past it (the cliff at n*)."""
    d, vocab = 16, 64
    # exact closed form: 1 for N<=d, <1 for N>d
    assert mqar.slot_accuracy_exact(d, d) == 1.0
    assert mqar.slot_accuracy_exact(d - 1, d) == 1.0
    assert mqar.slot_accuracy_exact(d + 1, d) < 1.0
    # measured slot reader == exact at representative loads (reproduces the figure table, rtol=0)
    for n in (8, 16, 24, 32):
        accs = []
        for s in range(4):
            key = jax.random.fold_in(jax.random.PRNGKey(0), 1000 * n + 7919 * s)
            inst = mqar.make_mqar(key, n, 2048, vocab)
            accs.append(mqar.accuracy(mqar.slot_reader(inst, d), inst.answers))
        measured = float(np.mean(accs))
        np.testing.assert_allclose(measured, mqar.slot_accuracy_exact(n, d), rtol=0, atol=0)


def test_max_recallable_length_floor() -> None:
    # 96 bits / log2(50)=5.6439 -> floor(17.01..) = 17
    assert cb.max_recallable_length(16, 6.0, 50) == int(math.floor(96.0 / math.log2(50)))
    # exact division case: 96 / 6 = 16
    assert cb.max_recallable_length(16, 6.0, 64) == 16


def test_validation_raises() -> None:
    with pytest.raises(ValueError):
        cb.state_capacity_bits(0, 6.0)
    with pytest.raises(ValueError):
        cb.state_capacity_bits(8, 0.0)
    with pytest.raises(ValueError):
        cb.min_lossless_state_bits(0, 64)
    with pytest.raises(ValueError):
        cb.min_lossless_state_bits(8, 1)
    with pytest.raises(ValueError):
        cb.recall_cliff_load(0)
    with pytest.raises(ValueError):
        cb.slot_bits_per_pair(1)
