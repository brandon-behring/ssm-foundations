"""Tests for companions/ch16/jax/mqar.py.

Conventions (match ch14): float64 everywhere; exact identities checked with
``assert_allclose(rtol=0, atol=...)`` or integer equality; the oracle is an
independent code path (NumPy scan); every figure/caption number is pinned at
its printed precision.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from companions.ch16.jax.mqar import (  # noqa: E402
    _FIG_D1,
    _FIG_D2,
    _FIG_DECAY_DIM,
    _FIG_DECAY_PAIRS,
    _FIG_LOADS,
    _FIG_RHOS,
    _FIG_SEPARATIONS,
    MQARInstance,
    _decay_curve,
    _instance,
    _mean_reader_accuracy,
    accuracy,
    auc_log2,
    decay_reader,
    decay_reader_naive,
    induction_reader,
    l90,
    make_mqar,
    oracle_recall,
    outer_product_reader,
    slot_accuracy_exact,
    slot_reader,
)

_KEY = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Generator + oracle.
# ---------------------------------------------------------------------------


def test_generator_shapes_and_layout() -> None:
    inst = make_mqar(_KEY, n_pairs=6, n_keys=32, n_values=8, gap=5, n_distractors=4)
    assert inst.tokens.shape == (2 * (6 + 4) + 5 + 6,)
    assert inst.query_positions.shape == (6,)
    assert inst.answers.shape == (6,)
    assert inst.n_pairs == 6 and inst.n_distractors == 4 and inst.n_stored == 10
    toks = np.asarray(inst.tokens)
    # Layout: keys at even slots / values at odd slots of the stored region.
    assert np.all(toks[0:20:2] < 32)
    assert np.all((toks[1:20:2] >= 32) & (toks[1:20:2] < 40))
    # Gap region is all filler; queries are key tokens.
    assert np.all(toks[20:25] == inst.filler_id)
    assert np.all(toks[25:] < 32)
    # Answers are value tokens.
    assert np.all((np.asarray(inst.answers) >= 32) & (np.asarray(inst.answers) < 40))


def test_generator_determinism() -> None:
    a = make_mqar(_KEY, 8, 64, 16, gap=3, n_distractors=5)
    b = make_mqar(_KEY, 8, 64, 16, gap=3, n_distractors=5)
    assert bool(jnp.all(a.tokens == b.tokens))
    assert bool(jnp.all(a.answers == b.answers))


def test_generator_distinct_keys() -> None:
    inst = make_mqar(_KEY, 20, 64, 8, n_distractors=30)
    stored_keys = np.asarray(inst.tokens)[0 : 2 * inst.n_stored : 2]
    assert np.unique(stored_keys).shape[0] == inst.n_stored


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_pairs": 0, "n_keys": 8, "n_values": 4},
        {"n_pairs": 4, "n_keys": 8, "n_values": 1},
        {"n_pairs": 9, "n_keys": 8, "n_values": 4},
        {"n_pairs": 4, "n_keys": 8, "n_values": 4, "n_distractors": 5},
        {"n_pairs": 4, "n_keys": 8, "n_values": 4, "n_distractors": -1},
        {"n_pairs": 4, "n_keys": 8, "n_values": 4, "gap": -1},
    ],
)
def test_generator_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        make_mqar(_KEY, **kwargs)


@pytest.mark.parametrize("gap,n_distractors", [(0, 0), (7, 0), (0, 12), (9, 12)])
def test_answers_match_oracle(gap: int, n_distractors: int) -> None:
    # The oracle is an independent NumPy scan over raw tokens; it never reads
    # ``answers``, so equality validates the generator's bookkeeping.
    inst = make_mqar(jax.random.PRNGKey(3), 16, 128, 16, gap=gap, n_distractors=n_distractors)
    assert bool(jnp.all(oracle_recall(inst) == inst.answers))


# ---------------------------------------------------------------------------
# Readers: exactness and restrictions.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gap,n_distractors", [(0, 0), (5, 0), (0, 20), (5, 20)])
def test_induction_reader_equals_oracle(gap: int, n_distractors: int) -> None:
    inst = make_mqar(jax.random.PRNGKey(5), 12, 256, 32, gap=gap, n_distractors=n_distractors)
    assert bool(jnp.all(induction_reader(inst) == oracle_recall(inst)))


def test_induction_reader_validation() -> None:
    inst = make_mqar(_KEY, 4, 16, 4)
    with pytest.raises(ValueError):
        induction_reader(inst, beta=0.0)


def test_orthonormal_outer_product_exact() -> None:
    # Zero interference once the key alphabet embeds orthonormally.
    inst = make_mqar(jax.random.PRNGKey(11), 24, 48, 16)
    preds = outer_product_reader(inst, dim=48, orthonormal=True)
    assert bool(jnp.all(preds == oracle_recall(inst)))


def test_orthonormal_requires_fit() -> None:
    inst = make_mqar(_KEY, 4, 64, 8)
    with pytest.raises(ValueError):
        outer_product_reader(inst, dim=32, orthonormal=True)
    with pytest.raises(ValueError):
        outer_product_reader(inst, dim=0)


def test_outer_product_degrades_past_capacity() -> None:
    small = _mean_reader_accuracy(lambda i: outer_product_reader(i, 16), (8,))[0]
    large = _mean_reader_accuracy(lambda i: outer_product_reader(i, 16), (256,))[0]
    assert small == 1.0
    assert large < 0.5


@pytest.mark.parametrize("n_pairs", [4, 16, 24, 64, 256, 512])
@pytest.mark.parametrize("n_slots", [16, 64])
def test_slot_reader_exact_identity(n_pairs: int, n_slots: int) -> None:
    # The proposition's object: accuracy == min(1, d/N) exactly, rtol=0.
    inst = _instance(n_pairs)
    measured = accuracy(slot_reader(inst, n_slots), inst.answers)
    assert_allclose(measured, slot_accuracy_exact(n_pairs, n_slots), rtol=0, atol=0)


def test_slot_reader_abstains_with_filler() -> None:
    # Evicted keys return the filler id — never a valid value, so accidental
    # correctness is impossible (the hypothesis the closed form needs).
    inst = _instance(64)
    preds = np.asarray(slot_reader(inst, 16))
    n_abstain = int(np.sum(preds == inst.filler_id))
    assert n_abstain == 64 - 16
    with pytest.raises(ValueError):
        slot_reader(inst, 0)


def test_slot_accuracy_exact_validation() -> None:
    with pytest.raises(ValueError):
        slot_accuracy_exact(0, 4)
    with pytest.raises(ValueError):
        slot_accuracy_exact(4, 0)


@pytest.mark.parametrize("rho", [0.9, 0.99, 1.0])
@pytest.mark.parametrize("gap,n_distractors", [(0, 0), (17, 24)])
def test_decay_reader_matches_naive(rho: float, gap: int, n_distractors: int) -> None:
    inst = make_mqar(jax.random.PRNGKey(13), 8, 256, 16, gap=gap, n_distractors=n_distractors)
    fast = decay_reader(inst, 32, rho)
    slow = decay_reader_naive(inst, 32, rho)
    assert bool(jnp.all(fast == slow))


def test_decay_rho_one_equals_outer_product() -> None:
    inst = make_mqar(jax.random.PRNGKey(17), 12, 128, 16, n_distractors=8)
    assert bool(jnp.all(decay_reader(inst, 32, 1.0) == outer_product_reader(inst, 32)))


def test_neutral_gap_negative_control() -> None:
    # Same draws, gap 0 vs 512: every stored weight rescales by rho^512 and the
    # argmax read-out is scale-invariant -> predictions exactly equal.
    key = jax.random.PRNGKey(19)
    base = make_mqar(key, 8, 64, 16, gap=0)
    padded = make_mqar(key, 8, 64, 16, gap=512)
    assert bool(jnp.all(base.answers == padded.answers))
    for rho in (0.97, 1.0):
        a = decay_reader(base, 32, rho)
        b = decay_reader(padded, 32, rho)
        assert bool(jnp.all(a == b))


def test_distractors_degrade_decay_reader() -> None:
    seps, accs = _decay_curve(0.99)
    assert accs[0] == 1.0
    assert accs[np.searchsorted(seps, 128)] < 0.1


def test_decay_reader_validation() -> None:
    inst = make_mqar(_KEY, 4, 16, 4)
    with pytest.raises(ValueError):
        decay_reader(inst, 8, 0.0)
    with pytest.raises(ValueError):
        decay_reader(inst, 8, 1.5)
    with pytest.raises(ValueError):
        decay_reader_naive(inst, 0, 0.5)


def test_accuracy_validation() -> None:
    with pytest.raises(ValueError):
        accuracy(jnp.zeros(3), jnp.zeros(4))


# ---------------------------------------------------------------------------
# Length-robustness metrics: exact on hand-built curves.
# ---------------------------------------------------------------------------


def test_l90_exact_hand_curve() -> None:
    seps = np.array([1.0, 2.0, 4.0, 8.0])
    accs = np.array([1.0, 0.95, 0.85, 0.5])
    assert l90(seps, accs) == 2.0  # 0.95 >= 0.9, 0.85 < 0.9
    assert l90(seps, accs, threshold=0.8) == 4.0
    assert l90(seps, accs, threshold=0.5) == 8.0
    # The shortest separation always qualifies.
    assert l90(seps, np.array([0.5, 0.1, 0.0, 0.0])) == 1.0


def test_auc_log2_exact_hand_curves() -> None:
    seps = np.array([1.0, 2.0, 4.0, 8.0])
    assert_allclose(auc_log2(seps, np.full(4, 0.8)), 0.8, rtol=0, atol=1e-15)
    # Accuracy linear in log2(s) from 1 to 0 -> trapezoid mean is 1/2.
    assert_allclose(auc_log2(seps, np.array([1.0, 2 / 3, 1 / 3, 0.0])), 0.5, rtol=0, atol=1e-15)


def test_metric_validation() -> None:
    good_s = np.array([1.0, 2.0, 4.0])
    good_a = np.array([1.0, 0.9, 0.8])
    with pytest.raises(ValueError):
        l90(np.array([2.0, 1.0, 4.0]), good_a)  # not increasing
    with pytest.raises(ValueError):
        l90(np.array([0.0, 1.0, 2.0]), good_a)  # non-positive separation
    with pytest.raises(ValueError):
        l90(good_s, np.array([1.0, 0.9]))  # shape mismatch
    with pytest.raises(ValueError):
        l90(good_s, np.array([1.0, 0.9, 1.2]))  # accuracy > 1
    with pytest.raises(ValueError):
        l90(good_s, good_a, threshold=0.0)
    with pytest.raises(ValueError):
        auc_log2(np.array([1.0]), np.array([1.0]))  # too few points


# ---------------------------------------------------------------------------
# Figure/caption pins (printed precision; deterministic seeds).
# ---------------------------------------------------------------------------


def test_fig_discriminative_gap_pins() -> None:
    loads = np.asarray(_FIG_LOADS)
    gap_hard = np.asarray(
        [slot_accuracy_exact(int(n), _FIG_D2) - slot_accuracy_exact(int(n), _FIG_D1)
         for n in loads]
    )
    # Exactly zero through N = d1; maximum exactly 1 - d1/d2 at N = d2; exact
    # (d2 - d1)/N tail.
    assert_allclose(gap_hard[loads <= _FIG_D1], 0.0, rtol=0, atol=0)
    assert gap_hard.max() == 1.0 - _FIG_D1 / _FIG_D2 == 0.75
    assert int(loads[gap_hard.argmax()]) == _FIG_D2
    assert gap_hard[-1] == (_FIG_D2 - _FIG_D1) / loads[-1] == 0.09375


def test_fig_outer_product_pins() -> None:
    outer_small = _mean_reader_accuracy(lambda i: outer_product_reader(i, _FIG_D1), (64,))[0]
    outer_big = _mean_reader_accuracy(lambda i: outer_product_reader(i, _FIG_D2), (512,))[0]
    assert_allclose(outer_small, 0.8477, rtol=0, atol=1e-4)
    assert_allclose(outer_big, 0.6394, rtol=0, atol=1e-4)


def test_fig_soft_gap_peak_pin() -> None:
    # The caption's soft-knee claim: the outer-product gap peaks at N = 256,
    # right of the slot readers' exact knee at N = d2 = 64.
    loads = np.asarray(_FIG_LOADS)
    small = np.asarray(_mean_reader_accuracy(lambda i: outer_product_reader(i, _FIG_D1),
                                             tuple(_FIG_LOADS)))
    big = np.asarray(_mean_reader_accuracy(lambda i: outer_product_reader(i, _FIG_D2),
                                           tuple(_FIG_LOADS)))
    gap = big - small
    assert int(loads[gap.argmax()]) == 256
    assert_allclose(gap.max(), 0.5688, rtol=0, atol=1e-4)


def test_fig_length_robustness_pins() -> None:
    expected = {
        0.97: (16.0, 0.4727),
        0.99: (32.0, 0.6234),
        0.999: (128.0, 0.8305),
        1.0: (256.0, 0.9227),
    }
    assert set(expected) == set(_FIG_RHOS)
    for rho, (pin_l90, pin_auc) in expected.items():
        seps, accs = _decay_curve(rho)
        assert l90(seps, accs) == pin_l90
        assert_allclose(auc_log2(seps, accs), pin_auc, rtol=0, atol=1e-4)


def test_fig_induction_baseline_pin() -> None:
    accs = []
    for s in range(8):
        inst = _instance(_FIG_DECAY_PAIRS, n_distractors=_FIG_SEPARATIONS[-1],
                         seed_offset=7919 * s)
        accs.append(accuracy(induction_reader(inst), inst.answers))
    assert float(np.mean(accs)) == 1.0


def test_decay_dim_constant_consistency() -> None:
    # The figure claims interference in dim 64; keep the constant honest.
    assert _FIG_DECAY_DIM == 64
    inst = _instance(_FIG_DECAY_PAIRS, n_distractors=4)
    assert isinstance(inst, MQARInstance)
