r"""Tests for ch14 two_timescale: the slow-regime HMM task and its exact predictors.

Pins the §§14.2 & 14.6 claims:

* the transition-family identity $T(\varepsilon) = M(\lambda^*)$ at
  $\lambda^* = \varepsilon K/(K-1)$, entrywise to ``< 1e-15``;
* the forward filter == brute-force path-enumeration oracle (an
  independent code path) to ``< 1e-12`` on tiny instances;
* window covering the full prefix == full filter (exact); matched decay
  $\lambda = \lambda^*$ == full filter — the fixed-decay state model *is*
  optimal when its decay matches the regime timescale;
* uniform bigrams ⇒ every predictor sits at exactly $\log V$ nats;
* the §14.6 reference numbers quoted in prose/captions (optimality
  ordering with measured gaps; the discrimination diagnostic's monotone
  response to the overlap dial).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from companions.ch14.jax.two_timescale import (  # noqa: E402
    TwoTimescaleHMM,
    decay_filter_predictions,
    enumeration_predictions,
    epsilon_to_lambda,
    forward_filter_predictions,
    make_hmm,
    make_transition,
    mean_cross_entropy,
    mean_pairwise_discrimination,
    mixing_to_uniform_transition,
    regime_stationary_unigrams,
    sample_sequence,
    unigram_filter_predictions,
    window_filter_predictions,
    window_filter_predictions_naive,
)

# The committed reference configuration: every number here is also printed by
# the module main and quoted in §14.6 prose/captions.
_REF_REGIMES = 4
_REF_VOCAB = 12
_REF_CONCENTRATION = 0.3
_REF_OVERLAP = 0.4
_REF_EPS = 0.02
_REF_LENGTH = 8192
_REF_BURN = 128
_REF_SEED = 0


def _reference_instance() -> tuple[TwoTimescaleHMM, jnp.ndarray]:
    key = jax.random.PRNGKey(_REF_SEED)
    key_bigrams, key_seq = jax.random.split(key)
    hmm = make_hmm(
        key_bigrams, _REF_REGIMES, _REF_VOCAB, _REF_EPS, _REF_CONCENTRATION, _REF_OVERLAP
    )
    seq_key = jax.random.fold_in(key_seq, int(_REF_EPS * 1e9) + int(_REF_OVERLAP * 1e3))
    tokens, _ = sample_sequence(seq_key, hmm, _REF_LENGTH)
    return hmm, tokens


def _small_instance(eps=0.1, length=64, seed=3, overlap=1.0):
    key = jax.random.PRNGKey(seed)
    hmm = make_hmm(key, 3, 5, eps, 0.5, overlap)
    tokens, regimes = sample_sequence(jax.random.fold_in(key, 1), hmm, length)
    return hmm, tokens, regimes


# ---------------------------------------------------------------------------
# Model structure.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_regimes,eps", [(2, 0.1), (4, 0.02), (4, 0.5), (8, 0.001)])
def test_transition_family_identity(num_regimes, eps):
    """T(eps) == M(lambda*) entrywise — the decay filter's optimality anchor."""
    lam = epsilon_to_lambda(eps, num_regimes)
    t = make_transition(num_regimes, eps)
    m = mixing_to_uniform_transition(num_regimes, lam)
    assert_allclose(np.asarray(t), np.asarray(m), rtol=0, atol=1e-15)


def test_transitions_are_row_stochastic():
    for build in (lambda: make_transition(4, 0.07), lambda: mixing_to_uniform_transition(5, 0.3)):
        t = np.asarray(build())
        assert_allclose(t.sum(axis=1), np.ones(t.shape[0]), rtol=0, atol=1e-15)
        assert t.min() >= 0.0


def test_make_hmm_shapes_and_positivity():
    hmm = make_hmm(jax.random.PRNGKey(0), 4, 12, 0.02, 0.3, 0.4)
    assert hmm.transition.shape == (4, 4)
    assert hmm.bigrams.shape == (4, 12, 12)
    assert bool(jnp.all(hmm.bigrams > 0.0))
    rows = np.asarray(hmm.bigrams).sum(axis=-1)
    assert_allclose(rows, np.ones_like(rows), rtol=0, atol=1e-12)


def test_sample_sequence_deterministic_and_in_range():
    hmm, tokens, regimes = _small_instance()
    assert tokens.shape == regimes.shape == (64,)
    assert int(tokens.min()) >= 0 and int(tokens.max()) < hmm.vocab
    assert int(regimes.min()) >= 0 and int(regimes.max()) < hmm.num_regimes
    _, tokens2, _ = _small_instance()
    assert bool(jnp.all(tokens == tokens2))


def test_epsilon_zero_freezes_the_regime():
    key = jax.random.PRNGKey(11)
    hmm = make_hmm(key, 3, 5, 0.0, 0.5)
    _, regimes = sample_sequence(jax.random.fold_in(key, 2), hmm, 100)
    assert bool(jnp.all(regimes == regimes[0]))


# ---------------------------------------------------------------------------
# Filter correctness against independent oracles.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_regimes,vocab,eps,seed", [(2, 3, 0.15, 7), (3, 4, 0.3, 9)])
def test_filter_equals_enumeration_oracle(num_regimes, vocab, eps, seed):
    """Forward filter == explicit sum over ALL K^(L+1) regime paths."""
    key = jax.random.PRNGKey(seed)
    hmm = make_hmm(key, num_regimes, vocab, eps, 0.5)
    tokens, _ = sample_sequence(jax.random.fold_in(key, 1), hmm, 7)
    filt, _ = forward_filter_predictions(hmm, tokens)
    enum = enumeration_predictions(hmm, tokens)
    assert_allclose(np.asarray(filt), np.asarray(enum), rtol=0, atol=1e-12)


def test_predictions_are_distributions():
    hmm, tokens, _ = _small_instance()
    for preds in (
        forward_filter_predictions(hmm, tokens)[0],
        decay_filter_predictions(hmm, tokens, 0.2)[0],
        unigram_filter_predictions(hmm, tokens)[0],
        window_filter_predictions(hmm, tokens, 8),
    ):
        p = np.asarray(preds)
        assert_allclose(p.sum(axis=1), np.ones(p.shape[0]), rtol=0, atol=1e-12)
        assert p.min() >= 0.0


@pytest.mark.parametrize("window", [1, 2, 5, 16])
def test_window_vmap_equals_loop_oracle(window):
    hmm, tokens, _ = _small_instance()
    fast = window_filter_predictions(hmm, tokens, window)
    slow = window_filter_predictions_naive(hmm, tokens, window)
    assert_allclose(np.asarray(fast), np.asarray(slow), rtol=0, atol=1e-12)


@pytest.mark.parametrize("window_extra", [0, 7])
def test_window_covering_prefix_equals_full_filter(window_extra):
    """w >= L: the window predictor IS the full filter (uniform start is exact)."""
    hmm, tokens, _ = _small_instance(length=128)
    full, _ = forward_filter_predictions(hmm, tokens)
    win = window_filter_predictions(hmm, tokens, 128 + window_extra)
    assert_allclose(np.asarray(win), np.asarray(full), rtol=0, atol=1e-12)


def test_matched_decay_equals_full_filter():
    """lambda = lambda*(eps): the fixed-decay state model is exactly optimal."""
    hmm, tokens, _ = _small_instance(eps=0.1)
    lam_star = epsilon_to_lambda(0.1, hmm.num_regimes)
    full, _ = forward_filter_predictions(hmm, tokens)
    matched, _ = decay_filter_predictions(hmm, tokens, lam_star)
    assert_allclose(np.asarray(matched), np.asarray(full), rtol=0, atol=1e-12)
    # A clearly mistimed decay is NOT the filter.
    mistimed, _ = decay_filter_predictions(hmm, tokens, 0.9)
    assert float(jnp.max(jnp.abs(mistimed - full))) > 1e-3


def test_uniform_bigrams_put_every_predictor_at_log_v():
    """No fast structure and no regime signal: everything predicts uniform."""
    k, v = 3, 5
    hmm = TwoTimescaleHMM(
        transition=make_transition(k, 0.1),
        bigrams=jnp.full((k, v, v), 1.0 / v),
    )
    rng = np.random.default_rng(0)
    tokens = jnp.asarray(rng.integers(0, v, size=200))
    target = float(np.log(v))
    for preds in (
        forward_filter_predictions(hmm, tokens)[0],
        decay_filter_predictions(hmm, tokens, 0.3)[0],
        unigram_filter_predictions(hmm, tokens)[0],
        window_filter_predictions(hmm, tokens, 4),
    ):
        assert mean_cross_entropy(preds, tokens) == pytest.approx(target, rel=0, abs=1e-12)


def test_frozen_regime_posterior_concentrates():
    """eps = 0 with sharply distinct regimes: the posterior finds the true regime."""
    key = jax.random.PRNGKey(5)
    hmm = make_hmm(key, 3, 8, 0.0, 0.3, 1.0)
    tokens, regimes = sample_sequence(jax.random.fold_in(key, 4), hmm, 200)
    _, posteriors = forward_filter_predictions(hmm, tokens)
    true_regime = int(regimes[0])
    assert float(posteriors[-1, true_regime]) > 0.99


# ---------------------------------------------------------------------------
# Stationary unigrams + discrimination diagnostic.
# ---------------------------------------------------------------------------


def test_stationary_unigrams_are_stationary():
    hmm, _, _ = _small_instance()
    pis = np.asarray(regime_stationary_unigrams(hmm))
    bigrams = np.asarray(hmm.bigrams)
    assert_allclose(pis.sum(axis=1), np.ones(pis.shape[0]), rtol=0, atol=1e-12)
    for j in range(pis.shape[0]):
        assert_allclose(pis[j] @ bigrams[j], pis[j], rtol=0, atol=1e-10)


def test_discrimination_monotone_in_overlap_and_zero_at_identical():
    # Same key path as the module's _figure_instance (split before drawing).
    key_bigrams, _ = jax.random.split(jax.random.PRNGKey(_REF_SEED))
    discs = {}
    for eta in (1.0, 0.4, 0.2):
        hmm = make_hmm(key_bigrams, _REF_REGIMES, _REF_VOCAB, 0.02, _REF_CONCENTRATION, eta)
        discs[eta] = mean_pairwise_discrimination(hmm)
    assert discs[1.0] > discs[0.4] > discs[0.2] > 0.0
    # The figure-legend values (window-crossover caption).
    assert discs[1.0] == pytest.approx(3.0417, rel=0, abs=2e-4)
    assert discs[0.4] == pytest.approx(0.5344, rel=0, abs=2e-4)
    assert discs[0.2] == pytest.approx(0.2103, rel=0, abs=2e-4)
    # Identical tables in every regime: zero discrimination, exactly.
    one_table = jax.random.dirichlet(key_bigrams, jnp.full((6,), 0.5), shape=(6,))
    same = TwoTimescaleHMM(
        transition=make_transition(3, 0.1),
        bigrams=jnp.broadcast_to(one_table, (3, 6, 6)),
    )
    assert mean_pairwise_discrimination(same) == pytest.approx(0.0, rel=0, abs=1e-14)


# ---------------------------------------------------------------------------
# The committed reference numbers (§14.6 prose/captions).
# ---------------------------------------------------------------------------


def test_reference_config_numbers_and_optimality_ordering():
    """Pins the exact numbers the chapter quotes; the filter is the floor."""
    hmm, tokens = _reference_instance()
    full, _ = forward_filter_predictions(hmm, tokens)
    ce_full = mean_cross_entropy(full, tokens, _REF_BURN)
    ce_win8 = mean_cross_entropy(
        window_filter_predictions(hmm, tokens, 8), tokens, _REF_BURN
    )
    mistimed, _ = decay_filter_predictions(hmm, tokens, epsilon_to_lambda(0.2, _REF_REGIMES))
    ce_decay = mean_cross_entropy(mistimed, tokens, _REF_BURN)
    uni, _ = unigram_filter_predictions(hmm, tokens)
    ce_uni = mean_cross_entropy(uni, tokens, _REF_BURN)

    # The quoted values (module main prints the same to 4 decimals).
    assert ce_full == pytest.approx(1.9289, rel=0, abs=5e-4)
    assert ce_win8 - ce_full == pytest.approx(0.0246, rel=0, abs=5e-4)
    assert ce_decay - ce_full == pytest.approx(0.0512, rel=0, abs=5e-4)
    assert ce_uni - ce_full == pytest.approx(0.4965, rel=0, abs=5e-4)
    assert mean_pairwise_discrimination(hmm) == pytest.approx(0.5344, rel=0, abs=2e-4)

    # Optimality ordering with real margins (well above the ~2e-4 MC floor).
    assert ce_full < ce_win8 - 0.01
    assert ce_full < ce_decay - 0.02
    assert ce_full < ce_uni - 0.3


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------


def test_argument_validation():
    hmm, tokens, _ = _small_instance()
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError):
        make_transition(1, 0.1)
    with pytest.raises(ValueError):
        make_transition(3, 1.5)
    with pytest.raises(ValueError):
        mixing_to_uniform_transition(3, -0.1)
    with pytest.raises(ValueError):
        epsilon_to_lambda(0.9, 4)  # lambda* > 1
    with pytest.raises(ValueError):
        make_hmm(key, 3, 1, 0.1)  # vocab too small
    with pytest.raises(ValueError):
        make_hmm(key, 3, 5, 0.1, concentration=0.0)
    with pytest.raises(ValueError):
        make_hmm(key, 3, 5, 0.1, overlap=0.0)  # identical regimes — degenerate
    with pytest.raises(ValueError):
        make_hmm(key, 3, 5, 0.1, overlap=1.0001)
    with pytest.raises(ValueError):
        sample_sequence(key, hmm, 1)
    with pytest.raises(ValueError):
        forward_filter_predictions(hmm, tokens[:1])
    with pytest.raises(ValueError):
        forward_filter_predictions(hmm, tokens.at[0].set(99))  # token out of range
    with pytest.raises(ValueError):
        window_filter_predictions(hmm, tokens, 0)
    with pytest.raises(ValueError):
        enumeration_predictions(hmm, tokens)  # too long for the exponential oracle
    preds, _ = forward_filter_predictions(hmm, tokens)
    with pytest.raises(ValueError):
        mean_cross_entropy(preds, tokens, burn=len(tokens))
    with pytest.raises(ValueError):
        mean_cross_entropy(preds[:-1], tokens)
