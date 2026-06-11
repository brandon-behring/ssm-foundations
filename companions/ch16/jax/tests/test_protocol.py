"""Tests for companions/ch16/jax/protocol.py.

Conventions (match ch14): float64; exact identities < 1e-12 with
``assert_allclose(rtol=0)``; the composite's Python-loop oracle is the
independent check; every printed/caption number pinned at its precision.
The reference instance replicates ch14's, so the full-filter CE pin (1.9289)
doubles as a cross-chapter consistency check.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from companions.ch14.jax.two_timescale import (  # noqa: E402
    decay_filter_predictions,
    epsilon_to_lambda,
    forward_filter_predictions,
    make_hmm,
    mean_cross_entropy,
    sample_sequence,
    window_filter_predictions,
)
from companions.ch16.jax.protocol import (  # noqa: E402
    _INFLATION_KS,
    _INFLATION_TRIALS,
    _REF_BURN,
    _REF_EPS,
    _REF_LAM_MIS,
    _REF_REGIMES,
    _REF_SEED,
    _REF_WINDOW,
    best_of_k_seed_inflation,
    composite_filter_predictions,
    composite_filter_predictions_naive,
    emission_min_singular_value,
    filter_regime_priors,
    gaussian_max_inflation,
    max_inflation_bound,
    paired_comparison_stats,
    per_token_log_losses,
    probe_signature,
    recover_prior_from_predictive,
    reference_instance,
    ridge_probe_accuracy,
)


@pytest.fixture(scope="module")
def ref():
    hmm, tokens, regimes = reference_instance()
    return hmm, tokens, regimes


@pytest.fixture(scope="module")
def small():
    # A short instance where the naive oracle is cheap.
    key = jax.random.PRNGKey(23)
    hmm = make_hmm(key, 3, 5, 0.05, 0.4, 0.5)
    tokens, regimes = sample_sequence(jax.random.fold_in(key, 1), hmm, 200)
    return hmm, tokens, regimes


# ---------------------------------------------------------------------------
# Composite restriction: exact identities + oracle.
# ---------------------------------------------------------------------------


def test_composite_uniform_equals_ch14_window(ref) -> None:
    hmm, tokens, _ = ref
    comp, _ = composite_filter_predictions(hmm, tokens, _REF_WINDOW, None)
    win = window_filter_predictions(hmm, tokens, _REF_WINDOW)
    assert_allclose(np.asarray(comp), np.asarray(win), rtol=0, atol=1e-15)


@pytest.mark.parametrize("window", [1, 4, 8, 64])
def test_composite_matched_lambda_is_full_filter(ref, window: int) -> None:
    # The headline identity: a matched carried prior + exact in-window updates
    # reproduce the Bayes filter at EVERY window size.
    hmm, tokens, _ = ref
    lam_star = epsilon_to_lambda(_REF_EPS, _REF_REGIMES)
    comp, _ = composite_filter_predictions(hmm, tokens, window, lam_star)
    full, _ = forward_filter_predictions(hmm, tokens)
    assert_allclose(np.asarray(comp), np.asarray(full), rtol=0, atol=1e-12)


def test_composite_window_covering_sequence_is_full_filter(small) -> None:
    hmm, tokens, _ = small
    comp, _ = composite_filter_predictions(hmm, tokens, int(tokens.shape[0]), 0.9)
    full, _ = forward_filter_predictions(hmm, tokens)
    assert_allclose(np.asarray(comp), np.asarray(full), rtol=0, atol=1e-12)


@pytest.mark.parametrize("window", [1, 3, 16])
@pytest.mark.parametrize("lam", [None, 0.3])
def test_composite_matches_naive_oracle(small, window: int, lam: float | None) -> None:
    hmm, tokens, _ = small
    preds, priors = composite_filter_predictions(hmm, tokens, window, lam)
    preds_n, priors_n = composite_filter_predictions_naive(hmm, tokens, window, lam)
    assert_allclose(np.asarray(preds), np.asarray(preds_n), rtol=0, atol=1e-12)
    assert_allclose(np.asarray(priors), np.asarray(priors_n), rtol=0, atol=1e-12)


def test_composite_validation(small) -> None:
    hmm, tokens, _ = small
    with pytest.raises(ValueError):
        composite_filter_predictions(hmm, tokens, 0, None)
    with pytest.raises(ValueError):
        composite_filter_predictions(hmm, jnp.array([1]), 2, None)
    with pytest.raises(ValueError):
        composite_filter_predictions_naive(hmm, tokens, 0, None)


def test_composite_priors_are_distributions(small) -> None:
    hmm, tokens, _ = small
    _, priors = composite_filter_predictions(hmm, tokens, 4, 0.2)
    assert_allclose(np.asarray(jnp.sum(priors, axis=1)), 1.0, rtol=0, atol=1e-12)
    assert float(jnp.min(priors)) >= 0.0


# ---------------------------------------------------------------------------
# Reference-config cross-entropies (cross-chapter pins) + the ordering.
# ---------------------------------------------------------------------------


def test_reference_cross_entropies_pinned(ref) -> None:
    hmm, tokens, _ = ref
    full, _ = forward_filter_predictions(hmm, tokens)
    win = window_filter_predictions(hmm, tokens, _REF_WINDOW)
    decay, _ = decay_filter_predictions(hmm, tokens, _REF_LAM_MIS)
    comp, _ = composite_filter_predictions(hmm, tokens, _REF_WINDOW, _REF_LAM_MIS)
    ce_full = mean_cross_entropy(full, tokens, _REF_BURN)
    ce_win = mean_cross_entropy(win, tokens, _REF_BURN)
    ce_decay = mean_cross_entropy(decay, tokens, _REF_BURN)
    ce_comp = mean_cross_entropy(comp, tokens, _REF_BURN)
    # ce_full == ch14's published reference number: the instance replication.
    assert_allclose(ce_full, 1.9289, rtol=0, atol=1e-4)
    assert_allclose(ce_win, 1.9535, rtol=0, atol=1e-4)
    assert_allclose(ce_decay, 1.9801, rtol=0, atol=1e-4)
    assert_allclose(ce_comp, 1.9390, rtol=0, atol=1e-4)
    # The measured ordering: the composite beats both pure restrictions.
    assert ce_full < ce_comp < ce_win < ce_decay


# ---------------------------------------------------------------------------
# Comparison statistics (§16.3).
# ---------------------------------------------------------------------------


def test_excess_ce_identity_closed_form(ref) -> None:
    # Proposition (i), numerically: per position, the EXPECTED excess log loss
    # under the true conditional equals KL(P* || Q) exactly; the realized
    # sample excess estimates its mean within Monte-Carlo error.
    hmm, tokens, _ = ref
    full, _ = forward_filter_predictions(hmm, tokens)
    win = window_filter_predictions(hmm, tokens, _REF_WINDOW)
    p = np.asarray(full)[_REF_BURN:]
    q = np.asarray(win)[_REF_BURN:]
    expected_excess = np.sum(p * (-np.log(q)), axis=1) - np.sum(p * (-np.log(p)), axis=1)
    kl = np.sum(p * (np.log(p) - np.log(q)), axis=1)
    assert_allclose(expected_excess, kl, rtol=0, atol=1e-12)
    sample_excess = (mean_cross_entropy(win, tokens, _REF_BURN)
                     - mean_cross_entropy(full, tokens, _REF_BURN))
    diffs = np.asarray(per_token_log_losses(win, tokens, _REF_BURN)
                       - per_token_log_losses(full, tokens, _REF_BURN))
    se = float(np.std(diffs, ddof=1) / np.sqrt(diffs.shape[0]))
    assert abs(sample_excess - float(np.mean(kl))) < 4.0 * se


def test_composite_w16_pin(ref) -> None:
    # Exercise 16.2's measured answer: widening the window shrinks the
    # mistimed composite's excess (0.0101 at w=8 -> 0.0029 at w=16).
    hmm, tokens, _ = ref
    full, _ = forward_filter_predictions(hmm, tokens)
    ce_full = mean_cross_entropy(full, tokens, _REF_BURN)
    c8, _ = composite_filter_predictions(hmm, tokens, 8, _REF_LAM_MIS)
    c16, _ = composite_filter_predictions(hmm, tokens, 16, _REF_LAM_MIS)
    e8 = mean_cross_entropy(c8, tokens, _REF_BURN) - ce_full
    e16 = mean_cross_entropy(c16, tokens, _REF_BURN) - ce_full
    assert_allclose(e16, 0.0029, rtol=0, atol=1e-4)
    assert e16 < e8


def test_per_token_log_losses_mean_matches_ce(small) -> None:
    hmm, tokens, _ = small
    preds, _ = forward_filter_predictions(hmm, tokens)
    losses = per_token_log_losses(preds, tokens, burn=10)
    assert_allclose(float(jnp.mean(losses)), mean_cross_entropy(preds, tokens, 10),
                    rtol=0, atol=1e-14)
    with pytest.raises(ValueError):
        per_token_log_losses(preds, tokens, burn=10_000)


def test_paired_stats_exact_decomposition() -> None:
    # se_paired^2 == se_unpaired^2 - 2 cov / n, exactly (the proposition).
    rng = np.random.default_rng(0)
    shared = rng.standard_normal(50)
    a = shared + 0.1 * rng.standard_normal(50)
    b = shared + 0.1 * rng.standard_normal(50)
    stats = paired_comparison_stats(a, b)
    n = stats["n"]
    cov = float(np.cov(a, b, ddof=1)[0, 1])
    assert_allclose(stats["se_paired"] ** 2,
                    stats["se_unpaired"] ** 2 - 2.0 * cov / n, rtol=0, atol=1e-15)
    assert_allclose(stats["mean_diff"], float(np.mean(a - b)), rtol=0, atol=1e-15)


def test_paired_stats_reference_pins(ref) -> None:
    hmm, tokens, _ = ref
    win = window_filter_predictions(hmm, tokens, _REF_WINDOW)
    comp, _ = composite_filter_predictions(hmm, tokens, _REF_WINDOW, _REF_LAM_MIS)
    stats = paired_comparison_stats(
        per_token_log_losses(win, tokens, _REF_BURN),
        per_token_log_losses(comp, tokens, _REF_BURN),
    )
    assert stats["n"] == 8063.0
    assert_allclose(stats["mean_diff"], 0.0145, rtol=0, atol=1e-4)
    assert_allclose(stats["correlation"], 0.9867, rtol=0, atol=1e-4)
    assert_allclose(stats["se_paired"], 0.00158, rtol=0, atol=1e-5)
    assert_allclose(stats["se_unpaired"], 0.01370, rtol=0, atol=1e-5)
    # The headline: same data, decisive with pairing, insignificant without.
    assert stats["mean_diff"] / stats["se_paired"] > 9.0
    assert stats["mean_diff"] / stats["se_unpaired"] < 1.2


def test_paired_stats_validation() -> None:
    with pytest.raises(ValueError):
        paired_comparison_stats(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError):
        paired_comparison_stats(np.zeros(1), np.zeros(1))


def test_max_inflation_bound_values_and_validation() -> None:
    assert max_inflation_bound(1, 1.0) == 0.0
    assert_allclose(max_inflation_bound(4, 2.0), 2.0 * np.sqrt(2.0 * np.log(4.0)),
                    rtol=0, atol=1e-15)
    with pytest.raises(ValueError):
        max_inflation_bound(0, 1.0)
    with pytest.raises(ValueError):
        max_inflation_bound(4, 0.0)


def test_gaussian_max_inflation_respects_bound() -> None:
    measured = gaussian_max_inflation(
        jax.random.PRNGKey(_REF_SEED + 1), _INFLATION_KS, _INFLATION_TRIALS
    )
    bounds = np.asarray([max_inflation_bound(k, 1.0) for k in _INFLATION_KS])
    mc_tol = 3.0 / np.sqrt(_INFLATION_TRIALS)  # Monte-Carlo noise on E[max]
    assert np.all(measured <= bounds + mc_tol)
    assert np.all(measured[np.asarray(_INFLATION_KS) >= 2] < bounds[np.asarray(_INFLATION_KS) >= 2])
    assert np.all(np.diff(measured) > 0)  # strictly more inflation with more variants
    # Figure pins (printed precision).
    assert_allclose(measured[_INFLATION_KS.index(16)], 1.7547, rtol=0, atol=1e-4)
    assert_allclose(measured[_INFLATION_KS.index(64)], 2.3383, rtol=0, atol=1e-4)
    with pytest.raises(ValueError):
        gaussian_max_inflation(jax.random.PRNGKey(0), (0, 2), 10)
    with pytest.raises(ValueError):
        gaussian_max_inflation(jax.random.PRNGKey(0), (2,), 0)


def test_best_of_k_seed_inflation_pins() -> None:
    sweep = best_of_k_seed_inflation()
    assert_allclose(sweep["mean"], 0.9724, rtol=0, atol=1e-4)
    assert_allclose(sweep["inflation"], 0.0051, rtol=0, atol=1e-4)
    assert_allclose(sweep["spread"], 0.0028, rtol=0, atol=1e-4)
    assert sweep["best"] > sweep["mean"]
    with pytest.raises(ValueError):
        best_of_k_seed_inflation(k=1)


# ---------------------------------------------------------------------------
# Probe machinery (§16.5).
# ---------------------------------------------------------------------------


def test_ridge_probe_perfect_features() -> None:
    labels = jnp.asarray(np.tile(np.arange(4), 50))
    feats = jax.nn.one_hot(labels, 4, dtype=jnp.float64)
    assert ridge_probe_accuracy(feats, labels, 4) == 1.0


def test_ridge_probe_validation() -> None:
    feats = jnp.zeros((10, 3))
    labels = jnp.zeros((10,), dtype=jnp.int64)
    with pytest.raises(ValueError):
        ridge_probe_accuracy(feats, jnp.zeros((9,), dtype=jnp.int64), 4)
    with pytest.raises(ValueError):
        ridge_probe_accuracy(feats[:3], labels[:3], 4)
    with pytest.raises(ValueError):
        ridge_probe_accuracy(feats, labels, 4, alpha=0.0)
    with pytest.raises(ValueError):
        ridge_probe_accuracy(feats, labels - 1, 4)


def test_filter_regime_priors_shapes_and_kinds(small) -> None:
    hmm, tokens, _ = small
    for kind in ("full", "decay", "unigram"):
        priors = filter_regime_priors(hmm, tokens, kind, lam=0.3)
        assert priors.shape == (tokens.shape[0] - 1, hmm.num_regimes)
        assert_allclose(np.asarray(jnp.sum(priors, axis=1)), 1.0, rtol=0, atol=1e-12)
    with pytest.raises(ValueError):
        filter_regime_priors(hmm, tokens, "decay")  # lam missing
    with pytest.raises(ValueError):
        filter_regime_priors(hmm, tokens, "nope")


def test_full_prior_is_filter_internal_prior(small) -> None:
    # The reconstructed prior must be exactly the prior the filter's own
    # prediction used: pred_i == prior_i @ B[:, x_i, :].
    hmm, tokens, _ = small
    preds, _ = forward_filter_predictions(hmm, tokens)
    priors = filter_regime_priors(hmm, tokens, "full")
    toks = np.asarray(tokens)
    rebuilt = np.einsum("ik,ikv->iv", np.asarray(priors),
                        np.asarray(hmm.bigrams)[:, toks[:-1], :].swapaxes(0, 1))
    assert_allclose(rebuilt, np.asarray(preds), rtol=0, atol=1e-12)


def test_probe_signature_pins(ref) -> None:
    hmm, tokens, regimes = ref
    sig = probe_signature(hmm, tokens, regimes)
    assert_allclose(sig["full"], 0.8390, rtol=0, atol=1e-4)
    assert_allclose(sig["composite"], 0.7795, rtol=0, atol=1e-4)
    assert_allclose(sig["decay"], 0.6912, rtol=0, atol=1e-4)
    assert_allclose(sig["window"], 0.7173, rtol=0, atol=1e-4)
    assert_allclose(sig["unigram"], 0.5476, rtol=0, atol=1e-4)
    # The measured signature ordering the prose describes.
    assert sig["full"] > sig["composite"] > sig["window"] > sig["unigram"] > 1.0 / _REF_REGIMES
    assert sig["composite"] > sig["decay"]


def test_probe_signature_validation(small) -> None:
    hmm, tokens, regimes = small
    with pytest.raises(ValueError):
        probe_signature(hmm, tokens, regimes[:-1], burn=10)


def test_posterior_argmax_ceiling_pin(ref) -> None:
    hmm, tokens, regimes = ref
    _, posts = forward_filter_predictions(hmm, tokens)
    ceiling = float(np.mean(
        np.argmax(np.asarray(posts)[_REF_BURN:], axis=1)
        == np.asarray(regimes[1:][_REF_BURN:])
    ))
    assert_allclose(ceiling, 0.8491, rtol=0, atol=1e-4)


# ---------------------------------------------------------------------------
# Probe-recoverability (the §16.5 proposition, numerically).
# ---------------------------------------------------------------------------


def test_emission_min_singular_value_pin(ref) -> None:
    hmm, _, _ = ref
    sigma = emission_min_singular_value(hmm)
    assert sigma > 0.0
    assert_allclose(sigma, 0.0402, rtol=0, atol=1e-4)


def test_prior_recovered_from_predictive(ref) -> None:
    # Invert pred = prior @ B[:, x, :] for two different restrictions: the
    # regime prior is readable off the predictive distribution alone.
    hmm, tokens, _ = ref
    for lam in (None, _REF_LAM_MIS):
        preds, priors = composite_filter_predictions(hmm, tokens, _REF_WINDOW, lam)
        recovered = recover_prior_from_predictive(hmm, preds, tokens)
        assert float(jnp.max(jnp.abs(recovered - priors))) < 1e-10


def test_recover_prior_validation(small) -> None:
    hmm, tokens, _ = small
    with pytest.raises(ValueError):
        recover_prior_from_predictive(hmm, jnp.zeros((3, hmm.vocab)), tokens)
