r"""Tests for ch12 gated_delta: the gated delta rule and its two exact reductions.

Pins the §12.6 claims:

* ``lax.scan`` == materialised-transition oracle to ``< 1e-12`` (float64);
* $\gamma \equiv 1$ recovers plain DeltaNet exactly (erase-only);
* $\beta \equiv 0$ is pure exponential decay: one step multiplies the state
  by exactly $\gamma$ (the misplaced-gate bug guard);
* uniform-vs-selective forgetting: an association orthogonal to all later
  writes decays through the gate by exactly $\gamma^T$.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from companions.ch12.jax.delta_rule import delta_rule_recurrent  # noqa: E402
from companions.ch12.jax.gated_delta import (  # noqa: E402
    gated_delta_naive,
    gated_delta_recurrent,
    gated_delta_step,
    stale_retrieval_after_orthogonal_writes,
)


def _stream(length=48, d_k=8, d_v=6, seed=0):
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((length, d_k)))
    k_raw = rng.standard_normal((length, d_k))
    k = jnp.asarray(k_raw / np.linalg.norm(k_raw, axis=1, keepdims=True))
    v = jnp.asarray(rng.standard_normal((length, d_v)))
    betas = jnp.asarray(rng.uniform(0.1, 0.9, size=length))
    gammas = jnp.asarray(rng.uniform(0.7, 1.0, size=length))
    return q, k, v, betas, gammas


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_scan_equals_naive_oracle(seed):
    q, k, v, betas, gammas = _stream(seed=seed)
    y_scan, s_scan = gated_delta_recurrent(q, k, v, betas, gammas)
    y_naive, s_naive = gated_delta_naive(q, k, v, betas, gammas)
    assert_allclose(np.asarray(y_scan), np.asarray(y_naive), rtol=0, atol=1e-12)
    assert_allclose(np.asarray(s_scan), np.asarray(s_naive), rtol=0, atol=1e-12)


def test_gamma_one_reduces_to_deltanet():
    """gamma == 1: the gate is open, the update IS the plain delta rule."""
    q, k, v, betas, _ = _stream(seed=2)
    ones = jnp.ones(q.shape[0])
    y_gated, s_gated = gated_delta_recurrent(q, k, v, betas, ones)
    y_plain, s_plain = delta_rule_recurrent(q, k, v, betas)
    assert_allclose(np.asarray(y_gated), np.asarray(y_plain), rtol=0, atol=1e-12)
    assert_allclose(np.asarray(s_gated), np.asarray(s_plain), rtol=0, atol=1e-12)


def test_beta_zero_is_pure_decay():
    """beta == 0: one step multiplies the state by exactly gamma — and ONLY gamma.

    This is the guard for the natural porting bug (gating the fresh write too):
    with beta = 0 there is no write, so any deviation from S_new = gamma * S
    means the gate landed on the wrong term.
    """
    rng = np.random.default_rng(5)
    state = jnp.asarray(rng.standard_normal((6, 8)))
    key = jnp.asarray(rng.standard_normal(8))
    value = jnp.asarray(rng.standard_normal(6))
    for gamma in (0.5, 0.9, 1.0):
        stepped = gated_delta_step(state, key, value, 0.0, gamma)
        assert_allclose(np.asarray(stepped), gamma * np.asarray(state), rtol=0, atol=1e-15)


def test_orthogonal_stale_association_decays_by_gamma_power():
    """Writes orthogonal to k_A never erase it; the gate decays it by exactly gamma^T."""
    for gamma, n_writes in ((0.95, 20), (0.9, 40), (1.0, 25)):
        measured, analytic = stale_retrieval_after_orthogonal_writes(n_writes, gamma)
        assert measured == pytest.approx(analytic, rel=0, abs=1e-12)
    # gamma = 1 means no forgetting at all: the association survives unchanged.
    measured, analytic = stale_retrieval_after_orthogonal_writes(25, 1.0)
    assert measured == pytest.approx(analytic, rel=0, abs=1e-12)
    assert measured > 0.1  # the stored |v_A| is O(1), not decayed


def test_argument_validation():
    q, k, v, betas, gammas = _stream(length=8)
    with pytest.raises(ValueError):
        gated_delta_recurrent(q, k, v, betas[:-1], gammas)  # mismatched betas
    with pytest.raises(ValueError):
        gated_delta_recurrent(q, k[:-1], v, betas, gammas)  # mismatched keys
    with pytest.raises(ValueError):
        gated_delta_step(jnp.zeros((6, 7)), jnp.zeros(8), jnp.zeros(6), 0.5, 0.9)  # shape
    with pytest.raises(ValueError):
        stale_retrieval_after_orthogonal_writes(5, 0.0)  # gamma out of range
    with pytest.raises(ValueError):
        stale_retrieval_after_orthogonal_writes(5, 1.2)  # gamma out of range
