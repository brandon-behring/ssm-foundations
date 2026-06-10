r"""Tests for ch12 delta_rule: the explicit gradient step and its recall semantics.

Pins the §12.1-§12.2 claims:

* ``lax.scan`` == materialised-projector oracle to ``< 1e-12`` (float64) —
  the rank-one and projector forms are the same operator;
* the fixed point $S^\star = vk^\top/\|k\|^2$ is invariant under any $\beta$;
* $\beta = 1$ on a unit key retrieves the just-written value exactly, and
  re-storing a key *replaces* its value (additive accumulation cannot);
* with orthonormal keys, additive and delta-rule storage are both exact —
  it is interference (key overlap) the delta rule erases, and with random
  unit keys its retrieval error is strictly below the additive state's.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from companions.ch12.jax.delta_rule import (  # noqa: E402
    additive_state,
    delta_rule_fixed_point,
    delta_rule_naive,
    delta_rule_recurrent,
    delta_rule_step,
    overwrite_retrieval,
    recall_errors,
)


def _stream(length=48, d_k=8, d_v=6, seed=0):
    """Unit-norm keys + beta < 1: the stable regime (beta * ||k||^2 < 2)."""
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((length, d_k)))
    k_raw = rng.standard_normal((length, d_k))
    k = jnp.asarray(k_raw / np.linalg.norm(k_raw, axis=1, keepdims=True))
    v = jnp.asarray(rng.standard_normal((length, d_v)))
    betas = jnp.asarray(rng.uniform(0.1, 0.9, size=length))
    return q, k, v, betas


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_scan_equals_naive_oracle(seed):
    """Rank-one scan == materialised (I - beta k k^T) Python loop, outputs and state."""
    q, k, v, betas = _stream(seed=seed)
    y_scan, s_scan = delta_rule_recurrent(q, k, v, betas)
    y_naive, s_naive = delta_rule_naive(q, k, v, betas)
    assert_allclose(np.asarray(y_scan), np.asarray(y_naive), rtol=0, atol=1e-12)
    assert_allclose(np.asarray(s_scan), np.asarray(s_naive), rtol=0, atol=1e-12)


@pytest.mark.parametrize("beta", [0.1, 1.0, 1.9, 2.5])
def test_fixed_point_invariant_for_any_beta(beta):
    """S* = v k^T / ||k||^2 zeroes the error term, so every step size leaves it fixed."""
    rng = np.random.default_rng(3)
    key = jnp.asarray(rng.standard_normal(8))
    value = jnp.asarray(rng.standard_normal(6))
    s_star = delta_rule_fixed_point(key, value)
    assert_allclose(
        np.asarray(delta_rule_step(s_star, key, value, beta)),
        np.asarray(s_star),
        rtol=0,
        atol=1e-12,
    )


def test_beta_one_unit_key_exact_write():
    """beta = 1 on a unit key: the just-written association is retrieved exactly."""
    rng = np.random.default_rng(5)
    key = rng.standard_normal(8)
    key = jnp.asarray(key / np.linalg.norm(key))
    value = jnp.asarray(rng.standard_normal(6))
    state = jnp.asarray(rng.standard_normal((6, 8)))  # arbitrary prior state
    after = delta_rule_step(state, key, value, 1.0)
    assert_allclose(np.asarray(after @ key), np.asarray(value), rtol=0, atol=1e-12)


def test_overwrite_replaces_additive_lingers():
    """Re-storing a key: delta retrieves v2 to machine zero; additive is off by |v1|."""
    delta_residual, additive_residual = overwrite_retrieval()
    assert delta_residual < 1e-12, f"delta overwrite should be exact; got {delta_residual:.2e}"
    assert additive_residual > 0.1, (
        f"additive state should retain the stale v1 residue; got {additive_residual:.2e}"
    )


def test_orthonormal_keys_both_exact():
    """No key overlap -> no interference -> both storage rules are exact (<= d_k pairs)."""
    for n_pairs in (8, 32):
        add_err, delta_err = recall_errors(n_pairs, orthonormal=True)
        assert add_err < 1e-12, f"additive should be exact on orthonormal keys; got {add_err:.2e}"
        assert delta_err < 1e-12, f"delta should be exact on orthonormal keys; got {delta_err:.2e}"


@pytest.mark.parametrize("n_pairs", [16, 32, 64])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_keys_delta_below_additive(n_pairs, seed):
    """Random unit keys interfere; the erase term strictly reduces the retrieval error."""
    add_err, delta_err = recall_errors(n_pairs, seed=seed)
    assert delta_err < add_err, (
        f"K={n_pairs}, seed={seed}: delta error {delta_err:.3f} "
        f"should be below additive {add_err:.3f}"
    )


def test_initial_state_seeding_matches_full_run():
    """Splitting a run and carrying the state reproduces the full-sequence outputs."""
    q, k, v, betas = _stream(length=32)
    y_full, s_full = delta_rule_recurrent(q, k, v, betas)
    y_a, s_a = delta_rule_recurrent(q[:16], k[:16], v[:16], betas[:16])
    y_b, s_b = delta_rule_recurrent(q[16:], k[16:], v[16:], betas[16:], initial_state=s_a)
    assert_allclose(np.asarray(jnp.concatenate([y_a, y_b])), np.asarray(y_full), rtol=0, atol=1e-12)
    assert_allclose(np.asarray(s_b), np.asarray(s_full), rtol=0, atol=1e-12)


def test_shape_and_argument_validation():
    q, k, v, betas = _stream(length=8)
    with pytest.raises(ValueError):
        delta_rule_recurrent(q, k[:-1], v, betas)  # mismatched key length
    with pytest.raises(ValueError):
        delta_rule_recurrent(q, k, v, betas[:-1])  # mismatched rates length
    with pytest.raises(ValueError):
        delta_rule_recurrent(q, k, v, betas, initial_state=jnp.zeros((3, 3)))  # wrong state shape
    with pytest.raises(ValueError):
        delta_rule_step(jnp.zeros((6, 8)), jnp.zeros(7), jnp.zeros(6), 0.5)  # d_k mismatch
    with pytest.raises(ValueError):
        delta_rule_fixed_point(jnp.zeros(8), jnp.ones(6))  # zero key
    with pytest.raises(ValueError):
        recall_errors(33, d_k=32, orthonormal=True)  # more orthonormal keys than d_k
    with pytest.raises(ValueError):
        additive_state(jnp.zeros((4, 8)), jnp.zeros((5, 6)))  # mismatched K
