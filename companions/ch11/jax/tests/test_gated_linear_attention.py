r"""Tests for ch11 gated_linear_attention: GLA/RetNet and the ch09 round-trip.

Pins the §11.3 claims (Theorem ``ch11:gla-ltv-duality``):

* the gated recurrence equals the masked-parallel form to ``< 1e-12``, for both a
  per-feature (GLA) gate and a scalar (RetNet) gate;
* a constant gate gives a **Toeplitz** decay mask (the LTI face);
* the GLA scalar decay mask coincides *exactly* with Chapter 9's
  ``ch09.segsum``-built decay mask under $\log\gamma_i = a\,\Delta_i$ — the
  cross-chapter round-trip to ``ch09:ssd-duality``.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from companions.ch11.jax.gated_linear_attention import (  # noqa: E402
    ch09_decay_mask,
    gated_masked,
    gated_recurrent,
    gla_scalar_decay_mask,
    retnet_decay_mask,
)


def _qkv(length=32, d=8, d_v=6, seed=0):
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((length, d)))
    k = jnp.asarray(rng.standard_normal((length, d)))
    v = jnp.asarray(rng.standard_normal((length, d_v)))
    return q, k, v, rng


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_gated_recurrent_equals_masked_per_feature(seed):
    """GLA per-feature gate: scan oracle == masked-parallel form."""
    q, k, v, rng = _qkv(seed=seed)
    length, d_k = q.shape
    log_gamma = jnp.log(0.90 + 0.09 * jax.nn.sigmoid(jnp.asarray(rng.standard_normal((length, d_k)))))
    y_rec = gated_recurrent(q, k, v, log_gamma)
    y_mask = gated_masked(q, k, v, log_gamma)
    assert_allclose(np.asarray(y_rec), np.asarray(y_mask), rtol=0, atol=1e-12)


@pytest.mark.parametrize("gamma", [0.85, 0.92, 0.99])
def test_gated_recurrent_equals_masked_scalar(gamma):
    """RetNet constant scalar gate: scan oracle == masked-parallel form."""
    q, k, v, _ = _qkv(seed=2)
    log_gamma = jnp.full((q.shape[0],), jnp.log(gamma))
    y_rec = gated_recurrent(q, k, v, log_gamma)
    y_mask = gated_masked(q, k, v, log_gamma)
    assert_allclose(np.asarray(y_rec), np.asarray(y_mask), rtol=0, atol=1e-12)


def test_retnet_constant_gamma_is_toeplitz():
    """A constant gate gives a Toeplitz mask: L[t,j] depends only on t-j."""
    mask = retnet_decay_mask(0.9, 24)
    # Shifting both indices by one leaves the lower triangle unchanged.
    assert_allclose(np.asarray(mask[1:, 1:]), np.asarray(mask[:-1, :-1]), rtol=0, atol=0.0)


@pytest.mark.parametrize("seed", [0, 3])
def test_gla_matches_ch09_decay_mask(seed):
    """The GLA scalar decay mask == Chapter 9's segsum mask under log gamma = a*delta."""
    rng = np.random.default_rng(seed)
    length = 28
    a = -0.3
    delta = jnp.asarray(0.5 + 0.5 * rng.random(length))  # positive steps
    gla_mask = gla_scalar_decay_mask(a * delta)
    ch09_mask = ch09_decay_mask(a, delta)
    assert_allclose(np.asarray(gla_mask), np.asarray(ch09_mask), rtol=0, atol=1e-12)


def test_shape_validation():
    q, k, v, _ = _qkv(length=8)
    with pytest.raises(ValueError):
        gated_recurrent(q, k, v, jnp.zeros((7,)))  # wrong gate length
    with pytest.raises(ValueError):
        retnet_decay_mask(1.5, 8)  # gamma out of (0, 1]
    with pytest.raises(ValueError):
        gla_scalar_decay_mask(jnp.zeros((4, 4)))  # must be 1D
