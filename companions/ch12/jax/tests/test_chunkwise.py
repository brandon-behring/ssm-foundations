r"""Tests for ch12 chunkwise: cross-chunk state passing and the WY representation.

Pins the §12.5 claims:

* ``chunkwise == recurrent`` (outputs and final state) to ``< 1e-12`` for
  every chunk size dividing $L$, including the degenerate $C = 1$ and $C = L$;
* the WY factors satisfy $I - W^\top Y = \prod_t (I - \beta_t k_t k_t^\top)$
  to ``< 1e-12``, and :func:`apply_wy_to_state` equals right-multiplication
  by the materialised product;
* a single factor ($C = 1$) reduces WY to the plain rank-one projector.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from companions.ch12.jax.chunkwise import (  # noqa: E402
    apply_wy_to_state,
    chunk_wy_representation,
    delta_rule_chunkwise,
    explicit_erase_product,
)
from companions.ch12.jax.delta_rule import delta_rule_recurrent  # noqa: E402


def _stream(length=64, d_k=8, d_v=6, seed=0):
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((length, d_k)))
    k_raw = rng.standard_normal((length, d_k))
    k = jnp.asarray(k_raw / np.linalg.norm(k_raw, axis=1, keepdims=True))
    v = jnp.asarray(rng.standard_normal((length, d_v)))
    betas = jnp.asarray(rng.uniform(0.1, 0.9, size=length))
    return q, k, v, betas


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 8, 16, 32, 64])
def test_chunkwise_equals_recurrent(chunk_size):
    """Cross-chunk state passing reproduces the monolithic scan for every C | L."""
    q, k, v, betas = _stream()
    y_ref, s_ref = delta_rule_recurrent(q, k, v, betas)
    y_c, s_c = delta_rule_chunkwise(q, k, v, betas, chunk_size)
    assert_allclose(np.asarray(y_c), np.asarray(y_ref), rtol=0, atol=1e-12)
    assert_allclose(np.asarray(s_c), np.asarray(s_ref), rtol=0, atol=1e-12)


@pytest.mark.parametrize("chunk_size", [2, 8, 16, 32, 64])
def test_wy_equals_explicit_product(chunk_size):
    """I - W^T Y == the materialised product of rank-one factors."""
    _, k, _, betas = _stream()
    keys_c, betas_c = k[:chunk_size], betas[:chunk_size]
    W, Y = chunk_wy_representation(keys_c, betas_c)
    p_wy = jnp.eye(k.shape[1]) - W.T @ Y
    p_explicit = explicit_erase_product(keys_c, betas_c)
    assert_allclose(np.asarray(p_wy), np.asarray(p_explicit), rtol=0, atol=1e-12)


def test_apply_wy_equals_matrix_product():
    """The two-GEMM application S - (S W^T) Y == S @ P with the materialised P."""
    rng = np.random.default_rng(3)
    _, k, _, betas = _stream(seed=3)
    keys_c, betas_c = k[:16], betas[:16]
    W, Y = chunk_wy_representation(keys_c, betas_c)
    state = jnp.asarray(rng.standard_normal((6, 8)))
    expected = state @ explicit_erase_product(keys_c, betas_c)
    assert_allclose(np.asarray(apply_wy_to_state(state, W, Y)), np.asarray(expected),
                    rtol=0, atol=1e-12)


def test_single_factor_reduces_to_projector():
    """C = 1: WY is exactly the single rank-one factor I - beta k k^T."""
    rng = np.random.default_rng(4)
    key = jnp.asarray(rng.standard_normal(8))[None, :]  # (1, d_k)
    beta = jnp.asarray([0.6])
    W, Y = chunk_wy_representation(key, beta)
    p_wy = jnp.eye(8) - W.T @ Y
    p_direct = jnp.eye(8) - 0.6 * jnp.outer(key[0], key[0])
    assert_allclose(np.asarray(p_wy), np.asarray(p_direct), rtol=0, atol=1e-15)


def test_argument_validation():
    q, k, v, betas = _stream(length=12)
    for bad_chunk in (0, -4, 5):  # 5 does not divide 12
        with pytest.raises(ValueError):
            delta_rule_chunkwise(q, k, v, betas, bad_chunk)
    with pytest.raises(ValueError):
        chunk_wy_representation(k[:4], betas[:3])  # mismatched C
    with pytest.raises(ValueError):
        chunk_wy_representation(k[0], betas[:1])  # keys not 2D
    W, Y = chunk_wy_representation(k[:4], betas[:4])
    with pytest.raises(ValueError):
        apply_wy_to_state(jnp.zeros((6, 9)), W, Y)  # d_k mismatch
