r"""Tests for ch11 linear_attention: the matrix-state recurrence and its two faces.

Pins the §11.2 / §11.6 claims:

* recurrent == parallel to ``< 1e-12`` (float64), for the normalized elu path and
  the unnormalized elu/relu paths (Theorem ``ch11:recurrent-parallel-equivalence``);
* the matrix state $S = \sum_i \phi(k_i)v_i^\top$ has rank exactly $\min(K, d_k)$
  (Proposition ``ch11:linattn-capacity``, the §11.6 capacity mechanism);
* the normalized output's recurrent/parallel agreement is ``< 1e-12`` in float64
  but only ``~1e-7`` in float32 — the precision point that forces float64.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from companions.ch11.jax.linear_attention import (  # noqa: E402
    linear_attention_parallel,
    linear_attention_recurrent,
    linear_attention_state,
    recurrent_parallel_residual,
)


def _qkv(length=48, d=8, d_v=6, seed=0):
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((length, d)))
    k = jnp.asarray(rng.standard_normal((length, d)))
    v = jnp.asarray(rng.standard_normal((length, d_v)))
    return q, k, v


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_recurrent_equals_parallel_normalized(seed):
    """Normalized elu linear attention: scan oracle == masked matmul."""
    q, k, v = _qkv(seed=seed)
    y_rec = linear_attention_recurrent(q, k, v, feature_map="elu")
    y_par = linear_attention_parallel(q, k, v, feature_map="elu")
    assert_allclose(np.asarray(y_rec), np.asarray(y_par), rtol=0, atol=1e-12)


@pytest.mark.parametrize("fm", ["elu", "relu"])
def test_recurrent_equals_parallel_unnormalized(fm):
    """The core identity holds for any phi in unnormalized mode (no division)."""
    q, k, v = _qkv(seed=3)
    y_rec = linear_attention_recurrent(q, k, v, feature_map=fm, normalize=False)
    y_par = linear_attention_parallel(q, k, v, feature_map=fm, normalize=False)
    assert_allclose(np.asarray(y_rec), np.asarray(y_par), rtol=0, atol=1e-12)


@pytest.mark.parametrize("n_kv,expected", [(4, 4), (16, 16), (48, 32)])
def test_linattn_state_rank(n_kv, expected):
    """rank S = min(K, d_k): K outer products cannot exceed the feature dim d_k=32."""
    rng = np.random.default_rng(11)
    k = jnp.asarray(rng.standard_normal((n_kv, 32)))  # elu -> d_k = 32
    v = jnp.asarray(rng.standard_normal((n_kv, 64)))  # d_v = 64 > d_k, so cap is min(K, d_k)
    s = linear_attention_state(k, v)
    assert int(jnp.linalg.matrix_rank(s)) == expected


def test_float64_vs_float32_normalizer():
    """float64 reaches the 1e-12 identity pin; float32 caps near 1e-7 (the precision symptom)."""
    r64 = recurrent_parallel_residual(512, dtype=jnp.float64)
    r32 = recurrent_parallel_residual(512, dtype=jnp.float32)
    assert r64 < 1e-12, f"float64 residual should hit the pin; got {r64:.2e}"
    assert r32 > 1e-8, f"float32 residual should be far larger; got {r32:.2e}"
    assert r32 / r64 > 1e4, f"float32 should be >=1e4x worse than float64; got ratio {r32 / r64:.1e}"


def test_shape_validation():
    q, k, v = _qkv(length=8)
    with pytest.raises(ValueError):
        linear_attention_recurrent(q, k[:-1], v)  # mismatched key length
    with pytest.raises(ValueError):
        linear_attention_parallel(q, k, v[:-1])  # mismatched value length
    with pytest.raises(ValueError):
        linear_attention_recurrent(q, k, v, feature_map="softmax")  # unknown phi
