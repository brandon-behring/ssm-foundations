r"""Tests for ch11 mqar_recall: the capacity mechanism behind the MQAR gap.

Pins the §11.6 claims (Proposition ``ch11:linattn-capacity``), all fixed-weight
(no training):

* orthonormal keys with $K \le d_k$ recall *exactly* (error ``< 1e-12``) — the
  rank bound is satisfiable below capacity;
* with generic keys the linear-attention retrieval error grows monotonically with
  $K$ and *shrinks* with the state size $d_k$ (capacity $\propto d_k$);
* the softmax oracle's error stays near zero independent of $K$ — the capacity-
  unbounded baseline.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from companions.ch11.jax.mqar_recall import (  # noqa: E402
    linear_retrieval_error,
    orthonormal_keys,
    orthonormal_values,
    random_unit_keys,
    softmax_retrieval_error,
)


def _mean_linear(n_pairs, dim, dim_v=160, n_seeds=4):
    return float(np.mean([
        linear_retrieval_error(random_unit_keys(n_pairs, dim, s), orthonormal_values(n_pairs, dim_v, 100 + s))
        for s in range(n_seeds)
    ]))


@pytest.mark.parametrize("n_pairs", [4, 16, 32])
def test_below_capacity_exact_recall(n_pairs):
    """Orthonormal keys with K <= d_k: linear attention recalls exactly (zero error)."""
    err = linear_retrieval_error(orthonormal_keys(n_pairs, 32, 0), orthonormal_values(n_pairs, 64, 1))
    assert err < 1e-12, f"orthonormal sub-capacity recall should be exact; got {err:.2e}"


def test_error_grows_with_K():
    """Generic-key retrieval error increases monotonically with the number of pairs."""
    errs = [_mean_linear(k, 32) for k in (8, 16, 32, 64, 128)]
    assert all(b > a for a, b in zip(errs, errs[1:])), f"error should grow with K; got {errs}"


def test_capacity_scales_with_dk():
    """At fixed K, a larger state d_k gives strictly smaller retrieval error."""
    for n_pairs in (16, 64, 128):
        assert _mean_linear(n_pairs, 64) < _mean_linear(n_pairs, 16), f"d_k=64 should beat d_k=16 at K={n_pairs}"


def test_softmax_oracle_near_zero():
    """The softmax oracle's retrieval error is ~0 and far below linear attention's."""
    for n_pairs in (16, 64, 128):
        soft = float(np.mean([
            softmax_retrieval_error(random_unit_keys(n_pairs, 64, s), orthonormal_values(n_pairs, 160, 100 + s))
            for s in range(4)
        ]))
        assert soft < 1e-3, f"softmax oracle should recall ~exactly; got {soft:.2e} at K={n_pairs}"
        assert soft < 0.05 * _mean_linear(n_pairs, 64), "softmax must be far below linear attention"


def test_shape_validation():
    keys = random_unit_keys(8, 16, 0)
    with pytest.raises(ValueError):
        linear_retrieval_error(keys, orthonormal_values(7, 16, 0))  # mismatched K
    with pytest.raises(ValueError):
        orthonormal_keys(20, 16)  # K > dim
    with pytest.raises(ValueError):
        orthonormal_values(20, 16)  # K > dim_v
