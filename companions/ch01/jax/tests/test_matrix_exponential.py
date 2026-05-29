"""Assertion-based tests for the Chapter 1 matrix-exponential companion (0527-F26).

Pins the pedagogical claims of §1.2: the truncated series converges to
``scipy.linalg.expm`` (fast for small spectral radius, slowly for large), the
``lax.scan`` partial-sum recurrence matches a naive NumPy accumulation, and the
spectral identity $e^{A} = V e^{\\Lambda} V^{-1}$ holds for diagonalizable $A$.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from companions.ch01.jax import matrix_exponential as me

_A_SMALL = np.array([[-0.5, 1.0], [-1.0, -0.5]])
_A_LARGE = np.array([[-5.0, 10.0], [-10.0, -5.0]])


def test_series_converges_for_small_norm() -> None:
    errs = me.convergence_errors(_A_SMALL, 39)
    assert errs[-1] < 1e-10, f"small-norm series did not converge: {errs[-1]:.2e}"


def test_series_slow_for_large_norm() -> None:
    errs_large = me.convergence_errors(_A_LARGE, 39)
    errs_small = me.convergence_errors(_A_SMALL, 39)
    # Few terms are nowhere near (the k=39 term is still ~0.45 in magnitude for
    # ‖M‖≈14), and even at K=39 the large-norm series is many orders worse than
    # the small-norm one — the "slow/catastrophic for large ‖M‖" lesson of §1.2.
    assert errs_large[5] > 1.0
    assert errs_large[-1] > 1e3 * errs_small[-1]


def test_truncated_series_matches_scipy() -> None:
    np.testing.assert_allclose(
        np.asarray(me.truncated_series(_A_SMALL, 40)), expm(_A_SMALL), atol=1e-12
    )


def test_exp_zero_is_identity() -> None:
    np.testing.assert_allclose(
        np.asarray(me.truncated_series(np.zeros((3, 3)), 10)), np.eye(3), atol=1e-15
    )


def test_scan_matches_numpy_loop() -> None:
    """series_partial_sums (lax.scan) equals an independent NumPy accumulation."""
    k_max = 12
    got = np.asarray(me.series_partial_sums(np.asarray(_A_SMALL), k_max))
    n = _A_SMALL.shape[0]
    term = np.eye(n)
    total = np.eye(n)
    ref = [total.copy()]
    for k in range(1, k_max + 1):
        term = term @ _A_SMALL / k
        total = total + term
        ref.append(total.copy())
    np.testing.assert_allclose(got, np.stack(ref), atol=1e-12)


def test_diagonalizable_spectral_identity() -> None:
    """e^{A} = V e^{Λ} V^{-1} for a diagonalizable A (the §1.2 eigen-decomposition)."""
    lam = np.array([-0.3, -1.7])
    V = np.array([[1.0, 1.0], [1.0, -1.0]])
    A = V @ np.diag(lam) @ np.linalg.inv(V)
    expected = V @ np.diag(np.exp(lam)) @ np.linalg.inv(V)
    np.testing.assert_allclose(np.asarray(me.truncated_series(A, 40)), expected, atol=1e-10)


def test_truncated_series_validation() -> None:
    with pytest.raises(ValueError):
        me.truncated_series(_A_SMALL, -1)
    with pytest.raises(ValueError):
        me.truncated_series(np.ones((2, 3)), 5)
