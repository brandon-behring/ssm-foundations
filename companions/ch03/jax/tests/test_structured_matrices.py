"""Assertion-based tests for the Chapter 3 structured-matrices companion (0527-F26).

Pins §3.4: the vectorised Toeplitz / Vandermonde / Cauchy / 1-semiseparable
constructions match independent oracles (scipy / NumPy), and the semiseparable
matrix realizes the scalar recurrence $h_t = a_t h_{t-1} + b_t$ it claims to.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from companions.ch03.jax import structured_matrices as sm


def _semisep_numpy(factors: np.ndarray) -> np.ndarray:
    """Independent NumPy double-loop oracle for the 1-semiseparable matrix."""
    n = len(factors) + 1
    M = np.eye(n)
    for j in range(n):
        for i in range(j + 1, n):
            M[i, j] = np.prod(factors[j:i])
    return M


def test_toeplitz_matches_scipy() -> None:
    first_col = np.array([3.0, 1.0, 0.5, 0.25, 0.125])
    first_row = np.array([3.0, 2.0, 0.0, 0.0, 0.0])
    got = np.asarray(sm.toeplitz(first_col, first_row))
    np.testing.assert_allclose(got, scipy.linalg.toeplitz(first_col, first_row), atol=1e-12)


def test_toeplitz_default_is_lower_triangular() -> None:
    got = np.asarray(sm.toeplitz(np.array([2.0, 1.0, 0.5])))
    assert np.allclose(np.triu(got, k=1), 0.0)  # strictly upper part is zero


def test_vandermonde_matches_numpy() -> None:
    nodes = np.linspace(0.5, 1.2, 6)
    np.testing.assert_allclose(
        np.asarray(sm.vandermonde(nodes)), np.vander(nodes, increasing=True), atol=1e-12
    )


def test_cauchy_values() -> None:
    xs = np.linspace(0.1, 1.0, 5)
    ys = np.linspace(1.5, 2.5, 5)
    got = np.asarray(sm.cauchy(xs, ys))
    np.testing.assert_allclose(got, 1.0 / (xs[:, None] - ys[None, :]), atol=1e-12)


def test_cauchy_disjoint_validation() -> None:
    with pytest.raises(ValueError):
        sm.cauchy(np.array([1.0, 2.0]), np.array([2.0, 3.0]))  # shares 2.0


def test_semiseparable_matches_numpy() -> None:
    factors = 0.9 * np.ones(7)
    np.testing.assert_allclose(
        np.asarray(sm.one_semiseparable(factors)), _semisep_numpy(factors), atol=1e-12
    )


def test_semiseparable_realizes_recurrence() -> None:
    """M @ b reproduces h_t = a_t h_{t-1} + b_t (the docstring's stated meaning)."""
    factors = np.array([0.9, 0.8, 0.7, 0.6])
    b = np.array([1.0, -0.5, 2.0, 0.3, 1.1])
    h_matrix = np.asarray(sm.one_semiseparable(factors)) @ b
    h = np.zeros(5)
    h[0] = b[0]
    for t in range(1, 5):
        h[t] = factors[t - 1] * h[t - 1] + b[t]
    np.testing.assert_allclose(h_matrix, h, atol=1e-12)
