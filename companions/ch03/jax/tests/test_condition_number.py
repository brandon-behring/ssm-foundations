"""Assertion-based tests for the Chapter 3 condition-number companion (0527-F26).

Pins §3.3, and is the **regression guard for 0527-F14**: the HiPPO-LegS condition
number grows ~quadratically in N (log-log slope ≈ 2), *not* "bounded". If the
construction or the claim ever rots back toward boundedness, the slope guard and
the explicit growth-ratio check below fail.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from companions.ch03.jax import condition_number as cn


def _hippo_numpy(N: int) -> np.ndarray:
    """Independent NumPy double-loop oracle for the HiPPO-LegS matrix."""
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i > j:
                A[i, j] = -np.sqrt((2 * i + 1) * (2 * j + 1))
            elif i == j:
                A[i, j] = -(i + 1)
    return A


def _loglog_slope(kappa: np.ndarray, sizes: np.ndarray) -> float:
    """Least-squares slope of log κ vs log N."""
    return float(np.polyfit(np.log(sizes), np.log(kappa), 1)[0])


def test_hippo_matches_numpy() -> None:
    for N in (1, 2, 8, 32):
        np.testing.assert_allclose(np.asarray(cn.hippo_legs(N)), _hippo_numpy(N), atol=1e-12)


def test_hippo_grows_subquadratically() -> None:
    """F14 guard: κ(HiPPO-LegS) log-log slope ≈ 2 (polynomial, not bounded)."""
    sizes = np.array([8, 16, 32, 64, 128])
    kappa = np.array([float(jnp.linalg.cond(cn.hippo_legs(int(N)))) for N in sizes])
    slope = _loglog_slope(kappa, sizes)
    assert 1.7 <= slope <= 2.2, f"HiPPO κ slope {slope:.3f} not ≈ 2 (quadratic growth)"


def test_hippo_is_not_bounded() -> None:
    """Directly refute the old 'κ stays bounded' claim: κ(N=128) ≫ κ(N=8)."""
    ratio = float(jnp.linalg.cond(cn.hippo_legs(128))) / float(jnp.linalg.cond(cn.hippo_legs(8)))
    assert ratio > 50.0, f"κ(128)/κ(8) = {ratio:.1f}; a 'bounded' κ would give ≈ 1"


def test_hilbert_ill_conditioned() -> None:
    assert float(jnp.linalg.cond(cn.hilbert(16))) > 1e10


def test_gaussian_grows_slowly() -> None:
    """Random Gaussian κ grows ~linearly — far below HiPPO's quadratic slope."""
    sizes = np.array([8, 16, 32, 64, 128])
    kappa = np.array([float(jnp.linalg.cond(cn.random_gaussian(int(N)))) for N in sizes])
    assert _loglog_slope(kappa, sizes) < 1.6


def test_hippo_validation() -> None:
    with pytest.raises(ValueError):
        cn.hippo_legs(0)
