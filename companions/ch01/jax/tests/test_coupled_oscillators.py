"""Assertion-based tests for the Chapter 1 coupled-oscillators companion (0527-F26).

Pins the §1.4 claims: the vectorised ``.at[].add`` ring-matrix assembly matches a
naive NumPy double-loop, the damped ring is asymptotically stable (all eigenvalues
in the open left half-plane), and the undamped ring is marginally stable
(eigenvalues on the imaginary axis).
"""

from __future__ import annotations

import numpy as np
import pytest

from companions.ch01.jax import coupled_oscillators as co


def _ring_numpy(n: int, k: float = 4.0, c: float = 0.2, kappa: float = 1.0) -> np.ndarray:
    """Independent NumPy double-loop oracle (the implementation being replaced)."""
    A = np.zeros((2 * n, 2 * n))
    for i in range(n):
        q_idx, v_idx = 2 * i, 2 * i + 1
        A[q_idx, v_idx] = 1.0
        A[v_idx, q_idx] = -k - 2.0 * kappa
        A[v_idx, v_idx] = -c
        A[v_idx, 2 * ((i - 1) % n)] = kappa
        A[v_idx, 2 * ((i + 1) % n)] = kappa
    return A


def test_ring_matches_numpy_loop() -> None:
    """Vectorised scatter equals the naive NumPy double-loop construction."""
    for n in (3, 5, 8, 16):
        got = np.asarray(co.build_ring_state_matrix(n))
        np.testing.assert_allclose(got, _ring_numpy(n), atol=1e-12)


def test_damped_ring_in_left_half_plane() -> None:
    """Damped ring (c>0) is asymptotically stable: max Re(λ) < 0."""
    A = co.build_ring_state_matrix(n=8, c=0.2)
    eigs = np.linalg.eigvals(np.asarray(A))
    assert eigs.real.max() < 0.0, f"max Re(λ) = {eigs.real.max():.3e} not < 0"


def test_undamped_ring_on_imaginary_axis() -> None:
    """Undamped ring (c=0) is marginally stable: eigenvalues are purely imaginary."""
    A = co.build_ring_state_matrix(n=8, c=0.0)
    eigs = np.linalg.eigvals(np.asarray(A))
    assert np.max(np.abs(eigs.real)) < 1e-9, f"max |Re(λ)| = {np.max(np.abs(eigs.real)):.3e}"


def test_build_ring_validation() -> None:
    with pytest.raises(ValueError):
        co.build_ring_state_matrix(n=2)
    with pytest.raises(ValueError):
        co.build_ring_state_matrix(n=8, k=-1.0)
    with pytest.raises(ValueError):
        co.build_ring_state_matrix(n=8, kappa=-1.0)
