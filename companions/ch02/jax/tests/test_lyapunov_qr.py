"""Assertion-based tests for the Chapter 2 QR-Lyapunov companion (0527-F26).

Pins the §2.3 claims: the ``lax.scan`` Benettin spectrum matches an independent
NumPy QR loop and the closed-form $\\Re(\\lambda_i)$ reference; the damped ring
contracts in every direction while the undamped ring is marginally stable; and
the exponents sum to $\\mathrm{tr}(A)$ (the Liouville/divergence identity — the
regression guard against sign/normalization bugs).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from companions.ch01.jax import coupled_oscillators as co
from companions.ch02.jax import lyapunov_qr as lq

_DT = 0.05
_N_STEPS = 2000


def _benettin_numpy(jacobians: np.ndarray, n_steps: int) -> np.ndarray:
    """Independent NumPy Benettin QR loop oracle for the scan implementation."""
    T, N, _ = jacobians.shape
    Q = np.eye(N)
    acc = np.zeros(N)
    for t in range(n_steps):
        Q, R = np.linalg.qr(jacobians[t % T] @ Q)
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1.0
        Q = Q * signs[np.newaxis, :]
        R = signs[:, np.newaxis] * R
        acc += np.log(np.abs(np.diag(R)) + 1e-300)
    return np.sort(acc / n_steps)[::-1]


def _discrete_jacobian(c: float) -> np.ndarray:
    A = np.asarray(co.build_ring_state_matrix(n=8, k=4.0, c=c, kappa=1.0))
    return expm(A * _DT)


def test_autonomous_matches_eigenvalue_reference() -> None:
    # All 8 ring modes share Re(λ) = -c/2 = -0.1, so the spectrum is degenerate;
    # the QR method cannot cleanly separate equal exponents and leaves O(1e-2)
    # separation noise on the individual values (their mean/sum stays exact — see
    # test_exponent_sum_equals_trace). Hence the loose elementwise tolerance here.
    A = np.asarray(co.build_ring_state_matrix(n=8, c=0.2))
    spec = lq.qr_lyapunov(_discrete_jacobian(0.2)[np.newaxis], _N_STEPS) / _DT
    ref = lq.autonomous_lyapunov_reference(A, _DT)
    np.testing.assert_allclose(spec, ref, atol=1.5e-2)
    assert abs(spec.mean() - ref.mean()) < 1e-3, "mean exponent should match tightly"


def test_damped_ring_all_negative() -> None:
    spec = lq.qr_lyapunov(_discrete_jacobian(0.2)[np.newaxis], _N_STEPS) / _DT
    assert spec.max() < 0.0, f"max λ = {spec.max():.3e} not < 0 (damped ring should contract)"


def test_undamped_ring_near_zero() -> None:
    spec = lq.qr_lyapunov(_discrete_jacobian(0.0)[np.newaxis], _N_STEPS) / _DT
    assert np.max(np.abs(spec)) < 1e-2, f"max |λ| = {np.max(np.abs(spec)):.3e} not ≈ 0"


def test_exponent_sum_equals_trace() -> None:
    """Regression guard: Σλ_i = tr(A) (= -n·c for the ring); catches sign/scale bugs."""
    A = np.asarray(co.build_ring_state_matrix(n=8, c=0.2))
    spec = lq.qr_lyapunov(_discrete_jacobian(0.2)[np.newaxis], _N_STEPS) / _DT
    assert abs(spec.sum() - np.trace(A)) < 1e-2, (
        f"Σλ = {spec.sum():.4f} vs tr(A) = {np.trace(A):.4f}"
    )


def test_scan_matches_numpy_benettin() -> None:
    """lax.scan spectrum equals an independent NumPy QR loop on a time-varying sequence."""
    rng = np.random.default_rng(0)
    jacobians = rng.standard_normal((3, 4, 4)) * 0.5 + np.eye(4)
    got = lq.qr_lyapunov(jacobians, 80)
    np.testing.assert_allclose(got, _benettin_numpy(jacobians, 80), atol=1e-10)


def test_qr_lyapunov_validation() -> None:
    with pytest.raises(ValueError):
        lq.qr_lyapunov(np.zeros((4, 4)), 10)  # not 3-D
    with pytest.raises(ValueError):
        lq.qr_lyapunov(np.zeros((1, 4, 4)), 0)  # n_steps < 1
