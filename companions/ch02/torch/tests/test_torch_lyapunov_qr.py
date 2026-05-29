"""Tests for the Chapter 2 PyTorch QR-Lyapunov companion (0527-F26).

Mirrors the JAX companion: the eager Benettin QR loop matches the closed-form
$\\Re(\\lambda_i)$ reference (degenerate spectrum) and an independent NumPy loop,
the damped ring contracts, and the exponents sum to $\\mathrm{tr}(A)$.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from companions.ch02.torch import lyapunov_qr as lq

_DT = 0.05
_N_STEPS = 2000


def _benettin_numpy(jacobians: np.ndarray, n_steps: int) -> np.ndarray:
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


def _discrete_jacobian(c: float) -> torch.Tensor:
    A = lq.ring_state_matrix(n=8, c=c)
    return torch.linalg.matrix_exp(A * _DT)


def test_autonomous_matches_eigenvalue_reference() -> None:
    # Degenerate spectrum (all Re(λ) = -c/2): individual exponents carry O(1e-2)
    # separation noise, mean is tight — same as the JAX companion.
    A = lq.ring_state_matrix(n=8, c=0.2)
    spec = lq.qr_lyapunov(_discrete_jacobian(0.2).unsqueeze(0), _N_STEPS) / _DT
    ref = lq.autonomous_lyapunov_reference(A)
    np.testing.assert_allclose(spec, ref, atol=1.5e-2)
    assert abs(spec.mean() - ref.mean()) < 1e-3


def test_damped_ring_all_negative() -> None:
    spec = lq.qr_lyapunov(_discrete_jacobian(0.2).unsqueeze(0), _N_STEPS) / _DT
    assert spec.max() < 0.0, f"max λ = {spec.max():.3e} not < 0"


def test_exponent_sum_equals_trace() -> None:
    A = lq.ring_state_matrix(n=8, c=0.2)
    spec = lq.qr_lyapunov(_discrete_jacobian(0.2).unsqueeze(0), _N_STEPS) / _DT
    assert abs(spec.sum() - float(torch.trace(A))) < 1e-2


def test_loop_matches_numpy_benettin() -> None:
    rng = np.random.default_rng(0)
    jacobians = rng.standard_normal((3, 4, 4)) * 0.5 + np.eye(4)
    got = lq.qr_lyapunov(jacobians, 80)
    np.testing.assert_allclose(got, _benettin_numpy(jacobians, 80), atol=1e-10)


def test_qr_lyapunov_validation() -> None:
    with pytest.raises(ValueError):
        lq.qr_lyapunov(np.zeros((4, 4)), 10)
    with pytest.raises(ValueError):
        lq.qr_lyapunov(np.zeros((1, 4, 4)), 0)
    with pytest.raises(ValueError):
        lq.ring_state_matrix(n=2)
