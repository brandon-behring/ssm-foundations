"""Tests for the Chapter 7 PyTorch HiPPO companion (0527-F26).

Mirrors the JAX companion's numeric claims so the two stay in lock-step (the
cross-framework-consistency goal of the three-language Ch 7 confluence): the closed
form, the spectrum, the eager-loop encoder vs a naive NumPy loop, and reconstruction
errors that reproduce the JAX/Julia values. ``HiPPOEncoder`` must have zero learnable
parameters (HiPPO is fixed initialization, not a trained layer).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from companions.ch07.torch import hippo_operator as ho

# Reconstruction errors produced by the JAX and Julia companions (identical algorithm).
# The torch companion must reproduce them — the strong cross-framework-consistency claim.
_JAX_RECON = {4: 7.991e-01, 8: 3.925e-01, 16: 1.1245e-02, 32: 7.227e-03, 64: 7.230e-03}


def _oracle_legs(n: int) -> np.ndarray:
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i > j:
                A[i, j] = -np.sqrt((2 * i + 1) * (2 * j + 1))
            elif i == j:
                A[i, j] = -(i + 1)
    return A


def test_matrix_matches_closed_form() -> None:
    """torch A == the §7.3 closed-form oracle (transitively == the JAX/Julia matrix)."""
    for n in (4, 6, 8):
        A = ho.make_hippo_legs(n)[0].numpy()
        np.testing.assert_allclose(A, _oracle_legs(n), atol=1e-12)


def test_B_vector() -> None:
    B = ho.make_hippo_legs(8)[1].numpy().squeeze(-1)
    np.testing.assert_allclose(B, np.sqrt(2 * np.arange(8) + 1.0), atol=1e-12)


def test_eigenvalues_minus_one_to_minus_n() -> None:
    for n in (4, 8, 16, 32):
        eigs = ho.legs_eigenvalues(n).numpy()
        assert np.max(np.abs(eigs.imag)) < 1e-8
        np.testing.assert_allclose(np.sort(eigs.real), np.sort(-(np.arange(n) + 1.0)), atol=1e-8)


def test_encoder_zero_learnable_params() -> None:
    """HiPPO matrices are buffers, not Parameters — the operator has nothing to train."""
    enc = ho.HiPPOEncoder(16)
    assert sum(p.numel() for p in enc.parameters()) == 0
    # but the buffers are present
    assert enc.A_pos.shape == (16, 16)


def test_forward_shape() -> None:
    enc = ho.HiPPOEncoder(8)
    u = torch.zeros(32, dtype=torch.float64)
    assert enc(u).shape == (32, 8)


def test_forward_matches_naive_numpy_loop() -> None:
    """The eager Module loop equals an independent NumPy accumulation (the loop guard)."""
    n, L = 8, 60
    z = np.linspace(0.0, 1.0, L)
    u = np.sin(2.0 * np.pi * 2.0 * z)
    with torch.no_grad():
        got = ho.HiPPOEncoder(n)(torch.as_tensor(u, dtype=torch.float64)).numpy()

    A = ho.make_hippo_legs(n)[0].numpy()
    B = ho.make_hippo_legs(n)[1].numpy().squeeze(-1)
    A_pos = -A
    eye = np.eye(n)
    c = np.zeros(n)
    ref = np.zeros((L, n))
    for k in range(1, L + 1):
        c = np.linalg.solve(eye + A_pos / (2.0 * k), (eye - A_pos / (2.0 * k)) @ c + (B / k) * u[k - 1])
        ref[k - 1] = c
    np.testing.assert_allclose(got, ref, atol=1e-10)


def test_reconstruction_decreases_with_N() -> None:
    errs = {n: ho.reconstruction_error(n) for n in (4, 8, 16, 64)}
    assert errs[8] < errs[4]
    assert errs[16] < errs[8]
    assert errs[16] < 0.05
    assert errs[64] < 0.01


def test_reconstruction_matches_jax_numbers() -> None:
    """Cross-framework consistency: torch reproduces the JAX/Julia reconstruction errors."""
    for n, expected in _JAX_RECON.items():
        got = ho.reconstruction_error(n)
        assert abs(got - expected) < 1e-4, f"N={n}: torch {got:.5e} vs jax {expected:.5e}"


def test_validation() -> None:
    with pytest.raises(ValueError):
        ho.make_hippo_legs(0)
    with pytest.raises(ValueError):
        ho.HiPPOEncoder(0)
    with pytest.raises(ValueError):
        ho.HiPPOEncoder(8)(torch.zeros((3, 3), dtype=torch.float64))  # not 1-D
