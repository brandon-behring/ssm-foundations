"""Tests for the Chapter 3 PyTorch condition-number companion (0527-F26).

Mirrors the JAX companion, including the 0527-F14 guard: the HiPPO-LegS condition
number grows ~quadratically (log-log slope ≈ 2), *not* "bounded". Confirms the
PyTorch construction reproduces the NumPy/JAX matrix and κ growth.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from companions.ch03.torch import condition_number as cn


def _hippo_numpy(N: int) -> np.ndarray:
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i > j:
                A[i, j] = -np.sqrt((2 * i + 1) * (2 * j + 1))
            elif i == j:
                A[i, j] = -(i + 1)
    return A


def test_hippo_matches_numpy() -> None:
    for N in (1, 2, 8, 32):
        np.testing.assert_allclose(cn.hippo_legs(N).numpy(), _hippo_numpy(N), atol=1e-12)


def test_hippo_grows_subquadratically() -> None:
    """F14 guard (torch): κ(HiPPO-LegS) log-log slope ≈ 2, not bounded."""
    sizes = [8, 16, 32, 64, 128]
    kappa = np.array([float(torch.linalg.cond(cn.hippo_legs(N))) for N in sizes])
    slope = float(np.polyfit(np.log(sizes), np.log(kappa), 1)[0])
    assert 1.7 <= slope <= 2.2, f"HiPPO κ slope {slope:.3f} not ≈ 2"


def test_hilbert_ill_conditioned() -> None:
    assert float(torch.linalg.cond(cn.hilbert(16))) > 1e10


def test_hippo_validation() -> None:
    with pytest.raises(ValueError):
        cn.hippo_legs(0)
