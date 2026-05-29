"""Tests for the Chapter 1 PyTorch matrix-exponential companion (0527-F26).

Mirrors the JAX companion's numeric claims so the two stay in agreement: the
truncated series converges to ``torch.linalg.matrix_exp`` (fast for small ‖M‖,
slow for large), and the eager partial-sum loop matches a NumPy accumulation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from companions.ch01.torch import matrix_exponential as me

_A_SMALL = np.array([[-0.5, 1.0], [-1.0, -0.5]])
_A_LARGE = np.array([[-5.0, 10.0], [-10.0, -5.0]])


def test_series_converges_for_small_norm() -> None:
    errs = me.convergence_errors(_A_SMALL, 39)
    assert errs[-1] < 1e-10, f"small-norm series did not converge: {errs[-1]:.2e}"


def test_series_slow_for_large_norm() -> None:
    errs_large = me.convergence_errors(_A_LARGE, 39)
    errs_small = me.convergence_errors(_A_SMALL, 39)
    assert errs_large[5] > 1.0
    assert errs_large[-1] > 1e3 * errs_small[-1]


def test_matches_matrix_exp() -> None:
    got = me.truncated_series(_A_SMALL, 40)
    ref = torch.linalg.matrix_exp(torch.as_tensor(_A_SMALL, dtype=torch.float64))
    np.testing.assert_allclose(got.numpy(), ref.numpy(), atol=1e-12)


def test_exp_zero_is_identity() -> None:
    np.testing.assert_allclose(me.truncated_series(np.zeros((3, 3)), 10).numpy(), np.eye(3), atol=1e-15)


def test_loop_matches_numpy() -> None:
    """The eager partial-sum loop equals an independent NumPy accumulation."""
    k_max = 12
    got = me.series_partial_sums(_A_SMALL, k_max).numpy()
    term = np.eye(2)
    total = np.eye(2)
    ref = [total.copy()]
    for k in range(1, k_max + 1):
        term = term @ _A_SMALL / k
        total = total + term
        ref.append(total.copy())
    np.testing.assert_allclose(got, np.stack(ref), atol=1e-12)


def test_truncated_series_validation() -> None:
    with pytest.raises(ValueError):
        me.truncated_series(_A_SMALL, -1)
    with pytest.raises(ValueError):
        me.truncated_series(np.ones((2, 3)), 5)
