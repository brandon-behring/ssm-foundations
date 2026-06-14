r"""Cross-framework parity: PyTorch Ch 1 matrix-exponential vs the JAX reference (§1.2).

Backfills JAX↔torch parity for Ch 1 (audit 0527-F27). The companion's existing torch
test (``test_torch_matrix_exponential.py``) pins torch against its own
``torch.linalg.matrix_exp`` oracle; this file instead locks the *two companion
languages* together — the truncated-series partial sums must agree on identical
inputs, in float64 — matching the cross-framework-consistency goal Chapter 7
introduced and the dedicated parity file Chapter 4 carries.

Run: ``PYTHONPATH=. .venv/bin/pytest companions/ch01/torch -q``
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from companions.ch01.jax import matrix_exponential as jax_me  # noqa: E402
from companions.ch01.torch import matrix_exponential as torch_me  # noqa: E402

_A_SMALL = np.array([[-0.5, 1.0], [-1.0, -0.5]])
_A_LARGE = np.array([[-5.0, 10.0], [-10.0, -5.0]])
_PARITY_TOL = 1e-9


def test_truncated_series_matches_jax() -> None:
    """torch $S_K$ equals the JAX $S_K$ on identical inputs (small and large ‖M‖)."""
    for A in (_A_SMALL, _A_LARGE):
        for K in (5, 20, 39):
            j = np.asarray(jax_me.truncated_series(A, K))
            t = torch_me.truncated_series(A, K).numpy()
            assert np.max(np.abs(j - t)) < _PARITY_TOL, f"divergence at K={K}"


def test_partial_sum_stack_matches_jax() -> None:
    """The full partial-sum stack $S_0..S_k$ agrees across frameworks."""
    j = np.asarray(jax_me.series_partial_sums(_A_SMALL, 25))
    t = torch_me.series_partial_sums(_A_SMALL, 25).numpy()
    assert np.max(np.abs(j - t)) < _PARITY_TOL


def test_convergence_curve_matches_jax() -> None:
    """Per-order error curves coincide (scipy.expm and torch.matrix_exp agree ~1e-15)."""
    j = jax_me.convergence_errors(_A_SMALL, 39)
    t = torch_me.convergence_errors(_A_SMALL, 39)
    np.testing.assert_allclose(j, t, atol=_PARITY_TOL)
