r"""Cross-framework parity: PyTorch Ch 2 QR-Lyapunov vs the JAX reference (§2.3).

Backfills JAX↔torch parity for Ch 2 (audit 0527-F27). The existing torch test
(``test_torch_lyapunov_qr.py``) checks torch against a NumPy Benettin loop and the
closed-form $\Re(\lambda_i)$ oracle; this file locks the two companion languages
together — fed an *identical* Jacobian sequence, ``jax.lax.scan`` and the eager torch
loop must return the same Lyapunov spectrum (both fix the $\mathrm{diag}(R)>0$ sign
convention, so the QR factor is unique).

Run: ``PYTHONPATH=. .venv/bin/pytest companions/ch02/torch -q``
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from companions.ch02.jax import lyapunov_qr as jax_lq  # noqa: E402
from companions.ch02.torch import lyapunov_qr as torch_lq  # noqa: E402

_PARITY_TOL = 1e-9


def test_qr_spectrum_matches_jax_on_shared_jacobians() -> None:
    """Identical Jacobian sequence ⇒ identical Lyapunov spectrum across frameworks."""
    rng = np.random.default_rng(0)
    jacs = rng.standard_normal((3, 4, 4)) * 0.5 + np.eye(4)
    j = jax_lq.qr_lyapunov(jacs, 200)
    t = torch_lq.qr_lyapunov(jacs, 200)
    np.testing.assert_allclose(j, t, atol=_PARITY_TOL)


def test_autonomous_spectrum_matches_jax() -> None:
    """On the §2.3 damped ring, the full QR spectrum coincides across frameworks."""
    dt = 0.05
    A = torch_lq.ring_state_matrix(n=8, c=0.2).numpy()
    J = np.asarray(torch.linalg.matrix_exp(torch.as_tensor(A)))
    j = jax_lq.qr_lyapunov(J[np.newaxis, ...], 2000) / dt
    t = torch_lq.qr_lyapunov(J[np.newaxis, ...], 2000) / dt
    np.testing.assert_allclose(j, t, atol=_PARITY_TOL)


def test_autonomous_reference_matches_jax() -> None:
    r"""Closed-form $\Re(\mathrm{eigvals})$ reference agrees (both sort descending)."""
    A = torch_lq.ring_state_matrix(n=8, c=0.2).numpy()
    j = jax_lq.autonomous_lyapunov_reference(A, 0.05)  # jax sig (A, dt); dt unused
    t = torch_lq.autonomous_lyapunov_reference(A)
    np.testing.assert_allclose(j, t, atol=_PARITY_TOL)
