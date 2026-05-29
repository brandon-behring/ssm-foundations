"""Assertion-based tests for the Chapter 7 JAX HiPPO companions.

Pins the pedagogical numerical claims of Ch 7 so they cannot silently rot
(audit 0527-F26), mirroring ``companions/ch04/jax/tests/test_discretization.py``:

* the HiPPO-LegS closed form (§7.3) — exact entries, lower-triangular structure;
* the spectrum (§7.7) — eigenvalues are exactly $-1,\\ldots,-N$;
* online function approximation (§7.1) — reconstruction error falls as $N$ grows;
* the ``lax.scan`` recurrence equals an independent NumPy loop (the scan-refactor guard).

Run: ``PYTHONPATH=. .venv/bin/pytest companions/ch07/jax/tests -q``
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from companions.ch07.jax import hippo_matrix as hm
from companions.ch07.jax import hippo_reconstruction as hr


# ---------------------------------------------------------------------------
# §7.3 — HiPPO-LegS closed form
# ---------------------------------------------------------------------------


def _expected_legs(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Reference closed form built with explicit Python loops (the oracle)."""
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i > j:
                A[i, j] = -np.sqrt((2 * i + 1) * (2 * j + 1))
            elif i == j:
                A[i, j] = -(i + 1)
    B = np.sqrt(2 * np.arange(n) + 1.0)[:, None]
    return A, B


def test_hippo_shapes() -> None:
    for n in (1, 4, 8, 16):
        A, B = hm.make_hippo_legs(n)
        assert A.shape == (n, n), f"A shape for N={n}: {A.shape}"
        assert B.shape == (n, 1), f"B shape for N={n}: {B.shape}"


def test_hippo_matches_closed_form() -> None:
    """Every entry of A and B must match the §7.3 closed form exactly."""
    for n in (4, 6, 8):
        A_exp, B_exp = _expected_legs(n)
        A, B = hm.make_hippo_legs(n)
        np.testing.assert_allclose(np.asarray(A), A_exp, atol=1e-12)
        np.testing.assert_allclose(np.asarray(B), B_exp, atol=1e-12)


def test_hippo_lower_triangular() -> None:
    """A is lower-triangular: the strict upper triangle is exactly zero."""
    A = np.asarray(hm.make_hippo_legs(16)[0])
    assert np.max(np.abs(np.triu(A, k=1))) == 0.0


def test_hippo_named_entry() -> None:
    """Spot-check A[3,1] for N=8 against -sqrt(7*3) (a load-bearing prose value)."""
    A = np.asarray(hm.make_hippo_legs(8)[0])
    assert A[3, 1] == pytest.approx(-np.sqrt(7 * 3), abs=1e-12)


# ---------------------------------------------------------------------------
# §7.7 — spectrum
# ---------------------------------------------------------------------------


def test_eigenvalues_are_minus_one_to_minus_n() -> None:
    """Eigenvalues of HiPPO-LegS A are exactly -1, -2, ..., -N (stable, real)."""
    for n in (4, 8, 16, 32):
        eigs = np.asarray(hm.legs_eigenvalues(n))
        assert np.max(np.abs(eigs.imag)) < 1e-8, "eigenvalues should be real"
        got = np.sort(eigs.real)
        expected = np.sort(-(np.arange(n) + 1.0))
        np.testing.assert_allclose(got, expected, atol=1e-8)


def test_eigenvalues_negative_real_part() -> None:
    """All eigenvalues strictly in the open left half-plane (asymptotic stability)."""
    eigs = np.asarray(hm.legs_eigenvalues(32))
    assert np.all(eigs.real < 0.0)


# ---------------------------------------------------------------------------
# §7.1 — online function approximation
# ---------------------------------------------------------------------------


def test_reconstruction_error_decreases_with_N() -> None:
    """More Legendre modes ⇒ better reconstruction, down to the discretization floor."""
    errs = hr.reconstruction_errors((4, 8, 16, 64))
    assert errs[8] < errs[4], f"N=8 ({errs[8]:.3e}) should beat N=4 ({errs[4]:.3e})"
    assert errs[16] < errs[8], f"N=16 ({errs[16]:.3e}) should beat N=8 ({errs[8]:.3e})"
    assert errs[16] < 0.05, f"N=16 should capture the signal: {errs[16]:.3e}"
    assert errs[64] < 0.01, f"N=64 reaches the floor: {errs[64]:.3e}"


def test_reconstruction_uses_converging_convention() -> None:
    """The locked (normalized, non-alternating) convention beats its alternatives at N=32."""
    n = 32
    locked = hr.reconstruction_errors((n,), normalized=True, alternating=False)[n]
    for nm, al in ((True, True), (False, True), (False, False)):
        other = hr.reconstruction_errors((n,), normalized=nm, alternating=al)[n]
        assert locked < other, f"locked {locked:.3e} should beat ({nm},{al}) {other:.3e}"


# ---------------------------------------------------------------------------
# scan-refactor guard (mirrors ch04 test_scan_matches_naive_loop)
# ---------------------------------------------------------------------------


def test_encode_scan_matches_naive_loop() -> None:
    """The lax.scan LegS encoder equals an independent NumPy loop of the same recurrence."""
    n, L = 8, 60
    z = np.linspace(0.0, 1.0, L)
    u = np.sin(2.0 * np.pi * 2.0 * z)
    got = np.asarray(hr.hippo_legs_encode(jnp.asarray(u), n))

    A_pos = -np.asarray(hm.make_hippo_legs(n)[0])
    B = np.asarray(hm.make_hippo_legs(n)[1]).squeeze(-1)
    eye = np.eye(n)
    c = np.zeros(n)
    ref = np.zeros((L, n))
    for k in range(1, L + 1):
        lhs = eye + A_pos / (2.0 * k)
        rhs = (eye - A_pos / (2.0 * k)) @ c + (B / k) * u[k - 1]
        c = np.linalg.solve(lhs, rhs)
        ref[k - 1] = c
    np.testing.assert_allclose(got, ref, atol=1e-10)


# ---------------------------------------------------------------------------
# input validation (no silent failure)
# ---------------------------------------------------------------------------


def test_make_hippo_legs_validation() -> None:
    with pytest.raises(ValueError):
        hm.make_hippo_legs(0)


def test_encode_validation() -> None:
    with pytest.raises(ValueError):
        hr.hippo_legs_encode(jnp.zeros((3, 3)), 8)  # not 1-D
    with pytest.raises(ValueError):
        hr.hippo_legs_encode(jnp.zeros(10), 0)  # n < 1
