r"""Assertion tests for the Chapter 8 S4D (§8.5) and S5 (§8.6) JAX companions.

Pins the load-bearing claims:

* **S4D stability is structural** — $\mathrm{Re}(A) < 0$ for *any* parameter
  values, and $|\bar A_n| < 1$ under ZOH (§8.5, the cure for the §7.5 danger);
* the S4D **Vandermonde kernel** equals an independent complex-diagonal
  recurrence impulse response (cross-checks the two formulas);
* **S5 scan-equivalence** — the parallel associative scan equals the sequential
  recurrence to machine precision (§8.6, Exercise 8.6).

Run: ``PYTHONPATH=. .venv/bin/pytest companions/ch08/jax/tests/test_s4d_s5.py -q``
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from companions.ch08.jax import s4d_kernel as s4d
from companions.ch08.jax import s5_scan as s5

# ---------------------------------------------------------------------------
# §8.5 — S4D stability is enforced by construction
# ---------------------------------------------------------------------------


def test_s4d_real_part_negative_for_any_parameters() -> None:
    r"""$A = -e^{\texttt{log\_A\_real}} + i\,\texttt{A\_imag} \Rightarrow \mathrm{Re}(A) < 0$ always."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        # Deliberately extreme/odd parameter values — stability must survive all.
        log_A_real = jnp.asarray(rng.uniform(-50.0, 50.0, size=8))
        A_imag = jnp.asarray(rng.standard_normal(8) * 100.0)
        A = s4d.assemble_diagonal(log_A_real, A_imag)
        assert np.all(np.asarray(A.real) < 0.0), "Re(A) must be < 0 by construction"


def test_s4d_lin_modes_and_zoh_stability() -> None:
    r"""S4D-Lin gives $A_n = -\tfrac12 + i\pi n$ and $|\bar A_n| = e^{-\Delta/2} < 1$."""
    A = np.asarray(s4d.make_s4d_lin(16))
    np.testing.assert_allclose(A.real, -0.5, atol=1e-12, rtol=0)
    np.testing.assert_allclose(A.imag, np.pi * np.arange(16), atol=1e-12, rtol=0)
    for dt in (0.01, 0.1, 1.0):
        Abar = np.exp(A * dt)
        assert np.max(np.abs(Abar)) < 1.0, f"S4D unstable at dt={dt}"


def test_s4d_lin_params_validation() -> None:
    with pytest.raises(ValueError):
        s4d.s4d_lin_params(0)


# ---------------------------------------------------------------------------
# §8.5 — Vandermonde kernel == independent recurrence impulse response
# ---------------------------------------------------------------------------


def _s4d_recurrence_kernel(A: np.ndarray, C: np.ndarray, dt: float, L: int) -> np.ndarray:
    """Independent oracle: impulse response of the complex diagonal recurrence.

    $h_k = \\bar A \\odot h_{k-1} + \\bar B u_k$ with an impulse $u_0 = 1$, read out
    as $y_k = 2\\,\\mathrm{Re}(C \\cdot h_k)$ — the kernel by a different route.
    """
    Abar = np.exp(A * dt)
    Bbar = (Abar - 1.0) / A
    h = np.zeros_like(A)
    K = np.zeros(L)
    for k in range(L):
        u_k = 1.0 if k == 0 else 0.0
        h = Abar * h + Bbar * u_k
        K[k] = 2.0 * np.real(np.sum(C * h))
    return K


def test_s4d_kernel_matches_recurrence_oracle() -> None:
    """The closed-form Vandermonde kernel equals the recurrence impulse response."""
    n_modes, dt, L = 12, 0.1, 96
    A = s4d.make_s4d_lin(n_modes)
    rng = np.random.default_rng(1)
    C = jnp.asarray(rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes))
    K_vander = np.asarray(s4d.s4d_kernel(A, C, dt, L))
    K_recur = _s4d_recurrence_kernel(np.asarray(A), np.asarray(C), dt, L)
    np.testing.assert_allclose(K_vander, K_recur, atol=1e-9, rtol=0)


def test_s4d_kernel_validation() -> None:
    A = s4d.make_s4d_lin(4)
    with pytest.raises(ValueError):
        s4d.s4d_kernel(A, A[:3], 0.1, 16)  # mismatched C length
    with pytest.raises(ValueError):
        s4d.s4d_kernel(A, A, 0.1, 0)  # L < 1
    with pytest.raises(ValueError):
        z = jnp.zeros(4, dtype=A.dtype)
        s4d.s4d_kernel(z, jnp.ones(4, dtype=A.dtype), 0.1, 16)  # singular (zero) mode


# ---------------------------------------------------------------------------
# §8.6 — S5 scan-equivalence (parallel associative scan == sequential)
# ---------------------------------------------------------------------------


def test_s5_parallel_equals_sequential_states() -> None:
    """THE load-bearing S5 test (§8.6): associative scan ≡ sequential recurrence."""
    n_modes = 8
    A = s4d.make_s4d_lin(n_modes)
    Abar = np.exp(np.asarray(A) * 0.1)
    rng = np.random.default_rng(2)
    for L in (16, 64, 257):  # include a non-power-of-two length
        Bu = jnp.asarray(rng.standard_normal((L, n_modes)) + 1j * rng.standard_normal((L, n_modes)))
        hp = s5.s5_parallel_scan(jnp.asarray(Abar), Bu)
        hs = s5.s5_sequential_scan(jnp.asarray(Abar), Bu)
        diff = float(jnp.max(jnp.abs(hp - hs)))
        assert diff < 1e-12, f"scan mismatch at L={L}: {diff:.3e}"


def test_s5_apply_parallel_equals_sequential() -> None:
    """Full MIMO forward agrees between the two scan modes (real outputs)."""
    A, B, C = s5._demo_system(n_modes=8, h_dim=4)
    L = 200
    z = jnp.linspace(0.0, 1.0, L)
    u = jnp.stack([jnp.sin(2 * jnp.pi * (k + 1) * z) for k in range(B.shape[1])], axis=1)
    y_par = s5.s5_apply(A, B, C, 0.1, u, parallel=True)
    y_seq = s5.s5_apply(A, B, C, 0.1, u, parallel=False)
    np.testing.assert_allclose(np.asarray(y_par), np.asarray(y_seq), atol=1e-12, rtol=0)


def test_s5_binary_operator_associative() -> None:
    """The scan operator is associative: (x . y) . z == x . (y . z)."""
    rng = np.random.default_rng(3)

    def elem() -> tuple[jnp.ndarray, jnp.ndarray]:
        return (
            jnp.asarray(rng.standard_normal(4) + 1j * rng.standard_normal(4)),
            jnp.asarray(rng.standard_normal(4) + 1j * rng.standard_normal(4)),
        )

    x, y, z = elem(), elem(), elem()
    left = s5.s5_binary_operator(s5.s5_binary_operator(x, y), z)
    right = s5.s5_binary_operator(x, s5.s5_binary_operator(y, z))
    for a, b in zip(left, right):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-12, rtol=0)


def _s5_dense_oracle(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, dt: float, u: np.ndarray
) -> np.ndarray:
    """Independent dense reference for the S5 MIMO wiring (explicit per-step loop).

    Reproduces the diagonal-ZOH discretization and the $B$/$C$ contractions by hand,
    so a transpose/axis swap in ``s5_apply`` (e.g. ``u @ Bbar`` instead of
    ``u @ Bbar.T``) diverges from this oracle even though parallel still equals
    sequential.
    """
    A, B, C, u = (np.asarray(x) for x in (A, B, C, u))
    Abar = np.exp(A * dt)
    Bbar = ((Abar - 1.0) / A)[:, None] * B  # (P, H)
    h = np.zeros(A.shape[0], dtype=complex)
    ys = np.zeros((u.shape[0], B.shape[1]))
    for k in range(u.shape[0]):
        h = Abar * h + Bbar @ u[k]  # (P,)
        ys[k] = 2.0 * np.real(C @ h)  # (H,)
    return ys


def test_s5_apply_matches_dense_mimo_oracle() -> None:
    """S5 forward matches an explicit dense MIMO loop — pins the B/C wiring, not just parallel==sequential."""
    A, B, C = s5._demo_system(n_modes=8, h_dim=4)
    L = 120
    z = jnp.linspace(0.0, 1.0, L)
    u = jnp.stack([jnp.sin(2 * jnp.pi * (k + 1) * z) for k in range(B.shape[1])], axis=1)
    y = np.asarray(s5.s5_apply(A, B, C, 0.1, u, parallel=True))
    y_ref = _s5_dense_oracle(A, B, C, 0.1, u)
    np.testing.assert_allclose(y, y_ref, atol=1e-12, rtol=0)
