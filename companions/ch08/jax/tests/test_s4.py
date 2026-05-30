r"""Assertion tests for the Chapter 8 S4 core (``companions/ch08/jax/s4_core.py``).

Pins the load-bearing numerical claims of §8.1–§8.3 so they cannot silently rot
(mirrors ``companions/ch07/jax/tests/test_hippo.py``):

* the HiPPO-LegS closed form + spectrum carried over from Chapter 7;
* ZOH (and bilinear) discretization is stable — $|\lambda(\bar A)| < 1$ (§8.1);
* the naive kernel equals an independent NumPy power-iteration oracle (§8.3);
* **the conv<->recurrence duality** — ``ssm_recurrent`` ≡ ``ssm_convolutional`` for
  zero initial state (§8.3, the chapter's central claim);
* the FFT path zero-pads to $2L$ and so matches a direct linear convolution
  (the regression behind Exercise 8.3).

Run: ``PYTHONPATH=. .venv/bin/pytest companions/ch08/jax/tests/test_s4.py -q``
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from companions.ch08.jax import s4_core as s4

# ---------------------------------------------------------------------------
# helpers — a fixed, reproducible SISO system for the duality / kernel checks
# ---------------------------------------------------------------------------


def _siso_system(n: int = 16, dt: float = 0.1) -> tuple[jnp.ndarray, ...]:
    """HiPPO-LegS system discretized by ZOH, with a fixed random readout C."""
    A, B = s4.make_hippo_legs(n)
    Ab, Bb = s4.discretize_zoh(A, B, dt)
    rng = np.random.default_rng(0)
    C = jnp.asarray(rng.standard_normal((1, n)))
    D = jnp.asarray(0.3)
    return Ab, Bb, C, D


def _signal(L: int) -> jnp.ndarray:
    z = jnp.linspace(0.0, 1.0, L)
    return jnp.sin(2.0 * jnp.pi * 3.0 * z) + 0.5 * jnp.cos(2.0 * jnp.pi * 7.0 * z)


# ---------------------------------------------------------------------------
# §8.2 — structured state matrix (carried over from Chapter 7)
# ---------------------------------------------------------------------------


def test_hippo_shapes_and_lower_triangular() -> None:
    for n in (1, 4, 8, 16):
        A, B = s4.make_hippo_legs(n)
        assert A.shape == (n, n) and B.shape == (n, 1)
    A16 = np.asarray(s4.make_hippo_legs(16)[0])
    assert np.max(np.abs(np.triu(A16, k=1))) == 0.0, "A must be lower-triangular"


def test_hippo_spectrum_is_minus_one_to_minus_n() -> None:
    """Eigenvalues are exactly -1,...,-N (real, stable) — the Chapter 7 result."""
    for n in (4, 8, 16, 32):
        eigs = np.linalg.eigvals(np.asarray(s4.make_hippo_legs(n)[0]))
        assert np.max(np.abs(eigs.imag)) < 1e-9, "eigenvalues should be real"
        np.testing.assert_allclose(np.sort(eigs.real), np.sort(-(np.arange(n) + 1.0)), atol=1e-8)


def test_make_hippo_legs_validation() -> None:
    with pytest.raises(ValueError):
        s4.make_hippo_legs(0)


# ---------------------------------------------------------------------------
# §8.1 — discretization stability
# ---------------------------------------------------------------------------


def test_zoh_is_stable() -> None:
    r"""ZOH on the LTI-stable HiPPO A gives $|\lambda(\bar A)| < 1$ for every mode."""
    A, B = s4.make_hippo_legs(32)
    for dt in (0.01, 0.1, 1.0):
        Ab, _ = s4.discretize_zoh(A, B, dt)
        rho = np.max(np.abs(np.linalg.eigvals(np.asarray(Ab))))
        assert rho < 1.0, f"ZOH unstable at dt={dt}: rho={rho}"


def test_bilinear_is_stable() -> None:
    """Bilinear (Tustin) maps the stable HiPPO A to a stable discrete system."""
    A, B = s4.make_hippo_legs(16)
    for dt in (0.01, 0.1, 1.0):
        Ab, _ = s4.discretize_bilinear(A, B, dt)
        rho = np.max(np.abs(np.linalg.eigvals(np.asarray(Ab))))
        assert rho < 1.0, f"bilinear unstable at dt={dt}: rho={rho}"


# ---------------------------------------------------------------------------
# §8.3 — kernel oracle + the conv<->recurrence duality
# ---------------------------------------------------------------------------


def test_kernel_matches_power_iteration_oracle() -> None:
    r"""``ssm_kernel_naive`` equals an independent NumPy loop $K_k = C\bar A^k\bar B$."""
    Ab, Bb, C, _ = _siso_system()
    L = 64
    K = np.asarray(s4.ssm_kernel_naive(Ab, Bb, C, L))
    Ab_np, Bb_np, C_np = np.asarray(Ab), np.asarray(Bb), np.asarray(C)
    ref = np.array(
        [float((C_np @ np.linalg.matrix_power(Ab_np, k) @ Bb_np).squeeze()) for k in range(L)]
    )
    np.testing.assert_allclose(K, ref, atol=1e-9)


def test_conv_recurrence_duality() -> None:
    """THE load-bearing test (§8.3): recurrent scan ≡ FFT convolution for h0 = 0."""
    Ab, Bb, C, D = _siso_system()
    L = 128
    u = _signal(L)
    K = s4.ssm_kernel_naive(Ab, Bb, C, L)
    y_rec = s4.ssm_recurrent(Ab, Bb, C, D, u)
    y_conv = s4.ssm_convolutional(K, D, u)
    residual = float(jnp.max(jnp.abs(y_rec - y_conv)))
    assert residual < 1e-12, f"duality broken: max|y_rec - y_conv| = {residual:.3e}"


def test_fft_pad_matches_direct_convolution() -> None:
    """The 2L-padded FFT path equals a direct causal linear convolution (Ex 8.3)."""
    Ab, Bb, C, _ = _siso_system()
    L = 96
    u = _signal(L)
    K = np.asarray(s4.ssm_kernel_naive(Ab, Bb, C, L))
    y_fft = np.asarray(s4.causal_conv_fft(u, jnp.asarray(K)))
    y_direct = np.convolve(np.asarray(u), K)[:L]  # full linear conv, truncated to L
    np.testing.assert_allclose(y_fft, y_direct, atol=1e-9)
