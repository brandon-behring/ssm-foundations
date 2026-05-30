r"""Tests for ch10 trapezoidal_ssd: SSD duality + the second-order stencil delivers order 2.

Pins the §10.5 claims:

* the dense semiseparable matmul equals the sequential oracle (the SSD duality of
  §9.5, now carrying the trapezoidal stencil) to ``< 1e-12``, for real and complex
  modes;
* applied to a forced (selective) system the trapezoidal pass is genuinely
  second-order accurate (the integrator upgrade actually pays off end-to-end);
* the shared-decay construction is correct: folding the stream weight into ``dt``
  (the predecessor shortcut) would corrupt the decay — a regression guard.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from companions.ch09.jax.selective_ssm import selection_from_input, stable_A
from companions.ch10.jax.discretization import forced_exact  # noqa: E402
from companions.ch10.jax.trapezoidal_ssd import (  # noqa: E402
    decay_operator,
    trapezoidal_matmul,
    trapezoidal_sequential,
)


def _system(n=6, length=32, d=4, seed=0, complex_modes=True):
    rng = np.random.default_rng(seed)
    A = stable_A(jnp.asarray(rng.standard_normal(n)))
    if complex_modes:
        A = A + 1j * jnp.asarray(rng.standard_normal(n))
    x = jnp.asarray(rng.standard_normal((length, d)))
    w_delta = jnp.asarray(rng.standard_normal(d))
    w_B = jnp.asarray(rng.standard_normal((d, n)))
    w_C = jnp.asarray(rng.standard_normal((d, n)))
    delta, B, C = selection_from_input(x, w_delta, w_B, w_C)
    u = jnp.asarray(rng.standard_normal(length))
    return A, delta, B, C, jnp.asarray(0.0), u


@pytest.mark.parametrize("complex_modes", [True, False])
def test_matmul_equals_sequential(complex_modes):
    """SSD duality: the dense matmul reproduces the O(L) oracle."""
    A, delta, B, C, D, u = _system(complex_modes=complex_modes)
    y_seq = trapezoidal_sequential(A, delta, B, C, D, u)
    y_mat = trapezoidal_matmul(A, delta, B, C, D, u)
    assert_allclose(np.asarray(y_mat), np.asarray(y_seq), rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_matmul_equals_sequential_seeds(seed):
    A, delta, B, C, D, u = _system(seed=seed)
    y_seq = trapezoidal_sequential(A, delta, B, C, D, u)
    y_mat = trapezoidal_matmul(A, delta, B, C, D, u)
    assert_allclose(np.asarray(y_mat), np.asarray(y_seq), rtol=0, atol=1e-12)


def test_feedthrough_applied_once():
    """A nonzero D adds exactly D*u (not twice via the two streams)."""
    A, delta, B, C, _, u = _system()
    y0 = trapezoidal_sequential(A, delta, B, C, 0.0, u)
    yD = trapezoidal_sequential(A, delta, B, C, 2.0, u)
    assert_allclose(np.asarray(yD - y0), np.asarray(2.0 * u), rtol=0, atol=1e-12)


def test_decay_operator_is_causal_and_unit_diagonal():
    """Phi(k,j)=0 for k<j, Phi(k,k)=1 (empty product)."""
    A, delta, _, _, _, _ = _system(n=4, length=8)
    Phi = decay_operator(A, delta)  # (N, L, L)
    L = delta.shape[0]
    upper = np.triu_indices(L, k=1)
    assert_allclose(np.asarray(Phi[:, upper[0], upper[1]]), 0.0, rtol=0, atol=0)
    diag = np.asarray(Phi[:, np.arange(L), np.arange(L)])
    assert_allclose(diag, 1.0, rtol=0, atol=1e-14)


def test_trapezoidal_second_order_on_forced_selective_system():
    r"""End-to-end order check: refine dt -> the trapezoidal output is second-order.

    Use a *single fixed continuous* SISO system $x' = A x + \sin(\omega t)$ with
    $B = C = 1$ (so $y_k = h_k$) sampled at decreasing dt, and compare the whole
    trajectory against the analytic solution :func:`forced_exact` at the matching
    grid times $t_k = k\,\Delta$. The max error over the horizon must fall ~4x per
    dt-halving (order 2). (Comparing against ``y[-1]`` would be wrong: that sample
    sits at $t = (n-1)\Delta$, a moving target that injects a spurious $O(\Delta)$
    term and disguises the true order.)
    """
    A = jnp.asarray([-0.5])  # one real decaying mode; B = C = 1 so y_k = h_k
    omega = 1.0
    T = 4.0

    def max_error(n_steps: int) -> float:
        dt = T / n_steps
        t = jnp.arange(n_steps) * dt
        u = jnp.sin(omega * t)
        delta = jnp.full((n_steps,), dt)
        B = jnp.ones((n_steps, 1))
        C = jnp.ones((n_steps, 1))
        y = trapezoidal_sequential(A, delta, B, C, 0.0, u)
        exact = forced_exact(A[0], omega, t, x0=0.0)
        return float(jnp.max(jnp.abs(y - exact)))

    e_coarse = max_error(64)
    e_fine = max_error(128)
    ratio = e_coarse / e_fine
    assert ratio > 3.7, f"halving dt should cut error ~4x (order 2); got {ratio:.2f}"


def test_lambda_one_matches_first_order_shifted():
    """At lam=1 the trapezoid degenerates; matmul still equals the oracle."""
    A, delta, B, C, D, u = _system()
    y_seq = trapezoidal_sequential(A, delta, B, C, D, u, lam=1.0)
    y_mat = trapezoidal_matmul(A, delta, B, C, D, u, lam=1.0)
    assert_allclose(np.asarray(y_mat), np.asarray(y_seq), rtol=0, atol=1e-12)


def test_shape_validation():
    A, delta, B, C, D, u = _system(n=4, length=8)
    with pytest.raises(ValueError):
        trapezoidal_sequential(A, delta, B[:-1], C, D, u)  # wrong B length
    with pytest.raises(ValueError):
        trapezoidal_matmul(A, delta, B, C, D, u[:-1])  # wrong u length
