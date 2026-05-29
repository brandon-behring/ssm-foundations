"""Assertion-based tests for the Chapter 6 stiff-demo companion (0527-F26).

Pins §6.1: the autodiff (jacfwd) Jacobian matches the analytic one, the
fixed-iteration backward-Euler Newton actually converges (residual guard), the
implicit method stays bounded at a coarse dt where explicit RK4 blows up, and
the scan trajectory matches an independent NumPy Newton loop.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from companions.ch06.jax import stiff_demo as sd

_MU = sd._MU


def _vdp_jac_analytic(h: np.ndarray) -> np.ndarray:
    """Hand-coded van der Pol Jacobian — the oracle jacfwd must reproduce."""
    q, p = h
    return np.array([[0.0, 1.0], [-2.0 * _MU * q * p - 1.0, _MU * (1.0 - q * q)]])


def _be_numpy(h0: np.ndarray, dt: float, t_end: float, tol: float = 1e-10, max_iter: int = 60) -> np.ndarray:
    """Independent NumPy damped-Newton-to-tolerance backward-Euler oracle.

    Mirrors the JAX algorithm: damped Newton with an exact line search over the
    same step fractions, iterated until the residual falls below ``tol``.
    """
    def f(h):
        q, p = h
        return np.array([p, _MU * (1.0 - q * q) * p - q])

    alphas = 0.5 ** np.arange(16)
    n_steps = int(round(t_end / dt)) + 1
    hs = np.zeros((n_steps, 2))
    hs[0] = h0
    Id = np.eye(2)
    for k in range(n_steps - 1):
        h = hs[k]
        x = h + dt * f(h)
        for _ in range(max_iter):
            if np.linalg.norm(x - h - dt * f(x)) <= tol:
                break
            delta = np.linalg.solve(Id - dt * _vdp_jac_analytic(x), x - h - dt * f(x))
            cands = x[None, :] - alphas[:, None] * delta[None, :]
            x = cands[np.argmin([np.linalg.norm(c - h - dt * f(c)) for c in cands])]
        hs[k + 1] = x
    return hs


def test_jacfwd_matches_handcoded() -> None:
    for h in ([2.0, 0.0], [0.5, -1.3], [-1.0, 2.0]):
        got = np.asarray(sd._vdp_jac(jnp.array(h)))
        np.testing.assert_allclose(got, _vdp_jac_analytic(np.array(h)), atol=1e-12)


def test_backward_euler_residual() -> None:
    """Damped Newton converges along the WHOLE trajectory, not just step 1 — the
    guard that catches coarse-dt under-convergence during the van der Pol jumps."""
    for dt in (0.05, 0.2):
        _, hs = sd.simulate_be(np.array([2.0, 0.0]), dt, 50.0)
        max_res = max(
            float(jnp.linalg.norm(jnp.asarray(hs[k + 1]) - jnp.asarray(hs[k]) - dt * sd.vdp_rhs(jnp.asarray(hs[k + 1]))))
            for k in range(len(hs) - 1)
        )
        assert max_res < 1e-8, f"BE max residual {max_res:.2e} at dt={dt} — Newton under-converged"


def test_be_stable_at_coarse_dt() -> None:
    # BE stays on the bounded limit cycle at a coarse dt where RK4 diverges. The
    # position q is bounded (~2); the velocity p legitimately spikes to ~μ during
    # the fast relaxation jumps, so only q is range-checked.
    _, hs = sd.simulate_be(np.array([2.0, 0.0]), 0.2, 50.0)
    assert np.all(np.isfinite(hs)), "backward Euler must not diverge at coarse dt"
    assert np.max(np.abs(hs[:, 0])) < 3.0, "BE position should stay on the bounded cycle"


def test_rk4_blows_up_at_coarse_dt() -> None:
    """Regression guard for the figure's whole point: explicit RK4 diverges at dt=0.2."""
    _, hs = sd.simulate_rk4(np.array([2.0, 0.0]), 0.2, 50.0)
    assert np.any(np.isnan(hs)), "RK4 should diverge (NaN-masked) at the coarse dt"


def test_scan_matches_numpy_newton() -> None:
    _, hs = sd.simulate_be(np.array([2.0, 0.0]), 0.05, 5.0)
    np.testing.assert_allclose(hs, _be_numpy(np.array([2.0, 0.0]), 0.05, 5.0), atol=1e-8)


def test_simulate_validation() -> None:
    with pytest.raises(ValueError):
        sd.simulate_rk4(np.array([2.0, 0.0]), 0.0, 10.0)
    with pytest.raises(ValueError):
        sd.simulate_be(np.array([2.0, 0.0]), 0.1, -1.0)
