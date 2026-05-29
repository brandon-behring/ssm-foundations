"""Assertion-based tests for the Chapter 1 damped-oscillator companion (0527-F26).

Pins the §1.3 claims: the JAX ``lax.scan`` RK4 trajectory matches both an
independent NumPy RK4 loop and the high-accuracy ``solve_ivp(Radau)`` reference,
and the energy decays exponentially at the rate $c$ (the load-bearing physical
claim — its log-slope is the regression guard).
"""

from __future__ import annotations

import numpy as np
import pytest

from companions.ch01.jax import damped_oscillator as do


def _rk4_numpy(k: float, c: float, dt: float, n_steps: int, h0: tuple[float, float]) -> np.ndarray:
    """Independent NumPy RK4 loop oracle for the scan trajectory."""
    def f(h: np.ndarray) -> np.ndarray:
        q, qdot = h
        return np.array([qdot, -k * q - c * qdot])

    h = np.asarray(h0, dtype=float)
    hs = [h.copy()]
    for _ in range(n_steps):
        k1 = f(h)
        k2 = f(h + 0.5 * dt * k1)
        k3 = f(h + 0.5 * dt * k2)
        k4 = f(h + dt * k3)
        h = h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        hs.append(h.copy())
    return np.array(hs)


def test_scan_matches_numpy_rk4() -> None:
    """series of scan states equals an independent NumPy RK4 accumulation."""
    _, hs, _ = do.simulate_scan(k=4.0, c=0.2, t_max=10.0, n_steps=500)
    ref = _rk4_numpy(4.0, 0.2, 10.0 / 500, 500, (1.0, 0.0))
    np.testing.assert_allclose(hs, ref, atol=1e-10)


def test_rk4_matches_radau_reference() -> None:
    """JAX RK4 energy tracks the scipy Radau reference on the shared grid."""
    t, _, energy = do.simulate_scan(n_steps=2000)
    t_ref, _, energy_ref = do.simulate_reference(n_points=2001)
    np.testing.assert_allclose(t, t_ref, atol=1e-12)  # identical sample grid
    assert np.max(np.abs(energy - energy_ref)) < 1e-3, (
        f"max energy gap {np.max(np.abs(energy - energy_ref)):.2e} exceeds 1e-3"
    )


def test_energy_decays_at_rate_c() -> None:
    """Regression guard: log-energy slope ≈ -c (energy ~ amplitude² ~ e^{-c t})."""
    c = 0.2
    t, _, energy = do.simulate_scan(c=c, n_steps=2000)
    slope = float(np.polyfit(t, np.log(energy), 1)[0])
    assert -0.25 < slope < -0.15, f"energy log-slope {slope:.3f} not ≈ -c = -{c}"


def test_simulate_scan_validation() -> None:
    with pytest.raises(ValueError):
        do.simulate_scan(k=-1.0)
    with pytest.raises(ValueError):
        do.simulate_scan(t_max=-1.0)
    with pytest.raises(ValueError):
        do.simulate_scan(n_steps=0)
