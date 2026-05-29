"""Assertion-based tests for the Chapter 6 symplectic-demo companion (0527-F26).

Pins §6.5 and the Exercise 6.3 number (0527-F34): the symplectic Störmer-Verlet
integrator keeps the harmonic-oscillator energy in a bounded band while RK4
drifts, the RK4 drift rate at Δ=0.05 reproduces the exercise's ~1.4e-8 per
period, and the ``lax.scan`` trajectory matches an independent NumPy loop.
"""

from __future__ import annotations

import numpy as np

from companions.ch06.jax import symplectic_demo as syd


def _simulate_numpy(stepper, T_grad, V_grad, q0, p0, dt, n_steps):
    """Independent NumPy time-stepping loop oracle for the scan."""
    qs = np.zeros(n_steps + 1)
    ps = np.zeros(n_steps + 1)
    qs[0], ps[0] = q0, p0
    for k in range(n_steps):
        qs[k + 1], ps[k + 1] = stepper(T_grad, V_grad, qs[k], ps[k], dt)
    return qs, ps


def test_rk4_drift_at_dt005() -> None:
    """F34 guard: RK4 harmonic-oscillator drift at Δ=0.05 ≈ -1.4e-8 per period."""
    drift = syd.rk4_drift_per_period(dt=0.05, periods=100)
    assert -2e-8 < drift < -1e-8, f"RK4 drift {drift:.3e} not ≈ -1.4e-8 (Exercise 6.3)"


def test_verlet_bounded_rk4_drifts() -> None:
    """The symplectic signature: Verlet's energy band is horizon-independent,
    while RK4's energy error grows ~linearly with the integration horizon."""
    dt = 0.3

    def run(stepper, periods):
        n = int(round(periods * 2 * np.pi / dt))
        _, q, p = syd.simulate(stepper, syd.harmonic_T_grad, syd.harmonic_V_grad, 1.0, 0.0, dt, n)
        energy = 0.5 * (p**2 + q**2)
        return float(np.ptp(energy)), float(abs(energy[-1] - 0.5))

    band_100, _ = run(syd.verlet_step, 100)
    band_400, _ = run(syd.verlet_step, 400)
    _, drift_100 = run(syd.rk4_step_hamilton, 100)
    _, drift_400 = run(syd.rk4_step_hamilton, 400)

    # Verlet: bounded band, independent of horizon.
    assert band_400 < 1.2 * band_100 and band_400 < 0.05, f"Verlet band grew: {band_100:.3e}->{band_400:.3e}"
    # RK4: drift grows with horizon (4x periods -> ~4x error).
    assert drift_400 > 3.0 * drift_100, f"RK4 drift did not grow: {drift_100:.3e}->{drift_400:.3e}"


def test_scan_matches_numpy_loop() -> None:
    n_steps = 500
    _, qs, ps = syd.simulate(syd.verlet_step, syd.harmonic_T_grad, syd.harmonic_V_grad, 1.0, 0.0, 0.1, n_steps)
    qref, pref = _simulate_numpy(syd.verlet_step, syd.harmonic_T_grad, syd.harmonic_V_grad, 1.0, 0.0, 0.1, n_steps)
    np.testing.assert_allclose(qs, qref, atol=1e-10)
    np.testing.assert_allclose(ps, pref, atol=1e-10)


def test_pendulum_verlet_conserves_energy() -> None:
    """Verlet keeps the pendulum's H within a tight band over a long trapped orbit."""
    n_steps = 5000
    _, qs, ps = syd.simulate(syd.verlet_step, syd.pendulum_T_grad, syd.pendulum_V_grad, 1.0, 0.0, 0.05, n_steps)
    H = 0.5 * ps**2 - np.cos(qs)
    assert np.ptp(H) < 1e-2, f"Verlet pendulum energy band {np.ptp(H):.2e} too large"
