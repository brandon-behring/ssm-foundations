r"""Cross-framework parity: PyTorch Chapter 6 symplectic demo vs the JAX reference (§6.5).

The torch integrators are fed the *same* inputs as the JAX integrators and the
trajectories are compared in float64, so the two frameworks stay in lock-step
(the cross-framework consistency goal Chapter 7 introduced). The headline §6.5
claims are also re-pinned on the torch side: the symplectic Störmer-Verlet
integrator keeps the harmonic-oscillator energy in a bounded band while RK4
drifts ~linearly with the horizon, and the Δ=0.05 RK4 drift reproduces Exercise
6.3's ~-1.4e-8 per period.
"""

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

torch = pytest.importorskip("torch")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402, F401

from companions.ch06.jax import symplectic_demo as jax_sym  # noqa: E402
from companions.ch06.torch import symplectic_demo as torch_sym  # noqa: E402

_PARITY_TOL = 1e-9


# ---------------------------------------------------------------------------
# §6.5 — trajectory parity (identical inputs -> identical (ts, qs, ps))
# ---------------------------------------------------------------------------


def test_harmonic_verlet_trajectory_matches_jax() -> None:
    """torch Verlet reproduces the JAX harmonic-oscillator trajectory (§6.5 parity)."""
    ts_j, qs_j, ps_j = jax_sym.simulate(
        jax_sym.verlet_step, jax_sym.harmonic_T_grad, jax_sym.harmonic_V_grad, 1.0, 0.0, 0.3, 600
    )
    ts_t, qs_t, ps_t = torch_sym.simulate(
        torch_sym.verlet_step,
        torch_sym.harmonic_T_grad,
        torch_sym.harmonic_V_grad,
        1.0,
        0.0,
        0.3,
        600,
    )
    assert np.max(np.abs(np.asarray(ts_j) - ts_t)) < _PARITY_TOL
    assert np.max(np.abs(np.asarray(qs_j) - qs_t)) < _PARITY_TOL
    assert np.max(np.abs(np.asarray(ps_j) - ps_t)) < _PARITY_TOL


def test_harmonic_rk4_trajectory_matches_jax() -> None:
    """torch RK4 reproduces the JAX harmonic-oscillator trajectory (§6.5 parity)."""
    _, qs_j, ps_j = jax_sym.simulate(
        jax_sym.rk4_step_hamilton,
        jax_sym.harmonic_T_grad,
        jax_sym.harmonic_V_grad,
        1.0,
        0.0,
        0.3,
        600,
    )
    _, qs_t, ps_t = torch_sym.simulate(
        torch_sym.rk4_step_hamilton,
        torch_sym.harmonic_T_grad,
        torch_sym.harmonic_V_grad,
        1.0,
        0.0,
        0.3,
        600,
    )
    assert np.max(np.abs(np.asarray(qs_j) - qs_t)) < _PARITY_TOL
    assert np.max(np.abs(np.asarray(ps_j) - ps_t)) < _PARITY_TOL


def test_pendulum_verlet_trajectory_matches_jax() -> None:
    """torch Verlet reproduces the JAX nonlinear-pendulum trajectory (§6.5 parity)."""
    _, qs_j, ps_j = jax_sym.simulate(
        jax_sym.verlet_step, jax_sym.pendulum_T_grad, jax_sym.pendulum_V_grad, 1.0, 0.0, 0.05, 5000
    )
    _, qs_t, ps_t = torch_sym.simulate(
        torch_sym.verlet_step,
        torch_sym.pendulum_T_grad,
        torch_sym.pendulum_V_grad,
        1.0,
        0.0,
        0.05,
        5000,
    )
    assert np.max(np.abs(np.asarray(qs_j) - qs_t)) < _PARITY_TOL
    assert np.max(np.abs(np.asarray(ps_j) - ps_t)) < _PARITY_TOL


def test_rk4_drift_per_period_matches_jax() -> None:
    """torch and JAX agree on the Exercise 6.3 drift number (and both ≈ -1.4e-8)."""
    drift_j = jax_sym.rk4_drift_per_period(dt=0.05, periods=100)
    drift_t = torch_sym.rk4_drift_per_period(dt=0.05, periods=100)
    assert abs(drift_j - drift_t) < _PARITY_TOL
    # Mirror the JAX test_rk4_drift_at_dt005 headline claim on the torch side.
    assert -2e-8 < drift_t < -1e-8, f"torch RK4 drift {drift_t:.3e} not ≈ -1.4e-8 (Exercise 6.3)"


# ---------------------------------------------------------------------------
# §6.5 — the symplectic signature, re-pinned on the torch side
# ---------------------------------------------------------------------------


def test_verlet_bounded_rk4_drifts_torch() -> None:
    """Mirror of the JAX headline: Verlet's energy band is horizon-independent
    while RK4's energy error grows ~linearly with the integration horizon."""
    dt = 0.3

    def run(stepper, periods):
        n = int(round(periods * 2 * np.pi / dt))
        _, q, p = torch_sym.simulate(
            stepper, torch_sym.harmonic_T_grad, torch_sym.harmonic_V_grad, 1.0, 0.0, dt, n
        )
        energy = 0.5 * (p**2 + q**2)
        return float(np.ptp(energy)), float(abs(energy[-1] - 0.5))

    band_100, _ = run(torch_sym.verlet_step, 100)
    band_400, _ = run(torch_sym.verlet_step, 400)
    _, drift_100 = run(torch_sym.rk4_step_hamilton, 100)
    _, drift_400 = run(torch_sym.rk4_step_hamilton, 400)

    # Verlet: bounded band, independent of horizon.
    assert (
        band_400 < 1.2 * band_100 and band_400 < 0.05
    ), f"Verlet band grew: {band_100:.3e}->{band_400:.3e}"
    # RK4: drift grows with horizon (4x periods -> ~4x error).
    assert drift_400 > 3.0 * drift_100, f"RK4 drift did not grow: {drift_100:.3e}->{drift_400:.3e}"


def test_pendulum_verlet_conserves_energy_torch() -> None:
    """Mirror of the JAX headline: Verlet keeps the pendulum's H in a tight band."""
    _, qs, ps = torch_sym.simulate(
        torch_sym.verlet_step,
        torch_sym.pendulum_T_grad,
        torch_sym.pendulum_V_grad,
        1.0,
        0.0,
        0.05,
        5000,
    )
    H = 0.5 * ps**2 - np.cos(qs)
    assert np.ptp(H) < 1e-2, f"Verlet pendulum energy band {np.ptp(H):.2e} too large"


def test_simulate_validation_torch() -> None:
    """Input guards mirror the JAX-side validation contract."""
    with pytest.raises(ValueError):
        torch_sym.simulate(
            torch_sym.verlet_step,
            torch_sym.harmonic_T_grad,
            torch_sym.harmonic_V_grad,
            1.0,
            0.0,
            0.0,
            10,
        )
    with pytest.raises(ValueError):
        torch_sym.simulate(
            torch_sym.verlet_step,
            torch_sym.harmonic_T_grad,
            torch_sym.harmonic_V_grad,
            1.0,
            0.0,
            0.1,
            -1,
        )
