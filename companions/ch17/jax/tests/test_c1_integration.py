r"""Tests for Chapter 17 §17.2 — the C1 atlas cell (c1_integration).

Pins the integration signature (secular drift + oscillation band of the three integrators on a
harmonic-oscillator SSM mode) and the reduction to Chapter 6's reused RK4 path. The headline is
the NEW three-integrator comparison; the reduction confirms the reuse is faithful.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from companions.ch06.jax import symplectic_demo as sym  # noqa: E402
from companions.ch17.jax import c1_integration as c1  # noqa: E402

_DT = 0.1
_PERIODS = 200


def test_exact_exponential_conserves_energy() -> None:
    """The diagonal SSM's exact exponential conserves the imaginary-mode energy to machine precision."""
    n = c1._n_steps(_DT, _PERIODS)
    e = c1.exact_exponential_energy(_DT, n)
    assert abs(e[0] - 0.5) < 1e-13
    assert c1.energy_band(e) < 1e-10  # measured 2.8e-12
    assert abs(c1.secular_drift_per_period(e, _DT)) < 1e-12


def test_rk4_reduction_matches_ch06() -> None:
    """The reused RK4 path reproduces ch06's rk4_drift_per_period (endpoint metric) — faithful reuse."""
    n = c1._n_steps(_DT, _PERIODS)
    e_rk4 = c1.rk4_energy(_DT, n)
    mine = c1.endpoint_drift_per_period(e_rk4, _DT)
    ch06 = sym.rk4_drift_per_period(_DT, _PERIODS)
    assert abs(mine - ch06) < 1e-9  # measured 1.3e-11


def test_symplectic_kills_secular_drift() -> None:
    """Verlet's secular slope is far below RK4's; it trades a larger bounded oscillation for it."""
    cell = c1.atlas_cell(_DT, _PERIODS)
    assert abs(cell["verlet_drift"]) < 0.1 * abs(cell["rk4_drift"])  # measured ratio 1.3e-2
    assert cell["verlet_band"] > cell["rk4_band"]  # 1.25e-3 > 8.7e-5 (the symplectic trade-off)


def test_exact_exp_dominates_both_integrators() -> None:
    """On the conservative mode the exact exponential beats both integrators on band and drift."""
    cell = c1.atlas_cell(_DT, _PERIODS)
    assert cell["exact_exp_band"] < cell["rk4_band"]
    assert cell["exact_exp_band"] < cell["verlet_band"]
    assert abs(cell["exact_exp_drift"]) < abs(cell["rk4_drift"])


def test_rk4_secular_grows_with_step() -> None:
    """The atlas's second axis: RK4's secular drift accumulates faster as the step size grows."""
    drifts = [abs(c1.atlas_cell(dt, 100)["rk4_drift"]) for dt in (0.05, 0.1, 0.2, 0.4)]
    assert all(drifts[i] < drifts[i + 1] for i in range(len(drifts) - 1))


def test_atlas_cell_keys() -> None:
    cell = c1.atlas_cell(_DT, 50)
    for k in ("E0", "exact_exp_drift", "exact_exp_band", "verlet_drift", "verlet_band",
              "rk4_drift", "rk4_band"):
        assert k in cell
    assert abs(cell["E0"] - 0.5) < 1e-15


def test_validation() -> None:
    with pytest.raises(ValueError):
        c1._n_steps(0.0, 10)
    with pytest.raises(ValueError):
        c1._n_steps(0.1, 0)
    with pytest.raises(ValueError):
        c1.secular_drift_per_period(np.array([1.0]), 0.1)
    with pytest.raises(ValueError):
        c1.endpoint_drift_per_period(np.array([1.0]), 0.1)
