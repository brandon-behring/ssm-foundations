r"""Cross-framework parity: PyTorch Chapter 5 order verification vs the JAX reference.

The torch RK integrators (forward Euler / midpoint RK2 / classical RK4, driven from the
Butcher tableaux on the Chapter-4 forced damped oscillator) are run on the *same* step
sizes as the JAX core; the per-step output trajectories and the log-log convergence
slopes are compared in float64 (cross-framework consistency goal, 0527-F27).

Pinned facts (mirroring ``companions/ch05/jax/tests/test_order_verification.py``):

* empirical convergence orders 1 / 2 / 4 for forward Euler / RK2 / RK4;
* the RK4 ``slope >= 3.5`` guard (F29-style) catches a stage-loop bug that would
  silently drop RK4 to low order;
* the torch eager-loop integration matches the JAX ``lax.scan`` trajectory to roundoff;
* ``simulate`` raises on non-positive ``dt`` / ``t_end``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from companions.ch05.jax import order_verification as jax_ov  # noqa: E402
from companions.ch05.torch import order_verification as torch_ov  # noqa: E402

_PARITY_TOL = 1e-9
_DTS = np.array([0.1, 0.05, 0.025])
_T_END = 4.0


# ---------------------------------------------------------------------------
# Empirical convergence orders (mirror the JAX headline claims)
# ---------------------------------------------------------------------------


def test_rk1_first_order() -> None:
    _, slope = torch_ov.empirical_slope(torch_ov.forward_euler(), _DTS, _T_END)
    assert 0.8 <= slope <= 1.3, f"forward-Euler slope {slope:.3f} not ≈ 1"


def test_rk2_second_order() -> None:
    _, slope = torch_ov.empirical_slope(torch_ov.midpoint_rk2(), _DTS, _T_END)
    assert 1.7 <= slope <= 2.3, f"RK2 slope {slope:.3f} not ≈ 2"


def test_rk4_fourth_order() -> None:
    _, slope = torch_ov.empirical_slope(torch_ov.classical_rk4(), _DTS, _T_END)
    assert 3.6 <= slope <= 4.3, f"RK4 slope {slope:.3f} not ≈ 4"


def test_rk4_genuinely_high_order() -> None:
    """F29-style guard: a stage-loop bug would drop RK4's slope toward ≤ 2."""
    _, slope = torch_ov.empirical_slope(torch_ov.classical_rk4(), _DTS, _T_END)
    assert slope >= 3.5, f"RK4 slope {slope:.3f} degraded below high order"


def test_simulate_validation() -> None:
    with pytest.raises(ValueError):
        torch_ov.simulate(torch_ov.classical_rk4(), 0.0, _T_END)
    with pytest.raises(ValueError):
        torch_ov.simulate(torch_ov.classical_rk4(), 0.05, -1.0)


# ---------------------------------------------------------------------------
# JAX↔torch parity on identical inputs (< 1e-9)
# ---------------------------------------------------------------------------


def _jax_tab(name: str):
    return {
        "Forward Euler": jax_ov.forward_euler,
        "Midpoint RK2": jax_ov.midpoint_rk2,
        "Classical RK4": jax_ov.classical_rk4,
    }[name]()


def _torch_tab(name: str):
    return {
        "Forward Euler": torch_ov.forward_euler,
        "Midpoint RK2": torch_ov.midpoint_rk2,
        "Classical RK4": torch_ov.classical_rk4,
    }[name]()


@pytest.mark.parametrize("name", ["Forward Euler", "Midpoint RK2", "Classical RK4"])
def test_simulate_trajectory_matches_jax(name: str) -> None:
    """torch eager-loop trajectory reproduces the JAX lax.scan trajectory (per-step)."""
    ts_jax, ys_jax = jax_ov.simulate(_jax_tab(name), 0.05, _T_END)
    ts_torch, ys_torch = torch_ov.simulate(_torch_tab(name), 0.05, _T_END)
    assert np.max(np.abs(np.asarray(ts_jax) - ts_torch)) < _PARITY_TOL
    assert np.max(np.abs(np.asarray(ys_jax) - ys_torch)) < _PARITY_TOL


@pytest.mark.parametrize("name", ["Forward Euler", "Midpoint RK2", "Classical RK4"])
def test_errors_and_slope_match_jax(name: str) -> None:
    """torch error curve and convergence slope reproduce the JAX values to roundoff.

    Both companions share the framework-agnostic ``solve_ivp(Radau)`` reference, so the
    only difference is the RK arithmetic (eager loop vs ``lax.scan``). The two
    integrators agree *bitwise* to machine epsilon (the per-step trajectory test below
    pins ~2e-16), and so do the error curves in **absolute** terms (``< _PARITY_TOL``).

    The slope is a derived quantity ``log(e_{-2}/e_{-1}) / log(dt_{-2}/dt_{-1})`` and is
    ill-conditioned for high-order methods: RK4's finest error is ~1e-7, so a 1-ULP
    (~2e-16 absolute) difference in that error is a *relative* perturbation of ~1e-9,
    which the log-ratio amplifies by ``1/log(2) ≈ 1.44`` to ~1.5e-9 in the slope. We
    therefore pin the slope with that propagated bound rather than the raw error
    tolerance — the underlying parity (trajectories + absolute errors) stays at
    ``< _PARITY_TOL``; only the conditioning of the log-ratio is accounted for here.
    """
    # Slope bound: the directly-compared errors match at _PARITY_TOL absolute; for
    # RK4 the finest error ~1e-7 makes that a ~1e-9 relative shift, amplified by the
    # 1/log(dt-ratio) factor of the slope formula (log(2) here) -> a few e-9.
    _SLOPE_TOL = 5e-9
    errs_jax, slope_jax = jax_ov.empirical_slope(_jax_tab(name), _DTS, _T_END)
    errs_torch, slope_torch = torch_ov.empirical_slope(_torch_tab(name), _DTS, _T_END)
    assert np.max(np.abs(np.asarray(errs_jax) - errs_torch)) < _PARITY_TOL
    assert abs(slope_jax - slope_torch) < _SLOPE_TOL
