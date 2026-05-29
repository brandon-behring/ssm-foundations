"""Assertion-based tests for the Chapter 4 JAX discretization companions.

Pins the pedagogical numerical claims of Ch 4 so they cannot silently rot
(audit 0527-F26). In particular ``test_exp_trap_second_order`` is the direct
regression guard for the F29-class augmented-matrix bug: a wrong ``dt`` entry in
the exp-trapezoidal augmented matrix silently degrades the scheme from second-
to first-order, which this test would catch (slope would fall toward 1).

Run: ``PYTHONPATH=. .venv/bin/pytest companions/ch04/jax/tests -q``
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from companions.ch04.jax import discretization_comparison as dc
from companions.ch04.jax import exp_trapezoidal as et

_DTS = np.array([0.1, 0.05, 0.025])


def _slope(errs: np.ndarray, dts: np.ndarray) -> float:
    """Empirical convergence order from the two finest step sizes."""
    return float(np.log(errs[-2] / errs[-1]) / np.log(dts[-2] / dts[-1]))


def _et_max_err(sim_fn, dt: float, t_end: float = 4.0) -> float:
    ts, ys = sim_fn(dt, t_end)
    return float(np.max(np.abs(ys - et.continuous_reference(t_end, ts))))


def test_zoh_first_order() -> None:
    errs = np.array(
        [dc.measure_max_error(dc.discretize_zoh, dc.step_hold, float(dt)) for dt in _DTS]
    )
    assert 0.8 <= _slope(errs, _DTS) <= 1.3, f"ZOH slope {_slope(errs, _DTS):.3f} not ≈ 1"


def test_bilinear_second_order() -> None:
    errs = np.array(
        [dc.measure_max_error(dc.discretize_bilinear, dc.step_midpoint, float(dt)) for dt in _DTS]
    )
    assert 1.7 <= _slope(errs, _DTS) <= 2.3, f"bilinear slope {_slope(errs, _DTS):.3f} not ≈ 2"


def test_zoh_Ad_is_matrix_exponential() -> None:
    """ZOH's discrete dynamics matrix is exactly e^{A Δ} (autonomous-exactness)."""
    dt = 0.1
    Ad, _ = dc.discretize_zoh(dc._A_MAT, dc._B_VEC, dt)
    np.testing.assert_allclose(np.asarray(Ad), expm(np.asarray(dc._A_MAT) * dt), atol=1e-10)


def test_exp_trap_second_order() -> None:
    """F29 regression guard: a wrong augmented-matrix entry makes this slope ~1."""
    errs = np.array([_et_max_err(et.simulate_exp_trap, float(dt)) for dt in _DTS])
    assert (
        _slope(errs, _DTS) >= 1.6
    ), f"exp-trap slope {_slope(errs, _DTS):.3f} — degraded below 2nd order?"


def test_exp_trap_beats_zoh() -> None:
    err_exp = _et_max_err(et.simulate_exp_trap, 0.05)
    err_zoh = _et_max_err(et.simulate_zoh, 0.05)
    assert err_exp < err_zoh, f"exp-trap err {err_exp:.2e} should beat ZOH err {err_zoh:.2e}"


def test_scan_matches_naive_loop() -> None:
    """The lax.scan refactor of dc.simulate matches an independent Python loop."""
    dt, t_end = 0.05, 2.0
    ts, ys = dc.simulate(dc.discretize_zoh, dc.step_hold, dt, t_end)
    Ad, Bd = dc.discretize_zoh(dc._A_MAT, dc._B_VEC, dt)
    Ad, Bd, C = np.asarray(Ad), np.asarray(Bd), np.asarray(dc._C_ROW)
    h = np.zeros(2)
    ref = np.zeros(len(ts))
    u = np.sin(2.0 * ts)
    for k in range(len(ts) - 1):
        ref[k] = C @ h
        h = Ad @ h + Bd * u[k]
    ref[-1] = C @ h
    np.testing.assert_allclose(ys, ref, atol=1e-10)
