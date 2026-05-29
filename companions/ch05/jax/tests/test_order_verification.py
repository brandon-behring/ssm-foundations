"""Assertion-based tests for the Chapter 5 order-verification companion (0527-F26).

Pins the §5 claims: forward Euler / midpoint RK2 / classical RK4 exhibit
empirical convergence orders 1 / 2 / 4, and the ``lax.scan`` + trace-time-unrolled
RK step matches an independent NumPy integration. The RK4 ``slope >= 3.5`` guard
(F29-style) catches a stage-unroll bug that would silently drop RK4 to low order.
"""

from __future__ import annotations

import numpy as np
import pytest

from companions.ch05.jax import order_verification as ov

_DTS = np.array([0.1, 0.05, 0.025])
_T_END = 4.0


def _simulate_numpy(tab, dt: float, t_end: float) -> np.ndarray:
    """Independent NumPy RK integration (oracle for the scan + unrolled step)."""
    A, B, C = np.asarray(ov._A_MAT), np.asarray(ov._B_VEC), np.asarray(ov._C_ROW)
    An, bn, cn = np.asarray(tab.A), np.asarray(tab.b), np.asarray(tab.c)
    n_steps = int(round(t_end / dt)) + 1
    ts = np.arange(n_steps) * dt
    h = np.zeros(2)
    ys = np.zeros(n_steps)
    for kk in range(n_steps - 1):
        ys[kk] = C @ h
        s = An.shape[0]
        k = np.zeros((s, 2))
        for i in range(s):
            stage_h = h.copy()
            for j in range(i):
                stage_h = stage_h + dt * An[i, j] * k[j]
            k[i] = A @ stage_h + B * np.sin(2.0 * (ts[kk] + cn[i] * dt))
        h = h + dt * (bn @ k)
    ys[-1] = C @ h
    return ys


def test_rk1_first_order() -> None:
    _, slope = ov.empirical_slope(ov.forward_euler(), _DTS, _T_END)
    assert 0.8 <= slope <= 1.3, f"forward-Euler slope {slope:.3f} not ≈ 1"


def test_rk2_second_order() -> None:
    _, slope = ov.empirical_slope(ov.midpoint_rk2(), _DTS, _T_END)
    assert 1.7 <= slope <= 2.3, f"RK2 slope {slope:.3f} not ≈ 2"


def test_rk4_fourth_order() -> None:
    _, slope = ov.empirical_slope(ov.classical_rk4(), _DTS, _T_END)
    assert 3.6 <= slope <= 4.3, f"RK4 slope {slope:.3f} not ≈ 4"


def test_rk4_genuinely_high_order() -> None:
    """F29-style guard: a stage-unroll bug would drop RK4's slope toward ≤ 2."""
    _, slope = ov.empirical_slope(ov.classical_rk4(), _DTS, _T_END)
    assert slope >= 3.5, f"RK4 slope {slope:.3f} degraded below high order"


def test_scan_matches_numpy_integration() -> None:
    """lax.scan + unrolled rk_step equals an independent NumPy RK4 loop."""
    _, ys = ov.simulate(ov.classical_rk4(), 0.05, _T_END)
    ys_ref = _simulate_numpy(ov.classical_rk4(), 0.05, _T_END)
    np.testing.assert_allclose(ys, ys_ref, atol=1e-10)


def test_simulate_validation() -> None:
    with pytest.raises(ValueError):
        ov.simulate(ov.classical_rk4(), 0.0, _T_END)
    with pytest.raises(ValueError):
        ov.simulate(ov.classical_rk4(), 0.05, -1.0)
