"""Chapter 5 — empirical order verification of RK1, RK2, RK4 via Butcher tableaux.

Implements forward Euler (RK1, order 1), midpoint RK2 (order 2), and classical
RK4 (order 4) directly from their Butcher tableaux and verifies the empirical
order by running them on the forced damped oscillator from Chapter 4 and
fitting a log-log slope of error vs step size.

The slopes are taken from the two finest step sizes — fine enough that the
method error still dominates over float64 roundoff (\\sim10^{-13}), as
discussed in Exercise 5.3.

Output
------
``public/figures/ch05/order_verification.png``

Usage
-----
::

    PYTHONPATH=. python companions/ch05/jax/order_verification.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from companions._shared.plot_utils import (
    SSM_COLORS,
    apply_style,
    create_tufte_figure,
    save_figure,
    set_tufte_labels,
    set_tufte_title,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_PATH = _REPO_ROOT / "public" / "figures" / "ch05" / "order_verification"


# ---------------------------------------------------------------------------
# Test problem: same forced damped oscillator as Chapter 4.
#   d/dt [q, q̇] = [[0, 1], [-4, -0.5]] [q, q̇] + [0, 1] · sin(2t)
#   y = [1, 0] · [q, q̇]
# Eigenvalues -0.25 ± i·√(15)/4 — firmly in the LHP, so RK methods with
# stability regions covering the eigenvalue × dt point are stable.
# ---------------------------------------------------------------------------

_A_MAT = np.array([[0.0, 1.0], [-4.0, -0.5]])
_B_VEC = np.array([0.0, 1.0])
_C_ROW = np.array([1.0, 0.0])


def drive(t: float) -> float:
    return float(np.sin(2.0 * t))


def rhs(t: float, h: np.ndarray) -> np.ndarray:
    return _A_MAT @ h + _B_VEC * drive(t)


# ---------------------------------------------------------------------------
# Tableau-driven Runge-Kutta integrators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tableau:
    name: str
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    expected_order: int


def forward_euler() -> Tableau:
    return Tableau("Forward Euler", A=np.array([[0.0]]), b=np.array([1.0]), c=np.array([0.0]), expected_order=1)


def midpoint_rk2() -> Tableau:
    return Tableau(
        "Midpoint RK2",
        A=np.array([[0.0, 0.0], [0.5, 0.0]]),
        b=np.array([0.0, 1.0]),
        c=np.array([0.0, 0.5]),
        expected_order=2,
    )


def classical_rk4() -> Tableau:
    return Tableau(
        "Classical RK4",
        A=np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
        b=np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]),
        c=np.array([0.0, 0.5, 0.5, 1.0]),
        expected_order=4,
    )


def rk_step(f: Callable[[float, np.ndarray], np.ndarray], tab: Tableau, t: float, h: np.ndarray, dt: float) -> np.ndarray:
    """One step of an explicit Runge-Kutta method from its Butcher tableau."""
    s = tab.A.shape[0]
    k = np.zeros((s, h.shape[0]))
    for i in range(s):
        stage_h = h.copy()
        for j in range(i):
            stage_h = stage_h + dt * tab.A[i, j] * k[j]
        k[i] = f(t + tab.c[i] * dt, stage_h)
    return h + dt * (tab.b @ k)


def simulate(tab: Tableau, dt: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    """Run the RK method on the forced oscillator over [0, t_end] from h = 0."""
    if dt <= 0 or t_end <= 0:
        raise ValueError(f"dt and t_end must be positive, got dt={dt}, t_end={t_end}")
    n_steps = int(round(t_end / dt)) + 1
    h = np.zeros(2)
    ts = np.array([k_idx * dt for k_idx in range(n_steps)])
    ys = np.zeros(n_steps)
    for k_idx in range(n_steps - 1):
        ys[k_idx] = float(_C_ROW @ h)
        h = rk_step(rhs, tab, ts[k_idx], h, dt)
    ys[-1] = float(_C_ROW @ h)
    return ts, ys


def continuous_reference(t_end: float, t_grid: np.ndarray) -> np.ndarray:
    sol = solve_ivp(rhs, t_span=(0.0, t_end), y0=np.zeros(2), t_eval=t_grid, method="Radau", rtol=1e-12, atol=1e-14)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return _C_ROW @ sol.y


# ---------------------------------------------------------------------------
# Figure: log-log error vs step size
# ---------------------------------------------------------------------------


def make_figure() -> plt.Figure:
    apply_style()
    dts = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
    t_end = 4.0

    schemes = {
        "Forward Euler": (forward_euler(), SSM_COLORS["baseline"], "o"),
        "Midpoint RK2": (midpoint_rk2(), SSM_COLORS["accent"], "s"),
        "Classical RK4": (classical_rk4(), SSM_COLORS["highlight"], "^"),
    }

    fig, ax = create_tufte_figure(figsize=(6.4, 5.0))
    print("Empirical RK orders (forced damped oscillator, t_end = 4):")
    print("-" * 60)
    for name, (tab, color, marker) in schemes.items():
        errs = np.zeros(len(dts))
        for i, dt in enumerate(dts):
            ts, ys = simulate(tab, float(dt), t_end)
            y_ref = continuous_reference(t_end, ts)
            errs[i] = float(np.max(np.abs(ys - y_ref)))
        slope = np.log(errs[-2] / errs[-1]) / np.log(dts[-2] / dts[-1])
        ax.loglog(dts, errs, color=color, marker=marker, linewidth=1.4, label=f"{name} (empirical slope $\\approx$ {slope:.2f}, expected {tab.expected_order})")
        print(f"  {name:<16s} slope ≈ {slope:5.2f}   (expected {tab.expected_order})")

    set_tufte_title(ax, "Empirical convergence orders of RK1 / RK2 / RK4")
    set_tufte_labels(ax, xlabel=r"step size $\Delta$", ylabel=r"max $|y_k - y_{\text{ref}}(t_k)|$")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Chapter 5 — order_verification.py")
    print("=" * 60)
    fig = make_figure()
    paths = save_figure(fig, _OUT_PATH, formats=("png",))
    for p in paths:
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
