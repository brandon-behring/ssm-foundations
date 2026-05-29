"""Chapter 6 — stiff systems: explicit RK4 versus backward Euler.

Demonstrates the central pedagogical point of §6.1: on a stiff ODE, an
explicit Runge-Kutta method requires step sizes inversely proportional to
the largest eigenvalue magnitude (here, the van der Pol stiffness parameter
$\\mu$), while an implicit method like backward Euler remains stable for
*every* positive step size.

Test problem: van der Pol oscillator
$$\\dot q = p, \\qquad \\dot p = \\mu (1 - q^2) p - q,$$
with $\\mu = 10$ (mildly stiff but well-behaved on float64). The limit
cycle period is $\\sim 2.0 \\mu$ for large $\\mu$, with rapid jumps lasting
$\\sim 1/\\mu$ — the fast time scale that constrains explicit step sizes.

Output
------
``public/figures/ch06/stiff_blowup.png`` — two-panel figure showing RK4's
divergence at moderate step size alongside backward Euler's stability.

Usage
-----
::

    PYTHONPATH=. python companions/ch06/jax/stiff_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from companions._shared.plot_utils import (
    SSM_COLORS,
    apply_style,
    create_tufte_figure,
    save_figure,
    set_tufte_labels,
    set_tufte_title,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_PATH = _REPO_ROOT / "public" / "figures" / "ch06" / "stiff_blowup"

_MU: float = 10.0  # Mild stiffness — visible without breaking cold-start Newton at coarse dt.


def vdp_rhs(h: np.ndarray) -> np.ndarray:
    """Van der Pol right-hand side $f(q, p) = (p, \\mu(1-q^2)p - q)$."""
    q, p = h
    return np.array([p, _MU * (1.0 - q * q) * p - q])


def vdp_jacobian(h: np.ndarray) -> np.ndarray:
    """Jacobian of the van der Pol RHS — needed for backward-Euler Newton."""
    q, p = h
    return np.array(
        [
            [0.0, 1.0],
            [-2.0 * _MU * q * p - 1.0, _MU * (1.0 - q * q)],
        ]
    )


# ---------------------------------------------------------------------------
# Explicit RK4
# ---------------------------------------------------------------------------


def rk4_step(f: Callable[[np.ndarray], np.ndarray], h: np.ndarray, dt: float) -> np.ndarray:
    """One step of classical Runge-Kutta 4 (autonomous form)."""
    k1 = f(h)
    k2 = f(h + 0.5 * dt * k1)
    k3 = f(h + 0.5 * dt * k2)
    k4 = f(h + dt * k3)
    return h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# Backward Euler — solved via Newton iteration on g(x) = x - h - dt f(x) = 0
# ---------------------------------------------------------------------------


def backward_euler_step(
    f: Callable[[np.ndarray], np.ndarray],
    jac: Callable[[np.ndarray], np.ndarray],
    h: np.ndarray,
    dt: float,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> np.ndarray:
    """Backward Euler with Newton iteration to solve the implicit equation.

    Solves $g(x) = x - h - \\Delta f(x) = 0$ for $x = h_{k+1}$ using
    Newton steps $x \\leftarrow x - g(x) / g'(x)$, where $g'(x) = I - \\Delta J(x)$.

    Parameters
    ----------
    f : Callable
        Right-hand side $f(h)$.
    jac : Callable
        Jacobian $\\partial f / \\partial h$.
    h : ndarray, shape (n,)
        Current state.
    dt : float
        Step size; positive.
    tol : float
        Convergence tolerance on the residual norm.
    max_iter : int
        Maximum Newton iterations.

    Returns
    -------
    h_next : ndarray of shape (n,)
        State at $t + \\Delta$.

    Raises
    ------
    ValueError
        If ``dt <= 0`` or Newton fails to converge within ``max_iter``.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    n = h.shape[0]
    Id = np.eye(n)
    # Forward-Euler initial guess: closer to root than `h` alone for stiff problems.
    x = h + dt * f(h)
    g = x - h - dt * f(x)
    g_norm = float(np.linalg.norm(g))
    for _ in range(max_iter):
        if g_norm < tol:
            return x
        Jg = Id - dt * jac(x)
        delta = np.linalg.solve(Jg, g)
        # Damped Newton: backtrack if the full step doesn't reduce the residual.
        step = 1.0
        for _ in range(20):
            x_trial = x - step * delta
            g_trial = x_trial - h - dt * f(x_trial)
            g_trial_norm = float(np.linalg.norm(g_trial))
            if g_trial_norm < g_norm:
                x = x_trial
                g = g_trial
                g_norm = g_trial_norm
                break
            step *= 0.5
        else:
            # Even the smallest step didn't help — give up and report.
            raise ValueError(
                f"Backward Euler damped Newton stalled at dt={dt}; "
                f"residual {g_norm:.3e}, state {x.tolist()}"
            )
    raise ValueError(f"Backward Euler Newton did not converge in {max_iter} iters at dt={dt}")


# ---------------------------------------------------------------------------
# Simulate up to t_end and return trajectories
# ---------------------------------------------------------------------------


def simulate_rk4(h0: np.ndarray, dt: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    n_steps = int(round(t_end / dt)) + 1
    ts = np.arange(n_steps) * dt
    hs = np.zeros((n_steps, h0.shape[0]))
    hs[0] = h0
    for k in range(n_steps - 1):
        try:
            hs[k + 1] = rk4_step(vdp_rhs, hs[k], dt)
        except (FloatingPointError, OverflowError):
            hs[k + 1 :] = np.nan
            break
        # Detect overflow / explosion explicitly so the plot truncates.
        if np.any(np.abs(hs[k + 1]) > 1e8):
            hs[k + 1 :] = np.nan
            break
    return ts, hs


def simulate_be(h0: np.ndarray, dt: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    n_steps = int(round(t_end / dt)) + 1
    ts = np.arange(n_steps) * dt
    hs = np.zeros((n_steps, h0.shape[0]))
    hs[0] = h0
    for k in range(n_steps - 1):
        hs[k + 1] = backward_euler_step(vdp_rhs, vdp_jacobian, hs[k], dt)
    return ts, hs


# ---------------------------------------------------------------------------
# Figure: side-by-side panels at increasing dt
# ---------------------------------------------------------------------------


def make_figure() -> plt.Figure:
    apply_style()
    h0 = np.array([2.0, 0.0])
    t_end = 50.0  # ~2.5 limit-cycle periods at mu=10
    dts = [0.005, 0.05, 0.2]  # progressively coarser; RK4 blows up at the coarsest

    fig, axes = create_tufte_figure(nrows=1, ncols=2, figsize=(12.0, 4.6))
    ax_rk4, ax_be = axes  # type: ignore[misc]

    palette = [SSM_COLORS["accent"], SSM_COLORS["highlight"], SSM_COLORS["alert"]]

    for dt, color in zip(dts, palette):
        ts_rk4, hs_rk4 = simulate_rk4(h0, dt, t_end)
        ax_rk4.plot(ts_rk4, hs_rk4[:, 0], color=color, linewidth=1.2, label=rf"$\Delta = {dt:g}$")

        ts_be, hs_be = simulate_be(h0, dt, t_end)
        ax_be.plot(ts_be, hs_be[:, 0], color=color, linewidth=1.2, label=rf"$\Delta = {dt:g}$")

    for ax, title in zip((ax_rk4, ax_be), ("Classical RK4 (explicit)", "Backward Euler (implicit, L-stable)")):
        set_tufte_title(ax, title)
        set_tufte_labels(ax, xlabel="time $t$", ylabel=r"$q(t)$ (position)")
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        ax.set_xlim(0, t_end)
        ax.set_ylim(-3.0, 3.0)

    fig.suptitle(rf"Van der Pol ($\mu = {_MU:g}$): RK4 blowup vs backward Euler stability", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def main() -> None:
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Chapter 6 — stiff_demo.py")
    print("=" * 60)
    fig = make_figure()
    paths = save_figure(fig, _OUT_PATH, formats=("png",))
    for p in paths:
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    np.seterr(over="raise", invalid="raise")
    main()
