"""Chapter 1 — damped harmonic oscillator energy-decay figure.

Simulates the underdamped damped harmonic oscillator $\\ddot q + c\\dot q + k q = 0$
via the state-space lift $h = (q, \\dot q)^\\top$ and `scipy.integrate.solve_ivp`
(Radau, an A-stable implicit method that handles stiff and non-stiff cases
gracefully). Computes the total energy $E(t) = \\tfrac{1}{2}(\\dot q^2 + k q^2)$
and plots it on linear and log scales — the constant log-slope confirms
exponential decay at rate $c/2$ in the energy norm.

Output
------
Writes ``public/figures/ch01/energy_decay.png`` (referenced from
``src/content/chapters/ch01-linear-odes.mdx`` §1.3).

Usage
-----
::

    PYTHONPATH=. python companions/ch01/jax/damped_oscillator.py
"""

from __future__ import annotations

from pathlib import Path

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

# Repo-root-relative output path so the chapter's <Figure src="/figures/..."> resolves.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUTPUT_PATH = _REPO_ROOT / "public" / "figures" / "ch01" / "energy_decay"


def damped_oscillator_rhs(t: float, h: np.ndarray, k: float, c: float) -> np.ndarray:
    """Right-hand side of the damped-oscillator ODE in state-space form.

    Parameters
    ----------
    t : float
        Time (unused; system is autonomous, but solve_ivp requires the signature).
    h : ndarray of shape (2,)
        State vector $(q, \\dot q)$.
    k : float
        Spring stiffness; must be positive.
    c : float
        Damping coefficient; non-negative (zero gives the undamped limit).

    Returns
    -------
    ndarray of shape (2,)
        Time derivative $\\dot h = (\\dot q, \\ddot q) = (\\dot q, -k q - c \\dot q)$.
    """
    q, qdot = h
    return np.array([qdot, -k * q - c * qdot])


def simulate(
    k: float = 4.0,
    c: float = 0.2,
    t_max: float = 40.0,
    n_points: int = 2001,
    h0: tuple[float, float] = (1.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the damped oscillator and compute its total energy.

    Parameters
    ----------
    k, c : float
        Stiffness and damping; defaults give the underdamped regime ($c^2 < 4k$).
    t_max : float
        Final simulation time. Must be positive.
    n_points : int
        Number of sample points (uniform spacing on $[0, t_{\\max}]$).
    h0 : tuple of two floats
        Initial state $(q(0), \\dot q(0))$.

    Returns
    -------
    t : ndarray of shape (n_points,)
    h : ndarray of shape (n_points, 2)
        Trajectory; columns are $q(t)$ and $\\dot q(t)$.
    energy : ndarray of shape (n_points,)
        $E(t) = \\tfrac{1}{2}(\\dot q^2 + k q^2)$ along the trajectory.

    Raises
    ------
    ValueError
        If ``k <= 0`` or ``t_max <= 0``.
    RuntimeError
        If ``solve_ivp`` fails to converge.
    """
    if k <= 0:
        raise ValueError(f"stiffness k must be positive, got {k}")
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")

    t_eval = np.linspace(0.0, t_max, n_points)
    sol = solve_ivp(
        damped_oscillator_rhs,
        t_span=(0.0, t_max),
        y0=np.asarray(h0, dtype=float),
        t_eval=t_eval,
        method="Radau",  # A-stable implicit; handles any damping regime.
        args=(k, c),
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    h = sol.y.T  # shape (n_points, 2)
    q, qdot = h[:, 0], h[:, 1]
    energy = 0.5 * (qdot**2 + k * q**2)
    return t_eval, h, energy


def make_figure() -> plt.Figure:
    """Build the two-panel energy-decay figure."""
    apply_style()
    t, _, energy = simulate()

    fig, axes = create_tufte_figure(nrows=1, ncols=2, figsize=(10.0, 4.0))
    ax_lin, ax_log = axes  # type: ignore[misc]

    ax_lin.plot(t, energy, color=SSM_COLORS["accent"], linewidth=1.6)
    set_tufte_title(ax_lin, "Energy E(t) — linear scale")
    set_tufte_labels(ax_lin, xlabel="time $t$", ylabel="$E(t)$")
    ax_lin.set_xlim(0, t[-1])
    ax_lin.set_ylim(bottom=0)

    ax_log.semilogy(t, energy, color=SSM_COLORS["accent"], linewidth=1.6)
    # Theoretical envelope: E ≈ E0 * exp(-c * t) since energy ~ amplitude^2.
    envelope = energy[0] * np.exp(-0.2 * t)
    ax_log.semilogy(t, envelope, color=SSM_COLORS["alert"], linewidth=1.0, linestyle="--",
                    label=r"theoretical envelope $E_0 e^{-c t}$")
    set_tufte_title(ax_log, "Energy E(t) — log scale")
    set_tufte_labels(ax_log, xlabel="time $t$", ylabel=r"$\log E(t)$")
    ax_log.legend(loc="upper right", frameon=False, fontsize=9)
    ax_log.set_xlim(0, t[-1])

    fig.suptitle("Damped harmonic oscillator: energy decay (k=4, c=0.2)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def main() -> None:
    fig = make_figure()
    paths = save_figure(fig, _OUTPUT_PATH, formats=("png",))
    plt.close(fig)
    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
