"""Chapter 6 — symplectic vs non-symplectic integrators on Hamiltonian systems.

Demonstrates the central pedagogical point of §6.5: standard order-optimized
RK methods accumulate linear-in-time energy drift on Hamiltonian systems,
while symplectic integrators preserve a *modified* Hamiltonian and exhibit
only bounded energy oscillation.

Two test systems:

1. Harmonic oscillator $\\dot q = p, \\dot p = -q$ with $\\hamilton = (p^2 + q^2)/2$
   — exactly solvable; the difference between RK4 and Verlet is a clean,
   quantitative measurement.
2. Pendulum $\\dot q = p, \\dot p = -\\sin q$ with $\\hamilton = p^2/2 - \\cos q$
   — nonlinear, exhibits closed orbits in the trapped regime and rotating
   solutions above the separatrix; phase-space visualization makes the
   geometric difference visible.

Idiomatic-JAX note (this companion is a NumPy→JAX teaching example)
------------------------------------------------------------------
* **``jax.lax.scan`` replaces the Python ``for k in range(n_steps)`` loop.** The
  carry is the phase-space point $(q, p)$; each step applies one integrator
  update and emits the pre-step state. No in-place ``qs[k+1] = ...`` mutation;
  the whole trajectory fuses into one compiled kernel — the same scan primitive
  as the S4 / Mamba selective scan.

Output
------
``public/figures/ch06/energy_drift.png`` — energy vs time for both methods.
``public/figures/ch06/phase_portrait.png`` — pendulum phase portrait, both methods.

Usage
-----
::

    PYTHONPATH=. python companions/ch06/jax/symplectic_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from companions._shared.plot_utils import (  # noqa: E402
    SSM_COLORS,
    apply_style,
    create_tufte_figure,
    save_figure,
    set_tufte_labels,
    set_tufte_title,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch06"


# ---------------------------------------------------------------------------
# Hamiltonian systems
# ---------------------------------------------------------------------------


def harmonic_T_grad(p: jnp.ndarray) -> jnp.ndarray:
    """$\\partial T/\\partial p = p$ for the harmonic oscillator."""
    return p


def harmonic_V_grad(q: jnp.ndarray) -> jnp.ndarray:
    """$\\partial V/\\partial q = q$ for the harmonic oscillator."""
    return q


def harmonic_H(q: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    """Harmonic-oscillator energy $H = (p^2 + q^2)/2$."""
    return 0.5 * (p * p + q * q)


def pendulum_T_grad(p: jnp.ndarray) -> jnp.ndarray:
    """$\\partial T/\\partial p = p$ for the pendulum (unit mass)."""
    return p


def pendulum_V_grad(q: jnp.ndarray) -> jnp.ndarray:
    """$\\partial V/\\partial q = \\sin q$ for the pendulum."""
    return jnp.sin(q)


def pendulum_H(q: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    """Pendulum energy $H = p^2/2 - \\cos q$."""
    return 0.5 * p * p - jnp.cos(q)


# ---------------------------------------------------------------------------
# Integrators (all autonomous; (q, p) phase coordinates)
# ---------------------------------------------------------------------------


def rk4_step_hamilton(T_grad: Callable, V_grad: Callable, q: jnp.ndarray, p: jnp.ndarray, dt: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Classical RK4 on $\\dot q = T'(p), \\dot p = -V'(q)$ — not symplectic."""

    def f(state: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        qq, pp = state
        return (T_grad(pp), -V_grad(qq))

    k1q, k1p = f((q, p))
    k2q, k2p = f((q + 0.5 * dt * k1q, p + 0.5 * dt * k1p))
    k3q, k3p = f((q + 0.5 * dt * k2q, p + 0.5 * dt * k2p))
    k4q, k4p = f((q + dt * k3q, p + dt * k3p))
    q_next = q + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)
    p_next = p + (dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
    return q_next, p_next


def verlet_step(T_grad: Callable, V_grad: Callable, q: jnp.ndarray, p: jnp.ndarray, dt: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Störmer-Verlet (symplectic, 2nd order, time-reversible).

    Half-kick / drift / half-kick:
        p_{n+1/2} = p_n - (Δ/2) V'(q_n)
        q_{n+1}   = q_n + Δ T'(p_{n+1/2})
        p_{n+1}   = p_{n+1/2} - (Δ/2) V'(q_{n+1})
    """
    p_half = p - 0.5 * dt * V_grad(q)
    q_next = q + dt * T_grad(p_half)
    p_next = p_half - 0.5 * dt * V_grad(q_next)
    return q_next, p_next


# ---------------------------------------------------------------------------
# Simulation harness
# ---------------------------------------------------------------------------


def simulate(
    stepper: Callable,
    T_grad: Callable,
    V_grad: Callable,
    q0: float,
    p0: float,
    dt: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(ts, qs, ps)`` arrays of length ``n_steps + 1`` from a stepper.

    The time loop is a ``jax.lax.scan``: the carry is the phase-space point
    $(q, p)$ and each step emits the pre-update state.
    """
    def step(carry, _):  # carry = (q, p); emit pre-step state
        q, p = carry
        q_next, p_next = stepper(T_grad, V_grad, q, p, dt)
        return (q_next, p_next), (q, p)

    init = (jnp.asarray(q0, dtype=jnp.float64), jnp.asarray(p0, dtype=jnp.float64))
    (q_f, p_f), (qs_head, ps_head) = jax.lax.scan(step, init, None, length=n_steps)
    qs = jnp.concatenate([qs_head, q_f[None]])
    ps = jnp.concatenate([ps_head, p_f[None]])
    ts = jnp.arange(n_steps + 1) * dt
    return np.asarray(ts), np.asarray(qs), np.asarray(ps)


def rk4_drift_per_period(dt: float = 0.05, periods: int = 100) -> float:
    """Mean RK4 energy drift per period on the harmonic oscillator (Exercise 6.3).

    Returns $(E(T) - E_0) / \\text{periods}$ for the unit harmonic oscillator from
    $(q_0, p_0) = (1, 0)$; at $\\Delta = 0.05$ this is $\\approx -1.4\\times 10^{-8}$.
    """
    n_steps = int(round(periods * 2 * np.pi / dt))
    _, qs, ps = simulate(rk4_step_hamilton, harmonic_T_grad, harmonic_V_grad, 1.0, 0.0, dt, n_steps)
    energy = 0.5 * (ps**2 + qs**2)
    return float((energy[-1] - 0.5) / periods)


# ---------------------------------------------------------------------------
# Figure 1: Energy drift on the harmonic oscillator
# ---------------------------------------------------------------------------


def make_energy_drift_figure() -> plt.Figure:
    apply_style()
    # Choose a step size large enough that RK4's per-step error is well above
    # roundoff: at dt=0.3 on the harmonic oscillator, RK4's local error is
    # ~O(dt^5) = ~2e-3 per step, accumulating to a clearly visible drift over
    # 100 periods (~2100 steps). Verlet at dt=0.3 has bounded O(dt^2) ~ 0.09
    # energy oscillation. The contrast is the point of the figure.
    dt = 0.3
    periods = 100
    n_steps = int(round(periods * 2 * np.pi / dt))  # 100 periods at omega = 1

    _, qs_rk4, ps_rk4 = simulate(rk4_step_hamilton, harmonic_T_grad, harmonic_V_grad, 1.0, 0.0, dt, n_steps)
    ts, qs_vrl, ps_vrl = simulate(verlet_step, harmonic_T_grad, harmonic_V_grad, 1.0, 0.0, dt, n_steps)

    E_rk4 = 0.5 * (ps_rk4**2 + qs_rk4**2)
    E_vrl = 0.5 * (ps_vrl**2 + qs_vrl**2)
    E_0 = 0.5

    fig, ax = create_tufte_figure(figsize=(7.5, 4.6))
    ax.axhline(E_0, color=SSM_COLORS["baseline"], linewidth=0.8, linestyle=":", label=r"$E_0 = 0.5$")
    ax.plot(ts / (2 * np.pi), E_rk4 - E_0, color=SSM_COLORS["highlight"], linewidth=1.2, label="Classical RK4 (non-symplectic)")
    ax.plot(ts / (2 * np.pi), E_vrl - E_0, color=SSM_COLORS["accent"], linewidth=1.2, label="Störmer-Verlet (symplectic)")
    set_tufte_title(ax, rf"Energy error vs time — harmonic oscillator ($\Delta = {dt}$, {periods} periods)")
    set_tufte_labels(ax, xlabel=r"time (periods)", ylabel=r"$E(t) - E_0$")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)

    drift_per_period = (E_rk4[-1] - E_0) / periods
    print(f"  RK4 drift rate: {drift_per_period:.2e} energy per period (at dt = {dt})")
    print(f"  Verlet band: [{(E_vrl - E_0).min():.2e}, {(E_vrl - E_0).max():.2e}] energy")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Phase portrait on the pendulum
# ---------------------------------------------------------------------------


def make_phase_portrait_figure() -> plt.Figure:
    apply_style()
    dt = 0.1  # deliberately coarse to expose qualitative differences
    periods = 50
    n_steps = int(round(periods * 2 * np.pi / dt))

    # Trapped (oscillating) initial condition.
    q0, p0 = 1.0, 0.0

    _, qs_rk4, ps_rk4 = simulate(rk4_step_hamilton, pendulum_T_grad, pendulum_V_grad, q0, p0, dt, n_steps)
    _, qs_vrl, ps_vrl = simulate(verlet_step, pendulum_T_grad, pendulum_V_grad, q0, p0, dt, n_steps)

    fig, ax = create_tufte_figure(figsize=(6.5, 6.0))
    # Phase portrait: q on x, p on y.
    ax.plot(qs_rk4, ps_rk4, color=SSM_COLORS["highlight"], linewidth=0.7, alpha=0.7, label="Classical RK4")
    ax.plot(qs_vrl, ps_vrl, color=SSM_COLORS["accent"], linewidth=0.7, alpha=0.7, label="Störmer-Verlet")
    ax.scatter([q0], [p0], s=40, color=SSM_COLORS["baseline"], zorder=3, label="initial $(q_0, p_0)$")
    set_tufte_title(ax, rf"Pendulum phase portrait ($\Delta = {dt}$, {periods} periods)")
    set_tufte_labels(ax, xlabel="$q$ (angle)", ylabel="$p$ (momentum)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Chapter 6 — symplectic_demo.py")
    print("=" * 60)

    fig1 = make_energy_drift_figure()
    paths = save_figure(fig1, _OUT_DIR / "energy_drift", formats=("png",))
    for p in paths:
        print(f"Wrote {p}")
    plt.close(fig1)

    fig2 = make_phase_portrait_figure()
    paths = save_figure(fig2, _OUT_DIR / "phase_portrait", formats=("png",))
    for p in paths:
        print(f"Wrote {p}")
    plt.close(fig2)

    # Exercise 6.3 (0527-F34): make the Δ=0.05 RK4 drift rate reproducible.
    drift_005 = rk4_drift_per_period(dt=0.05, periods=100)
    print(f"  RK4 drift @ Δ=0.05: {drift_005:.3e} energy/period (Exercise 6.3 ≈ 1.4e-8)")


if __name__ == "__main__":
    main()
