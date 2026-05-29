"""Chapter 1 — damped harmonic oscillator energy-decay figure.

Simulates the underdamped damped harmonic oscillator $\\ddot q + c\\dot q + k q = 0$
via the state-space lift $h = (q, \\dot q)^\\top$. The *demonstrated* integrator is
a fixed-step classical RK4 written with ``jax.lax.scan``; ``scipy.integrate.solve_ivp``
(Radau, an A-stable implicit method) is kept as the high-accuracy reference. The
figure plots the total energy $E(t) = \\tfrac{1}{2}(\\dot q^2 + k q^2)$ on linear
and log scales — the constant log-slope confirms exponential decay at rate $c$ in
the energy norm.

Idiomatic-JAX note (this companion is a NumPy/scipy→JAX teaching example)
------------------------------------------------------------------------
A fixed-step time integration is the canonical place to meet the ``lax.scan`` idiom:

* **``jax.lax.scan`` instead of a Python ``for k in range(n)`` time-stepping loop.**
  The carry is the state $h_k = (q_k, \\dot q_k)$; each step applies one RK4 update
  and emits $h_k$. There is no in-place ``hs[k] = ...`` mutation, and the whole
  trajectory fuses into a single compiled kernel — the same scan primitive that
  powers the S4 / Mamba selective scan in later chapters.
* **scipy stays the reference.** ``solve_ivp(Radau)`` is an adaptive, implicit,
  black-box loop; we do *not* reimplement it. It is the ground-truth curve the
  explicit JAX scheme is validated against (and overlaid against in the figure).

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

from functools import partial
from pathlib import Path

import jax

# Enable float64 before any jnp array exists; the log-scale energy decay spans
# many orders of magnitude and single precision would floor it prematurely.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

from companions._shared.plot_utils import (  # noqa: E402
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
    """Right-hand side of the damped-oscillator ODE in ``solve_ivp`` signature.

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


def _rhs_jax(h: jnp.ndarray, k: float, c: float) -> jnp.ndarray:
    """Autonomous RHS $(\\dot q, -k q - c \\dot q)$ for the JAX RK4 scan."""
    q, qdot = h
    return jnp.array([qdot, -k * q - c * qdot])


@partial(jax.jit, static_argnums=3)
def _rk4_trajectory(
    k: float, c: float, dt: float, n_steps: int, h0: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """RK4-integrate the oscillator with ``lax.scan``; return ``(ts, hs)``.

    ``n_steps`` is static (it sets the scan length and output shape). The scan
    carry is the state $h$; each step emits the pre-update state so the stacked
    output is $h_0, h_1, \\ldots, h_{n}$.
    """
    def step(h, _):  # carry = state (q, qdot); xs unused (autonomous system)
        k1 = _rhs_jax(h, k, c)
        k2 = _rhs_jax(h + 0.5 * dt * k1, k, c)
        k3 = _rhs_jax(h + 0.5 * dt * k2, k, c)
        k4 = _rhs_jax(h + dt * k3, k, c)
        return h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), h

    h_final, hs_head = jax.lax.scan(step, h0, None, length=n_steps)
    hs = jnp.concatenate([hs_head, h_final[None]])  # (n_steps + 1, 2)
    ts = jnp.arange(n_steps + 1) * dt
    return ts, hs


def simulate_scan(
    k: float = 4.0,
    c: float = 0.2,
    t_max: float = 40.0,
    n_steps: int = 2000,
    h0: tuple[float, float] = (1.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the damped oscillator with the JAX ``lax.scan`` RK4 and its energy.

    The *demonstrated* JAX method (contrast :func:`simulate_reference`).

    Parameters
    ----------
    k, c : float
        Stiffness and damping; defaults give the underdamped regime ($c^2 < 4k$).
    t_max : float
        Final simulation time. Must be positive.
    n_steps : int
        Number of RK4 steps (uniform $\\Delta = t_{\\max}/n_{\\text{steps}}$).
    h0 : tuple of two floats
        Initial state $(q(0), \\dot q(0))$.

    Returns
    -------
    t : ndarray of shape (n_steps + 1,)
    h : ndarray of shape (n_steps + 1, 2)
        Trajectory; columns are $q(t)$ and $\\dot q(t)$.
    energy : ndarray of shape (n_steps + 1,)
        $E(t) = \\tfrac{1}{2}(\\dot q^2 + k q^2)$ along the trajectory.

    Raises
    ------
    ValueError
        If ``k <= 0``, ``t_max <= 0``, or ``n_steps < 1``.
    """
    if k <= 0:
        raise ValueError(f"stiffness k must be positive, got {k}")
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")

    dt = t_max / n_steps
    ts, hs = _rk4_trajectory(k, c, dt, n_steps, jnp.asarray(h0, dtype=jnp.float64))
    energy = 0.5 * (hs[:, 1] ** 2 + k * hs[:, 0] ** 2)
    return np.asarray(ts), np.asarray(hs), np.asarray(energy)


def simulate_reference(
    k: float = 4.0,
    c: float = 0.2,
    t_max: float = 40.0,
    n_points: int = 2001,
    h0: tuple[float, float] = (1.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """High-accuracy ``solve_ivp(Radau)`` reference trajectory and its energy.

    Kept as the ground-truth oracle the JAX RK4 scheme is validated against; not
    reimplemented in JAX (see the module Idiomatic-JAX note).

    Returns ``(t, h, energy)`` with the same shapes as :func:`simulate_scan`.

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
    """Build the two-panel energy-decay figure (JAX RK4 + scipy reference)."""
    apply_style()
    c = 0.2
    t, _, energy = simulate_scan(c=c)
    t_ref, _, energy_ref = simulate_reference(c=c)

    fig, axes = create_tufte_figure(nrows=1, ncols=2, figsize=(10.0, 4.0))
    ax_lin, ax_log = axes  # type: ignore[misc]

    ax_lin.plot(t, energy, color=SSM_COLORS["accent"], linewidth=1.6,
                label="JAX RK4 (lax.scan)")
    ax_lin.plot(t_ref, energy_ref, color=SSM_COLORS["baseline"], linewidth=1.0,
                linestyle="--", label="scipy Radau (reference)")
    set_tufte_title(ax_lin, "Energy E(t) — linear scale")
    set_tufte_labels(ax_lin, xlabel="time $t$", ylabel="$E(t)$")
    ax_lin.legend(loc="upper right", frameon=False, fontsize=9)
    ax_lin.set_xlim(0, t[-1])
    ax_lin.set_ylim(bottom=0)

    ax_log.semilogy(t, energy, color=SSM_COLORS["accent"], linewidth=1.6)
    # Theoretical envelope: E ≈ E0 * exp(-c * t) since energy ~ amplitude^2.
    envelope = energy[0] * np.exp(-c * t)
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
