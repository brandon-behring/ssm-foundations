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

Idiomatic-JAX note (this companion is a NumPy→JAX teaching example)
------------------------------------------------------------------
Three idioms, each replacing a NumPy pattern:

* **``jax.jacfwd`` replaces the hand-coded Jacobian.** The backward-Euler Newton
  step needs $\\partial f/\\partial h$; instead of maintaining the analytic
  ``vdp_jacobian`` by hand (a transcription-error risk), we differentiate
  ``vdp_rhs`` with forward-mode autodiff. (The analytic form survives only as a
  test oracle.)
* **``lax.while_loop`` damped Newton mirrors NumPy's solve-to-tolerance loop.**
  Backward Euler needs an implicit solve each step. We iterate damped Newton until
  the residual falls below tolerance (``while_loop`` — the structured-control-flow
  analogue of NumPy's ``while not converged``), with a ``jax.vmap``'d backtracking
  line search that picks the residual-minimizing step fraction. (A *fixed*
  iteration count under-converges during the van der Pol fast jumps at coarse dt;
  the trajectory-wide ``test_backward_euler_residual`` guard catches that.)
* **``jnp.where(|h| > 1e8, nan)`` replaces ``np.seterr`` + ``try/except``.** XLA
  has no ``FloatingPointError``; explicit-RK divergence at coarse dt produces
  ``inf``/``nan`` silently, so we mask the blown-up tail to ``nan`` *after* the
  scan, letting the plot truncate at the same visual point.

Both time integrations are ``jax.lax.scan`` over the step count.

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

from functools import partial
from pathlib import Path

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
_OUT_PATH = _REPO_ROOT / "public" / "figures" / "ch06" / "stiff_blowup"

_MU: float = 10.0  # Stiffness parameter (mild; visible on float64).
_BLOWUP_THRESHOLD: float = 1e8  # |state| above this is treated as RK4 divergence.
_NEWTON_TOL: float = 1e-10  # backward-Euler Newton residual tolerance.
_NEWTON_MAX_ITER: int = 60  # safeguard cap on damped-Newton iterations.
# Backtracking step fractions (1, 1/2, ..., ~3e-5) for the line search: at a
# coarse dt the forward-Euler warm start sits outside the pure-Newton basin
# during the van der Pol fast jumps, so each iteration takes the
# residual-minimizing fraction of the Newton step rather than the full step.
_NEWTON_DAMPING = jnp.array([0.5**k for k in range(16)])


def vdp_rhs(h: jnp.ndarray) -> jnp.ndarray:
    """Van der Pol right-hand side $f(q, p) = (p, \\mu(1-q^2)p - q)$."""
    q, p = h
    return jnp.array([p, _MU * (1.0 - q * q) * p - q])


# Forward-mode autodiff Jacobian — the teaching point; replaces a hand-coded
# vdp_jacobian (kept only as the test oracle in test_stiff.py).
_vdp_jac = jax.jacfwd(vdp_rhs)


# ---------------------------------------------------------------------------
# Explicit RK4
# ---------------------------------------------------------------------------


def rk4_step(h: jnp.ndarray, dt: float) -> jnp.ndarray:
    """One step of classical Runge-Kutta 4 on the van der Pol system (autonomous)."""
    k1 = vdp_rhs(h)
    k2 = vdp_rhs(h + 0.5 * dt * k1)
    k3 = vdp_rhs(h + 0.5 * dt * k2)
    k4 = vdp_rhs(h + dt * k3)
    return h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# Backward Euler — Newton on g(x) = x - h - dt f(x) = 0, fixed iteration count
# ---------------------------------------------------------------------------


def backward_euler_step(
    h: jnp.ndarray, dt: float, tol: float = _NEWTON_TOL, max_iter: int = _NEWTON_MAX_ITER
) -> jnp.ndarray:
    """Backward Euler via a damped Newton solve iterated to tolerance.

    Solves $g(x) = x - h - \\Delta f(x) = 0$ with damped Newton steps
    $x \\leftarrow x - \\alpha\\,[I - \\Delta J(x)]^{-1} g(x)$ from a forward-Euler
    warm start. The Jacobian $J$ comes from ``jax.jacfwd``. Each iteration runs a
    JAX-clean backtracking line search — it evaluates the residual at a fixed set
    of step fractions $\\alpha$ (``_NEWTON_DAMPING``) with ``jax.vmap`` and keeps
    the smallest-residual candidate (``argmin``). The iteration count is *adaptive*
    via ``jax.lax.while_loop`` (the analogue of NumPy's while-not-converged loop):
    a fixed iteration count under-converges during the van der Pol fast jumps at
    coarse dt, so we loop until $\\|g\\| < \\text{tol}$ (capped at ``max_iter``).
    """
    Id = jnp.eye(h.shape[0])

    def resnorm(x):
        return jnp.linalg.norm(x - h - dt * vdp_rhs(x))

    def cond(state):  # keep iterating while not converged and under the cap
        x, it = state
        return (resnorm(x) > tol) & (it < max_iter)

    def body(state):
        x, it = state
        delta = jnp.linalg.solve(Id - dt * _vdp_jac(x), x - h - dt * vdp_rhs(x))
        candidates = x[None, :] - _NEWTON_DAMPING[:, None] * delta[None, :]  # (n_alpha, n)
        return candidates[jnp.argmin(jax.vmap(resnorm)(candidates))], it + 1

    x_final, _ = jax.lax.while_loop(cond, body, (h + dt * vdp_rhs(h), 0))
    return x_final


# ---------------------------------------------------------------------------
# Simulate up to t_end and return trajectories (lax.scan time stepping)
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=2)
def _rk4_trajectory(h0: jnp.ndarray, dt: float, n_steps: int) -> jnp.ndarray:
    def step(h, _):  # carry = state; emit pre-step state
        return rk4_step(h, dt), h

    h_final, hs_head = jax.lax.scan(step, h0, None, length=n_steps - 1)
    hs = jnp.concatenate([hs_head, h_final[None]])
    # Explicit-RK divergence -> sentinel to NaN so the plot truncates (replaces
    # NumPy's np.seterr/try-except; |nan| > thr is False so NaNs stay NaN).
    return jnp.where(jnp.abs(hs) > _BLOWUP_THRESHOLD, jnp.nan, hs)


@partial(jax.jit, static_argnums=2)
def _be_trajectory(h0: jnp.ndarray, dt: float, n_steps: int) -> jnp.ndarray:
    def step(h, _):
        return backward_euler_step(h, dt), h

    h_final, hs_head = jax.lax.scan(step, h0, None, length=n_steps - 1)
    return jnp.concatenate([hs_head, h_final[None]])


def simulate_rk4(h0: np.ndarray, dt: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    """Explicit RK4 trajectory; diverged tail is masked to NaN. Raises on dt/t_end <= 0."""
    if dt <= 0 or t_end <= 0:
        raise ValueError(f"dt and t_end must be positive, got dt={dt}, t_end={t_end}")
    n_steps = int(round(t_end / dt)) + 1
    hs = _rk4_trajectory(jnp.asarray(h0, dtype=jnp.float64), dt, n_steps)
    return np.asarray(jnp.arange(n_steps) * dt), np.asarray(hs)


def simulate_be(h0: np.ndarray, dt: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    """Backward-Euler trajectory (stable at any dt). Raises on dt/t_end <= 0."""
    if dt <= 0 or t_end <= 0:
        raise ValueError(f"dt and t_end must be positive, got dt={dt}, t_end={t_end}")
    n_steps = int(round(t_end / dt)) + 1
    hs = _be_trajectory(jnp.asarray(h0, dtype=jnp.float64), dt, n_steps)
    return np.asarray(jnp.arange(n_steps) * dt), np.asarray(hs)


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
    main()
