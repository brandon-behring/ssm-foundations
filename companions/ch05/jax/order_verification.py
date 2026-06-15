"""Chapter 5 — empirical order verification of RK1, RK2, RK4 via Butcher tableaux.

Implements forward Euler (RK1, order 1), midpoint RK2 (order 2), and classical
RK4 (order 4) directly from their Butcher tableaux and verifies the empirical
order by running them on the forced damped oscillator from Chapter 4 and
fitting a log-log slope of error vs step size.

The slopes are taken from the two finest step sizes — fine enough that the
method error still dominates over float64 roundoff (\\sim10^{-13}), as
discussed in Exercise 5.3.

Idiomatic-JAX note (this companion is a NumPy→JAX teaching example)
------------------------------------------------------------------
Two idioms appear, and one *anti*-idiom worth naming:

* **``jax.lax.scan`` for the time recurrence** (``simulate``): the carry is the
  state $h$, each step emits $y_k = C h_k$ — the same scan primitive as the S4 /
  Mamba selective scan.
* **Trace-time unrolling for the Butcher stage loop** (``rk_step``): the inner
  ``for j in range(i)`` has a *data-dependent* bound, which a traced loop cannot
  have. But the tableau is a compile-time constant with ``s <= 4`` stages, so the
  Python loops simply unroll at JAX trace time — each ``tab.A[i, j]`` is a
  concrete index. This is the right idiom here; masking the triangular structure
  with ``jnp.where`` (the "vectorise everything" reflex) would be strictly worse.
* ``scipy.integrate.solve_ivp(Radau)`` stays the high-accuracy reference.

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

import jax

# Enable float64 before any jnp array exists; the RK4 error reaches ~1e-7 at the
# finest dt and the order slopes need precision well below float32.
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

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_PATH = _REPO_ROOT / "public" / "figures" / "ch05" / "order_verification"


# ---------------------------------------------------------------------------
# Test problem: same forced damped oscillator as Chapter 4.
#   d/dt [q, q̇] = [[0, 1], [-4, -0.5]] [q, q̇] + [0, 1] · sin(2t)
#   y = [1, 0] · [q, q̇]
# Eigenvalues -0.25 ± i·√(15.75)/2 ≈ -0.25 ± 1.984i — firmly in the LHP, so RK methods with
# stability regions covering the eigenvalue × dt point are stable.
# ---------------------------------------------------------------------------

_A_MAT = jnp.array([[0.0, 1.0], [-4.0, -0.5]], dtype=jnp.float64)
_B_VEC = jnp.array([0.0, 1.0], dtype=jnp.float64)
_C_ROW = jnp.array([1.0, 0.0], dtype=jnp.float64)


def drive(t: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(2.0 * t)


def rhs(t: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
    """Forced-oscillator RHS $\\dot h = A h + B \\sin(2t)$ (pure JAX)."""
    return _A_MAT @ h + _B_VEC * drive(t)


# ---------------------------------------------------------------------------
# Tableau-driven Runge-Kutta integrators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tableau:
    name: str
    A: jnp.ndarray
    b: jnp.ndarray
    c: jnp.ndarray
    expected_order: int


def forward_euler() -> Tableau:
    return Tableau("Forward Euler", A=jnp.array([[0.0]]), b=jnp.array([1.0]),
                   c=jnp.array([0.0]), expected_order=1)


def midpoint_rk2() -> Tableau:
    return Tableau(
        "Midpoint RK2",
        A=jnp.array([[0.0, 0.0], [0.5, 0.0]]),
        b=jnp.array([0.0, 1.0]),
        c=jnp.array([0.0, 0.5]),
        expected_order=2,
    )


def classical_rk4() -> Tableau:
    return Tableau(
        "Classical RK4",
        A=jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
        b=jnp.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]),
        c=jnp.array([0.0, 0.5, 0.5, 1.0]),
        expected_order=4,
    )


def rk_step(tab: Tableau, t: jnp.ndarray, h: jnp.ndarray, dt: float) -> jnp.ndarray:
    """One explicit Runge-Kutta step from a Butcher tableau.

    The stage loop is unrolled at trace time: ``s`` and the triangular sparsity
    of ``A`` are compile-time constants, so the data-dependent inner
    ``for j in range(i)`` is a concrete Python loop building a traced expression
    (no ``jnp.where`` masking needed).
    """
    s = tab.A.shape[0]
    stages: list[jnp.ndarray] = []
    for i in range(s):
        stage_h = h
        for j in range(i):  # concrete range -> fully unrolled at trace time
            stage_h = stage_h + dt * tab.A[i, j] * stages[j]
        stages.append(rhs(t + tab.c[i] * dt, stage_h))
    k_stack = jnp.stack(stages)  # (s, n)
    return h + dt * (tab.b @ k_stack)


def simulate(tab: Tableau, dt: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    """Run the RK method on the forced oscillator over [0, t_end] from h = 0.

    The time stepping is a ``jax.lax.scan`` over the sample times; the carry is
    the state and each step emits $y_k = C h_k$.

    Raises
    ------
    ValueError
        If ``dt <= 0`` or ``t_end <= 0``.
    """
    if dt <= 0 or t_end <= 0:
        raise ValueError(f"dt and t_end must be positive, got dt={dt}, t_end={t_end}")
    n_steps = int(round(t_end / dt)) + 1
    ts = jnp.arange(n_steps) * dt

    def step(h, t_k):  # carry = state h; xs = current time t_k (pre-step)
        y_k = _C_ROW @ h
        return rk_step(tab, t_k, h, dt), y_k

    h_final, ys_head = jax.lax.scan(step, jnp.zeros(2), ts[:-1])
    ys = jnp.concatenate([ys_head, (_C_ROW @ h_final)[None]])
    return np.asarray(ts), np.asarray(ys)


def continuous_reference(t_end: float, t_grid: np.ndarray) -> np.ndarray:
    """High-accuracy ``solve_ivp(Radau)`` reference output $y(t)$ on ``t_grid``."""
    A_np, B_np, C_np = np.asarray(_A_MAT), np.asarray(_B_VEC), np.asarray(_C_ROW)

    def rhs_np(t: float, h: np.ndarray) -> np.ndarray:
        return A_np @ h + B_np * np.sin(2.0 * t)

    sol = solve_ivp(rhs_np, t_span=(0.0, t_end), y0=np.zeros(2), t_eval=t_grid,
                    method="Radau", rtol=1e-12, atol=1e-14)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return C_np @ sol.y


def empirical_slope(tab: Tableau, dts: np.ndarray, t_end: float = 4.0) -> tuple[np.ndarray, float]:
    """Return (errors over ``dts``, log-log slope from the two finest steps)."""
    errs = np.zeros(len(dts))
    for i, dt in enumerate(dts):
        ts, ys = simulate(tab, float(dt), t_end)
        y_ref = continuous_reference(t_end, ts)
        errs[i] = float(np.max(np.abs(ys - y_ref)))
    slope = float(np.log(errs[-2] / errs[-1]) / np.log(dts[-2] / dts[-1]))
    return errs, slope


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
        errs, slope = empirical_slope(tab, dts, t_end)
        ax.loglog(dts, errs, color=color, marker=marker, linewidth=1.4,
                  label=f"{name} (empirical slope $\\approx$ {slope:.2f}, expected {tab.expected_order})")
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
