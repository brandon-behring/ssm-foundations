"""Chapter 4 — ZOH vs bilinear vs forward Euler on a forced linear oscillator.

Discretizes the forced 2-state damped oscillator
$$\\ddot q + c \\dot q + k q = u(t)$$
under the state-space lift $h = (q, \\dot q)^\\top$ via three schemes:

- **Forward Euler** (first-order, conditionally stable; included for contrast).
- **Zero-order hold (ZOH)** via the augmented matrix exponential trick
  (first-order on forced systems, autonomous-exact, A-stable).
- **Bilinear (Tustin) transform** (second-order, A-stable, maps the imaginary
  axis exactly to the unit circle).

Compares each discretization against a high-accuracy continuous reference
solution from `scipy.integrate.solve_ivp` (Radau) and emits two figures:

1. ``eigenvalue_migration.png`` — continuous eigenvalues (left half-plane) and
   the discrete eigenvalues each scheme produces (inside the unit disk).
2. ``order_convergence.png`` — log–log plot of max pointwise error versus step
   size for all three schemes, with linear fits showing empirical order.

Output paths
------------
``public/figures/ch04/eigenvalue_migration.png``
``public/figures/ch04/order_convergence.png``

Usage
-----
::

    PYTHONPATH=. python companions/ch04/jax/discretization_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import jax

# Enable float64 BEFORE any jnp array is created so module-level constants are
# not silently truncated to float32. JAX's default is float32; the warning that
# fires when you request float64 without this config is non-fatal but masks the
# extra precision we need for the order-of-convergence slope on small dt.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.scipy.linalg as jsl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

from companions._shared.plot_utils import (
    SSM_COLORS,
    apply_style,
    create_tufte_figure,
    save_figure,
    set_tufte_labels,
    set_tufte_title,
)

# Repo-root-relative output paths so chapter `<Figure src="/figures/...">` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch04"


# ---------------------------------------------------------------------------
# Test system: forced damped oscillator
#   \ddot q + c \dot q + k q = u(t),   u(t) = sin(2 t),
# lifted to h = (q, \dot q)^T with
#   A = [[0, 1], [-k, -c]],   B = [0, 1]^T,   C = [1, 0].
# Constants are picked so that the eigenvalues -0.25 \pm i \sqrt{15.75}/2 \approx -0.25 \pm 1.984 i
# place the system firmly in the open left half-plane.
# ---------------------------------------------------------------------------

_K_STIFF: float = 4.0  # spring constant
_C_DAMP: float = 0.5  # damping coefficient
_A_MAT: jnp.ndarray = jnp.array([[0.0, 1.0], [-_K_STIFF, -_C_DAMP]], dtype=jnp.float64)
_B_VEC: jnp.ndarray = jnp.array([0.0, 1.0], dtype=jnp.float64)
_C_ROW: jnp.ndarray = jnp.array([1.0, 0.0], dtype=jnp.float64)


def drive(t: np.ndarray | float) -> np.ndarray | float:
    """Smooth scalar input $u(t) = \\sin(2 t)$ (the forced-oscillator driver)."""
    return np.sin(2.0 * t)


# ---------------------------------------------------------------------------
# Discretizers
# ---------------------------------------------------------------------------


def discretize_forward_euler(
    A: jnp.ndarray, B: jnp.ndarray, dt: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Forward-Euler discretization: $\\bar A = I + \\Delta A$, $\\bar B = \\Delta B$.

    Only conditionally stable: for fixed $A$ with $\\operatorname{Re}(\\lambda) < 0$
    there exists $\\Delta_{\\max}$ beyond which $\\rho(\\bar A) > 1$ and the
    recurrence diverges. Included for pedagogical contrast with the A-stable
    schemes below.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    n = A.shape[0]
    Id = jnp.eye(n, dtype=A.dtype)
    return Id + dt * A, dt * B


def discretize_zoh(
    A: jnp.ndarray, B: jnp.ndarray, dt: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Zero-order-hold discretization via the augmented matrix exponential.

    Builds the block matrix $M = [[A \\Delta, B \\Delta], [0, 0]]$, computes
    $\\exp(M)$, and reads $\\bar A$ (top-left $N \\times N$ block) and $\\bar B$
    (top-right $N \\times P$ block). This avoids inverting $A$ — see Exercise 4.4
    for the proof that the block structure of $\\exp(M)$ contains
    $A^{-1}(e^{A \\Delta} - I) B$ in the top-right block when $A$ is invertible.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    n = A.shape[0]
    B_col = B[:, None]
    p = B_col.shape[1]
    top = jnp.concatenate([A * dt, B_col * dt], axis=1)
    bottom = jnp.zeros((p, n + p), dtype=A.dtype)
    aug = jnp.concatenate([top, bottom], axis=0)
    exp_aug = jsl.expm(aug)
    Ad = exp_aug[:n, :n]
    Bd = exp_aug[:n, n:].squeeze(-1)
    return Ad, Bd


def discretize_bilinear(
    A: jnp.ndarray, B: jnp.ndarray, dt: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Bilinear (Tustin) transform: $\\bar A = (I - \\Delta A/2)^{-1}(I + \\Delta A/2)$.

    Second-order accurate for smooth forcing; A-stable; maps the imaginary axis
    of the continuous plane exactly to the unit circle of the discrete plane.
    See §4.4 for the Möbius-geometry argument.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    n = A.shape[0]
    Id = jnp.eye(n, dtype=A.dtype)
    half = 0.5 * dt
    L = Id - half * A
    Ad = jnp.linalg.solve(L, Id + half * A)
    Bd = jnp.linalg.solve(L, dt * B)
    return Ad, Bd


# ---------------------------------------------------------------------------
# Step functions
#   Forward Euler / ZOH: $h_{k+1} = \bar A h_k + \bar B u_k$ (input held).
#   Bilinear:           $h_{k+1} = \bar A h_k + \bar B (u_k + u_{k+1})/2$
#                       (input midpoint; needed for true 2nd-order accuracy).
# ---------------------------------------------------------------------------


def step_hold(Ad: jnp.ndarray, Bd: jnp.ndarray, h: jnp.ndarray, u_k: float, _u_kp1: float) -> jnp.ndarray:
    """Input-hold step (forward Euler and ZOH)."""
    return Ad @ h + Bd * u_k


def step_midpoint(
    Ad: jnp.ndarray, Bd: jnp.ndarray, h: jnp.ndarray, u_k: float, u_kp1: float
) -> jnp.ndarray:
    """Input-midpoint step (bilinear: $(u_k + u_{k+1})/2$)."""
    return Ad @ h + Bd * 0.5 * (u_k + u_kp1)


# ---------------------------------------------------------------------------
# Simulation harness
# ---------------------------------------------------------------------------


def simulate(
    discretizer,
    stepper,
    dt: float,
    t_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a discrete scheme on the forced oscillator over [0, t_end].

    Parameters
    ----------
    discretizer
        Callable ``(A, B, dt) -> (Ad, Bd)``.
    stepper
        Callable ``(Ad, Bd, h, u_k, u_kp1) -> h_next``.
    dt
        Step size; positive.
    t_end
        Final time; positive.

    Returns
    -------
    ts : ndarray of shape (n,)
        Sample times $0, \\Delta, 2\\Delta, \\ldots, n\\Delta$.
    ys : ndarray of shape (n,)
        Output samples $C h_k$.

    Raises
    ------
    ValueError
        If ``dt <= 0`` or ``t_end <= 0``.

    Notes
    -----
    The time recurrence is expressed with ``jax.lax.scan`` rather than a Python
    ``for`` loop: the carry is the hidden state ``h`` and each step emits
    ``y_k = C h_k``. This is the idiomatic JAX spelling of a linear recurrence —
    and the very same scan primitive that powers the S4 / Mamba selective scan in
    later chapters. It is also pure (no in-place ``ys[k] = ...`` mutation) and
    fuses into a single compiled kernel.
    """
    if dt <= 0 or t_end <= 0:
        raise ValueError(f"dt and t_end must both be positive, got dt={dt}, t_end={t_end}")
    Ad, Bd = discretizer(_A_MAT, _B_VEC, dt)
    n_steps = int(round(t_end / dt)) + 1
    ts = jnp.arange(n_steps, dtype=_A_MAT.dtype) * dt
    u_vals = jnp.sin(2.0 * ts)  # drive(t) = sin(2t) as a JAX array

    def step(h, u_pair):  # carry = hidden state h; xs = (u_k, u_{k+1}) pairs
        u_k, u_kp1 = u_pair
        y_k = _C_ROW @ h
        h_next = stepper(Ad, Bd, h, u_k, u_kp1)
        return h_next, y_k

    h0 = jnp.zeros(2, dtype=_A_MAT.dtype)
    h_final, ys_head = jax.lax.scan(step, h0, (u_vals[:-1], u_vals[1:]))
    ys = jnp.concatenate([ys_head, (_C_ROW @ h_final)[None]])
    return np.asarray(ts), np.asarray(ys)


def continuous_reference(t_end: float, t_grid: np.ndarray) -> np.ndarray:
    """High-accuracy reference solution from `scipy.integrate.solve_ivp` (Radau).

    Returns the output $y(t) = C h(t)$ sampled at ``t_grid``. Radau is an
    A-stable implicit method with adaptive step control — accurate enough that
    its residual on our 4-second horizon is below ``1e-10``.
    """

    def rhs(t: float, h: np.ndarray) -> np.ndarray:
        return np.array(_A_MAT) @ h + np.array(_B_VEC) * drive(t)

    sol = solve_ivp(
        rhs,
        t_span=(0.0, t_end),
        y0=np.zeros(2),
        t_eval=t_grid,
        method="Radau",
        rtol=1e-11,
        atol=1e-13,
    )
    if not sol.success:
        raise RuntimeError(f"scipy.integrate.solve_ivp failed: {sol.message}")
    return np.array(_C_ROW) @ sol.y  # shape (n_grid,)


# ---------------------------------------------------------------------------
# Figure 1: eigenvalue migration
# ---------------------------------------------------------------------------


def make_eigenvalue_migration_figure(dt: float = 0.1) -> plt.Figure:
    """Visualize the continuous-to-discrete eigenvalue map for each scheme."""
    apply_style()
    A_np = np.array(_A_MAT)
    cont_eigs = np.linalg.eigvals(A_np)

    discs = {
        "Forward Euler": discretize_forward_euler(_A_MAT, _B_VEC, dt)[0],
        "ZOH": discretize_zoh(_A_MAT, _B_VEC, dt)[0],
        "Bilinear": discretize_bilinear(_A_MAT, _B_VEC, dt)[0],
    }
    disc_eigs = {name: np.linalg.eigvals(np.array(M)) for name, M in discs.items()}

    fig, axes = create_tufte_figure(nrows=1, ncols=2, figsize=(11.0, 4.6))
    ax_cont, ax_disc = axes  # type: ignore[misc]

    # Continuous panel: LHP region shaded, eigenvalues marked.
    ax_cont.axvspan(-3.0, 0.0, alpha=0.08, color=SSM_COLORS["accent"])
    ax_cont.axhline(0.0, color=SSM_COLORS["baseline"], linewidth=0.6)
    ax_cont.axvline(0.0, color=SSM_COLORS["baseline"], linewidth=0.6)
    ax_cont.scatter(
        cont_eigs.real,
        cont_eigs.imag,
        s=80,
        color=SSM_COLORS["accent"],
        zorder=3,
        edgecolors="white",
        linewidths=1.2,
    )
    ax_cont.set_xlim(-3.0, 0.5)
    ax_cont.set_ylim(-2.5, 2.5)
    set_tufte_title(ax_cont, "Continuous eigenvalues (left half-plane)")
    set_tufte_labels(ax_cont, xlabel=r"$\operatorname{Re}(\lambda)$", ylabel=r"$\operatorname{Im}(\lambda)$")
    ax_cont.set_aspect("equal", adjustable="box")

    # Discrete panel: unit disk + each scheme's eigenvalues.
    theta = np.linspace(0, 2 * np.pi, 256)
    ax_disc.plot(np.cos(theta), np.sin(theta), color=SSM_COLORS["baseline"], linewidth=0.8)
    ax_disc.axhline(0.0, color=SSM_COLORS["baseline"], linewidth=0.6)
    ax_disc.axvline(0.0, color=SSM_COLORS["baseline"], linewidth=0.6)
    palette = [SSM_COLORS["accent"], SSM_COLORS["highlight"], SSM_COLORS["alert"]]
    markers = ["o", "s", "^"]
    for (name, eigs), color, marker in zip(disc_eigs.items(), palette, markers):
        ax_disc.scatter(
            eigs.real,
            eigs.imag,
            s=70,
            color=color,
            marker=marker,
            label=name,
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
        )
    ax_disc.set_xlim(-1.3, 1.3)
    ax_disc.set_ylim(-1.3, 1.3)
    ax_disc.set_aspect("equal", adjustable="box")
    set_tufte_title(ax_disc, rf"Discrete eigenvalues at $\Delta = {dt:g}$")
    set_tufte_labels(ax_disc, xlabel=r"$\operatorname{Re}(\mu)$", ylabel=r"$\operatorname{Im}(\mu)$")
    ax_disc.legend(loc="upper left", frameon=False, fontsize=9)

    fig.suptitle(
        rf"Eigenvalue migration: continuous LHP $\to$ unit disk ($\Delta = {dt:g}$)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: order of convergence
# ---------------------------------------------------------------------------


def measure_max_error(discretizer, stepper, dt: float, t_end: float = 4.0) -> float:
    """Max pointwise output error of a scheme against the Radau reference."""
    ts, ys = simulate(discretizer, stepper, dt, t_end)
    y_ref = continuous_reference(t_end, ts)
    return float(np.max(np.abs(ys - y_ref)))


def make_order_convergence_figure() -> plt.Figure:
    """Log-log error vs step size for ZOH, bilinear, forward Euler."""
    apply_style()
    dts = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
    schemes = {
        "Forward Euler": (discretize_forward_euler, step_hold, SSM_COLORS["baseline"], "o"),
        "ZOH": (discretize_zoh, step_hold, SSM_COLORS["accent"], "s"),
        "Bilinear": (discretize_bilinear, step_midpoint, SSM_COLORS["highlight"], "^"),
    }
    fig, ax = create_tufte_figure(figsize=(6.4, 5.0))
    print("Empirical order (slope on last two dt):")
    print("-" * 50)
    for name, (disc_fn, step_fn, color, marker) in schemes.items():
        errs = np.array([measure_max_error(disc_fn, step_fn, float(dt)) for dt in dts])
        # Slope from the two finest dt's (where method error dominates roundoff).
        slope = np.log(errs[-2] / errs[-1]) / np.log(dts[-2] / dts[-1])
        ax.loglog(dts, errs, color=color, marker=marker, linewidth=1.4, label=f"{name} (slope $\\approx$ {slope:.2f})")
        print(f"  {name:<14s} slope ≈ {slope:5.2f}")
    print()
    set_tufte_title(ax, "Max pointwise error vs step size — forced damped oscillator")
    set_tufte_labels(ax, xlabel=r"step size $\Delta$", ylabel=r"max $|y_k - y_{\text{ref}}(t_k)|$")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Chapter 4 — discretization_comparison.py")
    print("=" * 60)
    fig1 = make_eigenvalue_migration_figure(dt=0.1)
    paths = save_figure(fig1, _OUT_DIR / "eigenvalue_migration", formats=("png",))
    for p in paths:
        print(f"Wrote {p}")
    plt.close(fig1)

    fig2 = make_order_convergence_figure()
    paths = save_figure(fig2, _OUT_DIR / "order_convergence", formats=("png",))
    for p in paths:
        print(f"Wrote {p}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
