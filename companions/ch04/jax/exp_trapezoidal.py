"""Chapter 4 — exponential-trapezoidal scheme on the forced linear oscillator.

Implements the second-order exponential-trapezoidal integrator
$$h_{k+1} = e^{A \\Delta} h_k + \\Delta \\, \\varphi_1(A \\Delta) B u_k
    + \\Delta \\, \\varphi_2(A \\Delta) B (u_{k+1} - u_k),$$
where $\\varphi_1(z) = (e^z - 1)/z$ and $\\varphi_2(z) = (e^z - 1 - z)/z^2$
are the first two $\\varphi$-functions of the exponential family
(see §4.5 and Exercise 4.6 in Chapter 4).

The matrix $\\varphi$-functions are computed via the **augmented matrix
exponential** trick (the block-matrix identity is Van Loan 1978; Al-Mohy &
Higham 2011 give the modern simultaneous-$\\varphi$ formulation), which is
both numerically stable for small $\\Delta$ and well-defined for singular $A$:

  exp(diag([A \\Delta, B \\Delta, B \\Delta]) with one shifted block) returns
  $e^{A \\Delta}$, $\\Delta \\varphi_1(A \\Delta) B$, and
  $\\Delta \\varphi_2(A \\Delta) B$ in three diagonal blocks of one
  $(N + 2P) \\times (N + 2P)$ matrix exponential.

Output
------
``public/figures/ch04/exp_trap_convergence.png`` — log-log error-vs-step plot
showing exp-trapezoidal achieves slope 2 alongside the bilinear-trapezoidal
scheme and beats ZOH's slope 1.

Notes
-----
The ``simulate_*`` helpers below keep an eager Python loop for the linear
recurrence, rather than the ``jax.lax.scan`` spelling used in
``discretization_comparison.py``. This is deliberate: the loop maps verbatim
onto the PyTorch companion (``companions/ch04/torch/exp_trapezoidal.py``),
keeping the two ports line-for-line comparable for the §4.5 teaching contrast.

Usage
-----
::

    PYTHONPATH=. python companions/ch04/jax/exp_trapezoidal.py
"""

from __future__ import annotations

from pathlib import Path

import jax

# Enable float64 BEFORE jnp arrays exist (see discretization_comparison.py).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.scipy.linalg as jsl  # noqa: E402
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

# Repo-root-relative output path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_PATH = _REPO_ROOT / "public" / "figures" / "ch04" / "exp_trap_convergence"

# Same forced damped oscillator as discretization_comparison.py.
_A_MAT: jnp.ndarray = jnp.array([[0.0, 1.0], [-4.0, -0.5]], dtype=jnp.float64)
_B_VEC: jnp.ndarray = jnp.array([0.0, 1.0], dtype=jnp.float64)
_C_ROW: jnp.ndarray = jnp.array([1.0, 0.0], dtype=jnp.float64)


def drive(t: np.ndarray | float) -> np.ndarray | float:
    """$u(t) = \\sin(2 t)$ — the same smooth scalar input as §4.3 / §4.4."""
    return np.sin(2.0 * t)


def discretize_exp_trap(
    A: jnp.ndarray, B: jnp.ndarray, dt: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Exponential-trapezoidal discretization via augmented matrix exponential.

    Builds the $(N + 2) \\times (N + 2)$ block matrix
    $$
    M = \\begin{pmatrix} A \\Delta & B \\Delta & 0 \\\\ 0 & 0 & \\Delta \\\\ 0 & 0 & 0 \\end{pmatrix},
    $$
    whose exponential has top-row blocks
    $$
    \\exp(M)_{0:N} = \\bigl(e^{A \\Delta}, \\; \\Delta \\varphi_1(A \\Delta) B, \\; \\Delta^2 \\varphi_2(A \\Delta) B\\bigr).
    $$
    The last block carries the $\\Delta^2$ factor, which we divide back out to
    return $\\Delta \\varphi_2(A \\Delta) B$ as required by the scheme of §4.5.

    Returns
    -------
    Ad : jnp.ndarray, shape (N, N)
        Discrete dynamics matrix $e^{A \\Delta}$.
    B0 : jnp.ndarray, shape (N,)
        Coefficient of $u_k$: $\\Delta \\varphi_1(A \\Delta) B$.
    B1 : jnp.ndarray, shape (N,)
        Coefficient of $(u_{k+1} - u_k)$: $\\Delta \\varphi_2(A \\Delta) B$.

    Raises
    ------
    ValueError
        If ``dt <= 0``.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    n = A.shape[0]
    # Construct M = dt · Â where Â is the augmented matrix
    #     Â = [[A, B, 0], [0, 0, 1], [0, 0, 0]]
    # so M = [[A·dt, B·dt, 0], [0, 0, dt], [0, 0, 0]] (shape (n+2, n+2)).
    # The (n, n+1) entry MUST be `dt`, not `1`: solving dy/dt = Â y with
    # y(0) = e_{n+2} gives y_3(t) = 1, y_2(t) = t, y_1(t) = t² φ_2(t A) B —
    # which is exactly what we need exp(dt·Â) to evaluate at t = dt.
    M = jnp.zeros((n + 2, n + 2), dtype=A.dtype)
    M = M.at[:n, :n].set(A * dt)
    M = M.at[:n, n].set(B * dt)
    M = M.at[n, n + 1].set(dt)
    EM = jsl.expm(M)
    Ad = EM[:n, :n]
    B0 = EM[:n, n]  # Δ φ₁(A Δ) B  (the trapezoidal "constant" piece)
    B1_scaled = EM[:n, n + 1]  # Δ² φ₂(A Δ) B  (the trapezoidal "linear" piece, scaled by Δ)
    B1 = B1_scaled / dt
    return Ad, B0, B1


def simulate_exp_trap(dt: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    """Run the exp-trapezoidal scheme on the forced oscillator."""
    if dt <= 0 or t_end <= 0:
        raise ValueError(f"dt and t_end must be positive, got dt={dt}, t_end={t_end}")
    Ad, B0, B1 = discretize_exp_trap(_A_MAT, _B_VEC, dt)
    n_steps = int(round(t_end / dt)) + 1
    h = jnp.zeros(2, dtype=_A_MAT.dtype)
    ts = np.arange(n_steps) * dt
    u_vals = drive(ts)
    ys = np.zeros(n_steps)
    for k in range(n_steps - 1):
        ys[k] = float((_C_ROW @ h).item())
        u_k = float(u_vals[k])
        u_kp1 = float(u_vals[k + 1])
        h = Ad @ h + B0 * u_k + B1 * (u_kp1 - u_k)
    ys[-1] = float((_C_ROW @ h).item())
    return ts, ys


def simulate_zoh(dt: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    """ZOH scheme — baseline for comparison."""
    if dt <= 0 or t_end <= 0:
        raise ValueError(f"dt and t_end must be positive, got dt={dt}, t_end={t_end}")
    n = _A_MAT.shape[0]
    aug = jnp.block([[_A_MAT * dt, _B_VEC[:, None] * dt], [jnp.zeros((1, n + 1), dtype=_A_MAT.dtype)]])
    EX = jsl.expm(aug)
    Ad = EX[:n, :n]
    Bd = EX[:n, n]
    n_steps = int(round(t_end / dt)) + 1
    h = jnp.zeros(2, dtype=_A_MAT.dtype)
    ts = np.arange(n_steps) * dt
    u_vals = drive(ts)
    ys = np.zeros(n_steps)
    for k in range(n_steps - 1):
        ys[k] = float((_C_ROW @ h).item())
        h = Ad @ h + Bd * float(u_vals[k])
    ys[-1] = float((_C_ROW @ h).item())
    return ts, ys


def simulate_bilinear(dt: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear/Tustin scheme with input midpoint — baseline for comparison."""
    if dt <= 0 or t_end <= 0:
        raise ValueError(f"dt and t_end must be positive, got dt={dt}, t_end={t_end}")
    n = _A_MAT.shape[0]
    Id = jnp.eye(n, dtype=_A_MAT.dtype)
    half = 0.5 * dt
    L = Id - half * _A_MAT
    Ad = jnp.linalg.solve(L, Id + half * _A_MAT)
    Bd = jnp.linalg.solve(L, dt * _B_VEC)
    n_steps = int(round(t_end / dt)) + 1
    h = jnp.zeros(2, dtype=_A_MAT.dtype)
    ts = np.arange(n_steps) * dt
    u_vals = drive(ts)
    ys = np.zeros(n_steps)
    for k in range(n_steps - 1):
        ys[k] = float((_C_ROW @ h).item())
        u_mid = 0.5 * (float(u_vals[k]) + float(u_vals[k + 1]))
        h = Ad @ h + Bd * u_mid
    ys[-1] = float((_C_ROW @ h).item())
    return ts, ys


def continuous_reference(t_end: float, t_grid: np.ndarray) -> np.ndarray:
    """High-accuracy reference from `scipy.integrate.solve_ivp` (Radau)."""

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
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return np.array(_C_ROW) @ sol.y


def make_figure() -> plt.Figure:
    """Log-log error-vs-step plot for ZOH / bilinear / exp-trapezoidal."""
    apply_style()
    dts = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
    t_end = 4.0

    schemes = {
        "ZOH": (simulate_zoh, SSM_COLORS["accent"], "o"),
        "Bilinear (trap)": (simulate_bilinear, SSM_COLORS["highlight"], "s"),
        "Exp-trapezoidal": (simulate_exp_trap, SSM_COLORS["alert"], "^"),
    }

    fig, ax = create_tufte_figure(figsize=(6.4, 5.0))
    print("Empirical convergence orders (slope on finest two $\\Delta$):")
    print("-" * 60)
    for name, (sim_fn, color, marker) in schemes.items():
        errs = np.zeros(len(dts))
        for i, dt in enumerate(dts):
            ts, ys = sim_fn(float(dt), t_end)
            y_ref = continuous_reference(t_end, ts)
            errs[i] = float(np.max(np.abs(ys - y_ref)))
        slope = np.log(errs[-2] / errs[-1]) / np.log(dts[-2] / dts[-1])
        ax.loglog(dts, errs, color=color, marker=marker, linewidth=1.4, label=f"{name} (slope $\\approx$ {slope:.2f})")
        print(f"  {name:<18s} slope ≈ {slope:5.2f}")

    set_tufte_title(ax, "Exp-trapezoidal vs ZOH and bilinear (forced oscillator)")
    set_tufte_labels(ax, xlabel=r"step size $\Delta$", ylabel=r"max $|y_k - y_{\text{ref}}(t_k)|$")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Chapter 4 — exp_trapezoidal.py")
    print("=" * 60)
    fig = make_figure()
    paths = save_figure(fig, _OUT_PATH, formats=("png",))
    for p in paths:
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
