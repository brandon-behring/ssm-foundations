"""Chapter 5 (PyTorch companion) — empirical order verification of RK1 / RK2 / RK4.

Mirrors ``companions/ch05/jax/order_verification.py`` for the JAX↔PyTorch contrast.
This is the **compute-and-parity** port: it reproduces the numerical core (tableau-
driven forward Euler, midpoint RK2, classical RK4 run on the Chapter-4 forced damped
oscillator, with the log-log error-vs-step-size slope) but draws no figures — the
``make_figure`` / ``save_figure`` helpers live only in the JAX companion.

JAX↔PyTorch contrast
--------------------
* **Time recurrence.** The JAX companion threads the state $h$ through ``jax.lax.scan``,
  emitting $y_k = C h_k$ each step. PyTorch is define-by-run and has no ``scan``
  primitive, so ``simulate`` runs the recurrence as a plain eager Python loop — the
  loop *is* the program.
* **Butcher stage loop.** JAX *unrolls* the data-dependent inner ``for j in range(i)``
  at trace time (the tableau is a compile-time constant). PyTorch just runs the same
  nested loop eagerly; no unrolling distinction exists.
* **Reference.** ``scipy.integrate.solve_ivp(Radau)`` is the framework-agnostic
  high-accuracy reference for *both* companions, so the empirical slopes coincide to
  roundoff (the parity claim).
* **Precision.** float64 throughout (``torch.float64``); JAX sets it globally with
  ``jax_enable_x64``. The RK4 error reaches ~1e-7 at the finest ``dt`` and the slopes
  need precision well below float32.

Usage
-----
::

    PYTHONPATH=. python companions/ch05/torch/order_verification.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.integrate import solve_ivp

_DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Test problem: same forced damped oscillator as Chapter 4.
#   d/dt [q, q̇] = [[0, 1], [-4, -0.5]] [q, q̇] + [0, 1] · sin(2t)
#   y = [1, 0] · [q, q̇]
# Eigenvalues -0.25 ± i·√(15.75)/2 ≈ -0.25 ± 1.984i — firmly in the LHP, so RK methods with
# stability regions covering the eigenvalue × dt point are stable.
# ---------------------------------------------------------------------------

_A_MAT = torch.tensor([[0.0, 1.0], [-4.0, -0.5]], dtype=_DTYPE)
_B_VEC = torch.tensor([0.0, 1.0], dtype=_DTYPE)
_C_ROW = torch.tensor([1.0, 0.0], dtype=_DTYPE)


def drive(t: torch.Tensor) -> torch.Tensor:
    return torch.sin(2.0 * t)


def rhs(t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Forced-oscillator RHS $\\dot h = A h + B \\sin(2t)$ (pure torch)."""
    return _A_MAT @ h + _B_VEC * drive(t)


# ---------------------------------------------------------------------------
# Tableau-driven Runge-Kutta integrators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tableau:
    name: str
    A: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    expected_order: int


def forward_euler() -> Tableau:
    return Tableau(
        "Forward Euler",
        A=torch.tensor([[0.0]], dtype=_DTYPE),
        b=torch.tensor([1.0], dtype=_DTYPE),
        c=torch.tensor([0.0], dtype=_DTYPE),
        expected_order=1,
    )


def midpoint_rk2() -> Tableau:
    return Tableau(
        "Midpoint RK2",
        A=torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=_DTYPE),
        b=torch.tensor([0.0, 1.0], dtype=_DTYPE),
        c=torch.tensor([0.0, 0.5], dtype=_DTYPE),
        expected_order=2,
    )


def classical_rk4() -> Tableau:
    return Tableau(
        "Classical RK4",
        A=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=_DTYPE,
        ),
        b=torch.tensor([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0], dtype=_DTYPE),
        c=torch.tensor([0.0, 0.5, 0.5, 1.0], dtype=_DTYPE),
        expected_order=4,
    )


def rk_step(tab: Tableau, t: torch.Tensor, h: torch.Tensor, dt: float) -> torch.Tensor:
    """One explicit Runge-Kutta step from a Butcher tableau.

    The nested stage loop runs eagerly: where the JAX companion unrolls the inner
    ``for j in range(i)`` at trace time, PyTorch (define-by-run) simply executes it.
    """
    s = tab.A.shape[0]
    stages: list[torch.Tensor] = []
    for i in range(s):
        stage_h = h
        for j in range(i):
            stage_h = stage_h + dt * tab.A[i, j] * stages[j]
        stages.append(rhs(t + tab.c[i] * dt, stage_h))
    k_stack = torch.stack(stages)  # (s, n)
    return h + dt * (tab.b @ k_stack)


def simulate(tab: Tableau, dt: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    """Run the RK method on the forced oscillator over [0, t_end] from h = 0.

    The time stepping is an eager Python loop (the PyTorch counterpart of the JAX
    companion's ``jax.lax.scan``); the carry is the state and each step records
    $y_k = C h_k$.

    Raises
    ------
    ValueError
        If ``dt <= 0`` or ``t_end <= 0``.
    """
    if dt <= 0 or t_end <= 0:
        raise ValueError(f"dt and t_end must be positive, got dt={dt}, t_end={t_end}")
    n_steps = int(round(t_end / dt)) + 1
    ts = torch.arange(n_steps, dtype=_DTYPE) * dt

    h = torch.zeros(2, dtype=_DTYPE)
    ys = torch.empty(n_steps, dtype=_DTYPE)
    for k in range(n_steps - 1):
        ys[k] = _C_ROW @ h
        h = rk_step(tab, ts[k], h, dt)
    ys[n_steps - 1] = _C_ROW @ h
    return ts.numpy(), ys.numpy()


def continuous_reference(t_end: float, t_grid: np.ndarray) -> np.ndarray:
    """High-accuracy ``solve_ivp(Radau)`` reference output $y(t)$ on ``t_grid``."""
    A_np = _A_MAT.numpy()
    B_np = _B_VEC.numpy()
    C_np = _C_ROW.numpy()

    def rhs_np(t: float, h: np.ndarray) -> np.ndarray:
        return A_np @ h + B_np * np.sin(2.0 * t)

    sol = solve_ivp(
        rhs_np,
        t_span=(0.0, t_end),
        y0=np.zeros(2),
        t_eval=t_grid,
        method="Radau",
        rtol=1e-12,
        atol=1e-14,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return C_np @ sol.y


def empirical_slope(
    tab: Tableau, dts: np.ndarray, t_end: float = 4.0
) -> tuple[np.ndarray, float]:
    """Return (errors over ``dts``, log-log slope from the two finest steps)."""
    errs = np.zeros(len(dts))
    for i, dt in enumerate(dts):
        ts, ys = simulate(tab, float(dt), t_end)
        y_ref = continuous_reference(t_end, ts)
        errs[i] = float(np.max(np.abs(ys - y_ref)))
    slope = float(np.log(errs[-2] / errs[-1]) / np.log(dts[-2] / dts[-1]))
    return errs, slope


# ---------------------------------------------------------------------------
# Entry point (numeric summary only — no figures)
# ---------------------------------------------------------------------------


def main() -> None:
    print("Chapter 5 (torch) — order_verification.py")
    print("=" * 60)
    dts = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
    t_end = 4.0
    print("Empirical RK orders (forced damped oscillator, t_end = 4):")
    print("-" * 60)
    for tab in (forward_euler(), midpoint_rk2(), classical_rk4()):
        _errs, slope = empirical_slope(tab, dts, t_end)
        print(f"  {tab.name:<16s} slope ≈ {slope:5.2f}   (expected {tab.expected_order})")


if __name__ == "__main__":
    main()
