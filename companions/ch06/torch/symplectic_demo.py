"""Chapter 6 (PyTorch companion) — symplectic vs non-symplectic integrators.

Mirrors ``companions/ch06/jax/symplectic_demo.py`` for the JAX↔PyTorch contrast
(this is the **compute + parity** half of the Ch 6 companion — no figures). Same
pedagogy as §6.5: standard order-optimized RK methods accumulate linear-in-time
energy drift on Hamiltonian systems, while symplectic integrators preserve a
*modified* Hamiltonian and exhibit only bounded energy oscillation.

Two test systems:

1. Harmonic oscillator $\\dot q = p, \\dot p = -q$ with $H = (p^2 + q^2)/2$ —
   exactly solvable; the RK4-vs-Verlet difference is a clean, quantitative
   measurement.
2. Pendulum $\\dot q = p, \\dot p = -\\sin q$ with $H = p^2/2 - \\cos q$ —
   nonlinear, with closed orbits in the trapped regime.

JAX↔PyTorch contrast
--------------------
* **Time loop.** The JAX companion threads the phase-space point $(q, p)$ through
  ``jax.lax.scan`` and emits the pre-step state in one fused, compiled pass.
  PyTorch is *define-by-run* with no ``scan`` primitive, so :func:`simulate` runs
  the same recurrence as a plain eager Python loop, appending each pre-step state.
  The update *ordering* (half-kick / drift / half-kick for Verlet) is what makes
  the integrator symplectic, so it is mirrored byte-for-byte.
* **Precision.** JAX enables float64 globally (``jax_enable_x64``); PyTorch sets
  precision per tensor, so the simulation carries ``dtype=torch.float64`` and the
  per-system gradients return float64 tensors.

Usage
-----
::

    PYTHONPATH=. python companions/ch06/torch/symplectic_demo.py
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

_DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Hamiltonian systems
# ---------------------------------------------------------------------------


def harmonic_T_grad(p: torch.Tensor) -> torch.Tensor:
    """$\\partial T/\\partial p = p$ for the harmonic oscillator."""
    return p


def harmonic_V_grad(q: torch.Tensor) -> torch.Tensor:
    """$\\partial V/\\partial q = q$ for the harmonic oscillator."""
    return q


def harmonic_H(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Harmonic-oscillator energy $H = (p^2 + q^2)/2$."""
    return 0.5 * (p * p + q * q)


def pendulum_T_grad(p: torch.Tensor) -> torch.Tensor:
    """$\\partial T/\\partial p = p$ for the pendulum (unit mass)."""
    return p


def pendulum_V_grad(q: torch.Tensor) -> torch.Tensor:
    """$\\partial V/\\partial q = \\sin q$ for the pendulum."""
    return torch.sin(q)


def pendulum_H(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Pendulum energy $H = p^2/2 - \\cos q$."""
    return 0.5 * p * p - torch.cos(q)


# ---------------------------------------------------------------------------
# Integrators (all autonomous; (q, p) phase coordinates)
# ---------------------------------------------------------------------------


def rk4_step_hamilton(
    T_grad: Callable, V_grad: Callable, q: torch.Tensor, p: torch.Tensor, dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classical RK4 on $\\dot q = T'(p), \\dot p = -V'(q)$ — not symplectic."""

    def f(state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        qq, pp = state
        return (T_grad(pp), -V_grad(qq))

    k1q, k1p = f((q, p))
    k2q, k2p = f((q + 0.5 * dt * k1q, p + 0.5 * dt * k1p))
    k3q, k3p = f((q + 0.5 * dt * k2q, p + 0.5 * dt * k2p))
    k4q, k4p = f((q + dt * k3q, p + dt * k3p))
    q_next = q + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)
    p_next = p + (dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
    return q_next, p_next


def verlet_step(
    T_grad: Callable, V_grad: Callable, q: torch.Tensor, p: torch.Tensor, dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Störmer-Verlet (symplectic, 2nd order, time-reversible).

    Half-kick / drift / half-kick::

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

    PyTorch contrast: where the JAX companion uses ``jax.lax.scan`` (carry =
    $(q, p)$, emitting the pre-update state), this runs the same recurrence as a
    define-by-run Python loop. Each iterate stores the pre-step state, then the
    final state is appended, so the trajectory has length ``n_steps + 1``.

    Raises
    ------
    ValueError
        If ``dt <= 0`` or ``n_steps < 0``.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got dt={dt}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got n_steps={n_steps}")

    q = torch.as_tensor(q0, dtype=_DTYPE)
    p = torch.as_tensor(p0, dtype=_DTYPE)
    qs = [q.clone()]
    ps = [p.clone()]
    for _ in range(n_steps):
        q, p = stepper(T_grad, V_grad, q, p, dt)
        qs.append(q.clone())
        ps.append(p.clone())

    ts = torch.arange(n_steps + 1, dtype=_DTYPE) * dt
    return ts.numpy(), torch.stack(qs).numpy(), torch.stack(ps).numpy()


def rk4_drift_per_period(dt: float = 0.05, periods: int = 100) -> float:
    """Mean RK4 energy drift per period on the harmonic oscillator (Exercise 6.3).

    Returns $(E(T) - E_0) / \\text{periods}$ for the unit harmonic oscillator from
    $(q_0, p_0) = (1, 0)$; at $\\Delta = 0.05$ this is $\\approx -1.4\\times 10^{-8}$.
    """
    n_steps = int(round(periods * 2 * np.pi / dt))
    _, qs, ps = simulate(
        rk4_step_hamilton, harmonic_T_grad, harmonic_V_grad, 1.0, 0.0, dt, n_steps
    )
    energy = 0.5 * (ps**2 + qs**2)
    return float((energy[-1] - 0.5) / periods)


def main() -> None:
    print("Chapter 6 (torch) — symplectic_demo.py")
    print("=" * 60)

    # Energy drift on the harmonic oscillator (the §6.5 contrast).
    dt = 0.3
    periods = 100
    n_steps = int(round(periods * 2 * np.pi / dt))
    _, qs_rk4, ps_rk4 = simulate(
        rk4_step_hamilton, harmonic_T_grad, harmonic_V_grad, 1.0, 0.0, dt, n_steps
    )
    _, qs_vrl, ps_vrl = simulate(
        verlet_step, harmonic_T_grad, harmonic_V_grad, 1.0, 0.0, dt, n_steps
    )
    E_rk4 = 0.5 * (ps_rk4**2 + qs_rk4**2)
    E_vrl = 0.5 * (ps_vrl**2 + qs_vrl**2)
    E_0 = 0.5
    print(f"  Harmonic oscillator (Δ={dt}, {periods} periods):")
    print(f"    RK4 energy drift: {(E_rk4[-1] - E_0) / periods:.2e} per period")
    print(f"    Verlet band: [{(E_vrl - E_0).min():.2e}, {(E_vrl - E_0).max():.2e}]")

    # Exercise 6.3 (0527-F34): the Δ=0.05 RK4 drift rate.
    drift_005 = rk4_drift_per_period(dt=0.05, periods=100)
    print(f"  RK4 drift @ Δ=0.05: {drift_005:.3e} energy/period (Exercise 6.3 ≈ -1.4e-8)")

    # Pendulum: Verlet conserves H on a trapped orbit.
    _, qs_p, ps_p = simulate(
        verlet_step, pendulum_T_grad, pendulum_V_grad, 1.0, 0.0, 0.05, 5000
    )
    H_p = 0.5 * ps_p**2 - np.cos(qs_p)
    print(f"  Pendulum Verlet energy band (5000 steps): {np.ptp(H_p):.2e}")


if __name__ == "__main__":
    main()
