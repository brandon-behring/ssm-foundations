"""Chapter 4 (PyTorch companion) — ZOH vs bilinear vs forward Euler on a forced oscillator.

Mirrors ``companions/ch04/jax/discretization_comparison.py`` for the JAX↔PyTorch contrast.
This is the **compute-and-parity** core only: the JAX module remains the sole figure
producer (eigenvalue migration + order convergence plots), so matplotlib / ``save_figure``
are intentionally absent here. The public numerical functions (``discretize_forward_euler``,
``discretize_zoh``, ``discretize_bilinear``, ``step_hold``, ``step_midpoint``, ``simulate``,
``measure_max_error``, ``continuous_reference``) match the JAX signatures and return
semantics exactly, so a parity test can feed identical inputs to both frameworks.

Discretizes the forced 2-state damped oscillator $\\ddot q + c \\dot q + k q = u(t)$ under
the state-space lift $h = (q, \\dot q)^\\top$ via three schemes: forward Euler (first-order),
zero-order hold (ZOH, autonomous-exact), and the bilinear/Tustin transform (second-order).

JAX↔PyTorch contrast
--------------------
* **Matrix exponential.** ``jax.scipy.linalg.expm`` → ``torch.linalg.matrix_exp`` (ZOH).
* **Linear solve.** ``jnp.linalg.solve`` → ``torch.linalg.solve`` (bilinear).
* **Block assembly.** ``jnp.concatenate`` → ``torch.cat`` (ZOH augmented matrix).
* **Recurrence.** The JAX ``simulate`` threads the hidden state through ``jax.lax.scan`` and
  emits each $y_k = C h_k$ in one fused pass. PyTorch has no ``scan`` primitive, so the same
  linear recurrence is a define-by-run Python loop with a NumPy output buffer — bit-for-bit
  identical states, $O(L)$ sequential. (The eager loop *is* the program.)
* **Precision.** JAX enables float64 globally; PyTorch sets it per tensor, so we pass
  ``dtype=torch.float64`` explicitly.

Usage
-----
::

    PYTHONPATH=. python companions/ch04/torch/discretization_comparison.py
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp

_DTYPE = torch.float64

# ---------------------------------------------------------------------------
# Test system: forced damped oscillator (identical float64 constants to the JAX module)
#   \ddot q + c \dot q + k q = u(t),   u(t) = sin(2 t),
# lifted to h = (q, \dot q)^T with A = [[0, 1], [-k, -c]], B = [0, 1]^T, C = [1, 0].
# Eigenvalues -0.25 ± i sqrt(15.75)/2 ≈ -0.25 ± 1.984i sit firmly in the open left half-plane.
# ---------------------------------------------------------------------------

_K_STIFF: float = 4.0  # spring constant
_C_DAMP: float = 0.5  # damping coefficient
_A_MAT: torch.Tensor = torch.tensor([[0.0, 1.0], [-_K_STIFF, -_C_DAMP]], dtype=_DTYPE)
_B_VEC: torch.Tensor = torch.tensor([0.0, 1.0], dtype=_DTYPE)
_C_ROW: torch.Tensor = torch.tensor([1.0, 0.0], dtype=_DTYPE)


def drive(t: np.ndarray | float) -> np.ndarray | float:
    """Smooth scalar input $u(t) = \\sin(2 t)$ (the forced-oscillator driver)."""
    return np.sin(2.0 * t)


# ---------------------------------------------------------------------------
# Discretizers
# ---------------------------------------------------------------------------


def discretize_forward_euler(
    A: torch.Tensor, B: torch.Tensor, dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-Euler discretization: $\\bar A = I + \\Delta A$, $\\bar B = \\Delta B$.

    Only conditionally stable: for fixed $A$ with $\\operatorname{Re}(\\lambda) < 0$ there
    exists $\\Delta_{\\max}$ beyond which $\\rho(\\bar A) > 1$ and the recurrence diverges.
    Included for pedagogical contrast with the A-stable schemes below.

    Raises
    ------
    ValueError
        If ``dt <= 0``.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    A = torch.as_tensor(A, dtype=_DTYPE)
    B = torch.as_tensor(B, dtype=_DTYPE)
    n = A.shape[0]
    Id = torch.eye(n, dtype=_DTYPE)
    return Id + dt * A, dt * B


def discretize_zoh(
    A: torch.Tensor, B: torch.Tensor, dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-order-hold discretization via the augmented matrix exponential.

    Builds the block matrix $M = [[A \\Delta, B \\Delta], [0, 0]]$, computes $\\exp(M)$, and
    reads $\\bar A$ (top-left $N \\times N$ block) and $\\bar B$ (top-right $N \\times P$
    block). This avoids inverting $A$ — see Exercise 4.4.

    Raises
    ------
    ValueError
        If ``dt <= 0``.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    A = torch.as_tensor(A, dtype=_DTYPE)
    B = torch.as_tensor(B, dtype=_DTYPE)
    n = A.shape[0]
    B_col = B[:, None]
    p = B_col.shape[1]
    top = torch.cat([A * dt, B_col * dt], dim=1)
    bottom = torch.zeros((p, n + p), dtype=_DTYPE)
    aug = torch.cat([top, bottom], dim=0)
    exp_aug = torch.linalg.matrix_exp(aug)
    Ad = exp_aug[:n, :n]
    Bd = exp_aug[:n, n:].squeeze(-1)
    return Ad, Bd


def discretize_bilinear(
    A: torch.Tensor, B: torch.Tensor, dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear (Tustin) transform: $\\bar A = (I - \\Delta A/2)^{-1}(I + \\Delta A/2)$.

    Second-order accurate for smooth forcing; A-stable; maps the imaginary axis of the
    continuous plane exactly to the unit circle of the discrete plane. See §4.4.

    Raises
    ------
    ValueError
        If ``dt <= 0``.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    A = torch.as_tensor(A, dtype=_DTYPE)
    B = torch.as_tensor(B, dtype=_DTYPE)
    n = A.shape[0]
    Id = torch.eye(n, dtype=_DTYPE)
    half = 0.5 * dt
    L = Id - half * A
    Ad = torch.linalg.solve(L, Id + half * A)
    Bd = torch.linalg.solve(L, dt * B)
    return Ad, Bd


# ---------------------------------------------------------------------------
# Step functions
#   Forward Euler / ZOH: $h_{k+1} = \bar A h_k + \bar B u_k$ (input held).
#   Bilinear:           $h_{k+1} = \bar A h_k + \bar B (u_k + u_{k+1})/2$ (input midpoint).
# ---------------------------------------------------------------------------


def step_hold(
    Ad: torch.Tensor, Bd: torch.Tensor, h: torch.Tensor, u_k: float, _u_kp1: float
) -> torch.Tensor:
    """Input-hold step (forward Euler and ZOH)."""
    return Ad @ h + Bd * u_k


def step_midpoint(
    Ad: torch.Tensor, Bd: torch.Tensor, h: torch.Tensor, u_k: float, u_kp1: float
) -> torch.Tensor:
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
    dt : float
        Step size; positive.
    t_end : float
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
    Where the JAX companion expresses this recurrence with ``jax.lax.scan`` (pure, fused),
    PyTorch has no ``scan`` primitive, so the carry ``h`` is threaded through an eager Python
    loop and each $y_k = C h_k$ is written into a NumPy buffer. The numerical states are
    identical; only the spelling differs.
    """
    if dt <= 0 or t_end <= 0:
        raise ValueError(f"dt and t_end must both be positive, got dt={dt}, t_end={t_end}")
    Ad, Bd = discretizer(_A_MAT, _B_VEC, dt)
    n_steps = int(round(t_end / dt)) + 1
    ts = np.arange(n_steps) * dt
    u_vals = np.sin(2.0 * ts)  # drive(t) = sin(2t)
    h = torch.zeros(2, dtype=_DTYPE)
    ys = np.zeros(n_steps)
    for k in range(n_steps - 1):
        ys[k] = float((_C_ROW @ h).item())
        h = stepper(Ad, Bd, h, float(u_vals[k]), float(u_vals[k + 1]))
    ys[-1] = float((_C_ROW @ h).item())
    return ts, ys


def continuous_reference(t_end: float, t_grid: np.ndarray) -> np.ndarray:
    """High-accuracy reference solution from `scipy.integrate.solve_ivp` (Radau).

    Returns the output $y(t) = C h(t)$ sampled at ``t_grid``. Radau is an A-stable implicit
    method with adaptive step control — accurate enough that its residual on our 4-second
    horizon is below ``1e-10``.
    """

    # Convert the torch constants to NumPy once (clean for CPU float64) so the per-step
    # RHS does not re-trigger torch's __array__ on every solve_ivp evaluation.
    A_np = _A_MAT.numpy()
    B_np = _B_VEC.numpy()
    C_np = _C_ROW.numpy()

    def rhs(t: float, h: np.ndarray) -> np.ndarray:
        return A_np @ h + B_np * drive(t)

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
    return C_np @ sol.y  # shape (n_grid,)


def measure_max_error(discretizer, stepper, dt: float, t_end: float = 4.0) -> float:
    """Max pointwise output error of a scheme against the Radau reference."""
    ts, ys = simulate(discretizer, stepper, dt, t_end)
    y_ref = continuous_reference(t_end, ts)
    return float(np.max(np.abs(ys - y_ref)))


def main() -> None:
    """Print eigenvalue + empirical-order summaries (no plotting — JAX owns the figures)."""
    print("Chapter 4 (torch) — discretization_comparison.py")
    print("=" * 60)
    cont_eigs = np.linalg.eigvals(_A_MAT.numpy())
    print(f"Continuous eigenvalues: {np.sort_complex(cont_eigs)}")
    discs = {
        "Forward Euler": discretize_forward_euler(_A_MAT, _B_VEC, 0.1)[0],
        "ZOH": discretize_zoh(_A_MAT, _B_VEC, 0.1)[0],
        "Bilinear": discretize_bilinear(_A_MAT, _B_VEC, 0.1)[0],
    }
    print("\nDiscrete spectral radii at Δ=0.1:")
    for name, M in discs.items():
        rho = float(np.max(np.abs(np.linalg.eigvals(M.numpy()))))
        print(f"  {name:<14s} ρ(Ā) = {rho:.4f}")

    dts = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
    schemes = {
        "Forward Euler": (discretize_forward_euler, step_hold),
        "ZOH": (discretize_zoh, step_hold),
        "Bilinear": (discretize_bilinear, step_midpoint),
    }
    print("\nEmpirical order (slope on finest two Δ):")
    print("-" * 50)
    for name, (disc_fn, step_fn) in schemes.items():
        errs = np.array([measure_max_error(disc_fn, step_fn, float(dt)) for dt in dts])
        slope = np.log(errs[-2] / errs[-1]) / np.log(dts[-2] / dts[-1])
        print(f"  {name:<14s} slope ≈ {slope:5.2f}")


if __name__ == "__main__":
    main()
