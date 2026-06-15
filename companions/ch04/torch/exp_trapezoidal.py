"""Chapter 4 (PyTorch companion) — exponential-trapezoidal scheme on the forced oscillator.

Mirrors ``companions/ch04/jax/exp_trapezoidal.py`` for the JAX↔PyTorch contrast. This is
the **compute-and-parity** core only: the JAX module remains the sole figure producer, so
matplotlib / ``save_figure`` are intentionally absent here. The public numerical functions
(``discretize_exp_trap``, ``simulate_exp_trap``, ``simulate_zoh``, ``simulate_bilinear``,
``continuous_reference``) match the JAX signatures and return semantics exactly, so a parity
test can feed identical inputs to both frameworks.

Implements the second-order exponential-trapezoidal integrator
$$h_{k+1} = e^{A \\Delta} h_k + \\Delta \\, \\varphi_1(A \\Delta) B u_k
    + \\Delta \\, \\varphi_2(A \\Delta) B (u_{k+1} - u_k),$$
where $\\varphi_1(z) = (e^z - 1)/z$ and $\\varphi_2(z) = (e^z - 1 - z)/z^2$ are the first two
$\\varphi$-functions, computed via the **augmented matrix exponential** trick
(Van Loan 1978 for the block identity; Al-Mohy & Higham 2011 for the
simultaneous-$\\varphi$ formulation) — see §4.5 and Exercise 4.6.

JAX↔PyTorch contrast
--------------------
* **Matrix exponential.** The JAX companion calls ``jax.scipy.linalg.expm``; PyTorch has a
  native ``torch.linalg.matrix_exp``.
* **Linear solve.** ``jnp.linalg.solve`` → ``torch.linalg.solve`` (bilinear scheme).
* **Functional updates.** JAX builds the augmented matrix with ``arr.at[i, j].set(...)``
  (immutable). PyTorch tensors are mutable, so we allocate a zero tensor and assign slices
  in place — the define-by-run spelling.
* **Block assembly.** ``jnp.block([...])`` (ZOH augmented matrix) → ``torch.cat`` / zero
  blocks.
* **Recurrence.** Both modules already run the time-stepping as eager Python loops with a
  NumPy output buffer (JAX did not use ``lax.scan`` here), so the loop ports verbatim.
* **Precision.** JAX enables float64 globally; PyTorch sets it per tensor, so we pass
  ``dtype=torch.float64`` explicitly.

Usage
-----
::

    PYTHONPATH=. python companions/ch04/torch/exp_trapezoidal.py
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp

_DTYPE = torch.float64

# Same forced damped oscillator as discretization_comparison.py (identical float64 values).
_A_MAT: torch.Tensor = torch.tensor([[0.0, 1.0], [-4.0, -0.5]], dtype=_DTYPE)
_B_VEC: torch.Tensor = torch.tensor([0.0, 1.0], dtype=_DTYPE)
_C_ROW: torch.Tensor = torch.tensor([1.0, 0.0], dtype=_DTYPE)


def drive(t: np.ndarray | float) -> np.ndarray | float:
    """$u(t) = \\sin(2 t)$ — the same smooth scalar input as §4.3 / §4.4."""
    return np.sin(2.0 * t)


def discretize_exp_trap(
    A: torch.Tensor, B: torch.Tensor, dt: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exponential-trapezoidal discretization via augmented matrix exponential.

    Builds the $(N + 2) \\times (N + 2)$ block matrix
    $$
    M = \\begin{pmatrix} A \\Delta & B \\Delta & 0 \\\\ 0 & 0 & \\Delta \\\\ 0 & 0 & 0 \\end{pmatrix},
    $$
    whose exponential has top-row blocks
    $$
    \\exp(M)_{0:N} = \\bigl(e^{A \\Delta}, \\; \\Delta \\varphi_1(A \\Delta) B, \\; \\Delta^2 \\varphi_2(A \\Delta) B\\bigr).
    $$
    The last block carries the $\\Delta^2$ factor, which we divide back out to return
    $\\Delta \\varphi_2(A \\Delta) B$ as required by the scheme of §4.5.

    Parameters
    ----------
    A : torch.Tensor, shape (N, N)
        Continuous dynamics matrix.
    B : torch.Tensor, shape (N,)
        Continuous input vector.
    dt : float
        Step size; positive.

    Returns
    -------
    Ad : torch.Tensor, shape (N, N)
        Discrete dynamics matrix $e^{A \\Delta}$.
    B0 : torch.Tensor, shape (N,)
        Coefficient of $u_k$: $\\Delta \\varphi_1(A \\Delta) B$.
    B1 : torch.Tensor, shape (N,)
        Coefficient of $(u_{k+1} - u_k)$: $\\Delta \\varphi_2(A \\Delta) B$.

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
    # Construct M = dt · Â where Â is the augmented matrix
    #     Â = [[A, B, 0], [0, 0, 1], [0, 0, 0]]
    # so M = [[A·dt, B·dt, 0], [0, 0, dt], [0, 0, 0]] (shape (n+2, n+2)).
    # The (n, n+1) entry MUST be `dt`, not `1`: solving dy/dt = Â y with
    # y(0) = e_{n+2} gives y_3(t) = 1, y_2(t) = t, y_1(t) = t² φ_2(t A) B —
    # which is exactly what we need exp(dt·Â) to evaluate at t = dt.
    # PyTorch tensors are mutable, so we assign slices in place (vs JAX's `.at[...].set`).
    M = torch.zeros((n + 2, n + 2), dtype=_DTYPE)
    M[:n, :n] = A * dt
    M[:n, n] = B * dt
    M[n, n + 1] = dt
    EM = torch.linalg.matrix_exp(M)
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
    h = torch.zeros(2, dtype=_DTYPE)
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
    # Augmented matrix [[A·dt, B·dt], [0, 0]] assembled with torch.cat (vs jnp.block).
    top = torch.cat([_A_MAT * dt, (_B_VEC[:, None]) * dt], dim=1)
    bottom = torch.zeros((1, n + 1), dtype=_DTYPE)
    aug = torch.cat([top, bottom], dim=0)
    EX = torch.linalg.matrix_exp(aug)
    Ad = EX[:n, :n]
    Bd = EX[:n, n]
    n_steps = int(round(t_end / dt)) + 1
    h = torch.zeros(2, dtype=_DTYPE)
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
    Id = torch.eye(n, dtype=_DTYPE)
    half = 0.5 * dt
    L = Id - half * _A_MAT
    Ad = torch.linalg.solve(L, Id + half * _A_MAT)
    Bd = torch.linalg.solve(L, dt * _B_VEC)
    n_steps = int(round(t_end / dt)) + 1
    h = torch.zeros(2, dtype=_DTYPE)
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
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return C_np @ sol.y


def main() -> None:
    """Print an empirical-convergence summary (no plotting — JAX owns the figure)."""
    print("Chapter 4 (torch) — exp_trapezoidal.py")
    print("=" * 60)
    dts = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
    t_end = 4.0
    schemes = {
        "ZOH": simulate_zoh,
        "Bilinear (trap)": simulate_bilinear,
        "Exp-trapezoidal": simulate_exp_trap,
    }
    print("Empirical convergence orders (slope on finest two Δ):")
    print("-" * 60)
    for name, sim_fn in schemes.items():
        errs = np.zeros(len(dts))
        for i, dt in enumerate(dts):
            ts, ys = sim_fn(float(dt), t_end)
            y_ref = continuous_reference(t_end, ts)
            errs[i] = float(np.max(np.abs(ys - y_ref)))
        slope = np.log(errs[-2] / errs[-1]) / np.log(dts[-2] / dts[-1])
        print(f"  {name:<18s} slope ≈ {slope:5.2f}   (err at Δ=0.025: {errs[-1]:.2e})")


if __name__ == "__main__":
    main()
