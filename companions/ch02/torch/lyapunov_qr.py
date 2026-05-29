"""Chapter 2 (PyTorch companion) — QR-based Lyapunov spectrum.

Mirrors ``companions/ch02/jax/lyapunov_qr.py`` for the JAX↔PyTorch comparison
(Phase 9 notebooks). Same algorithm (Benettin et al. 1980 QR re-orthonormalization),
same ring-of-oscillators test system as §2.3.

JAX↔PyTorch contrast
--------------------
* **The Benettin iteration is the canonical scan-vs-loop case.** JAX expresses it as
  ``jax.lax.scan`` with a matrix-valued carry ``(Q, log_diag_sum)`` and a *pre-tiled*
  Jacobian schedule as the scan input. PyTorch runs the same re-orthonormalization as
  an eager Python loop, indexing ``jacobians[t % T]`` inline — define-by-run, no scan.
* **Mutable vs functional state.** The ring state matrix is built with in-place
  ``A[i, j] = ...`` assignment (PyTorch tensors are mutable, like NumPy); the JAX
  companion needs a functional ``jnp.zeros(...).at[...].add(...)`` scatter instead.
* **Native matrix exp / eig.** ``torch.linalg.matrix_exp`` and ``torch.linalg.eigvals``
  replace the JAX companion's ``scipy.linalg.expm`` and ``jnp.linalg.eigvals``.

Usage
-----
::

    PYTHONPATH=. python companions/ch02/torch/lyapunov_qr.py
"""

from __future__ import annotations

import numpy as np
import torch

_DTYPE = torch.float64


def ring_state_matrix(n: int, k: float = 4.0, c: float = 0.2, kappa: float = 1.0) -> torch.Tensor:
    """Ring-of-damped-oscillators $2n \\times 2n$ state matrix (mutable in-place build).

    Contrast with the JAX companion's functional ``.at[].add`` scatter: PyTorch
    tensors are mutable, so the natural spelling is NumPy-style element assignment.

    Raises
    ------
    ValueError
        If ``n < 3``, ``k <= 0``, ``c < 0``, or ``kappa < 0``.
    """
    if n < 3:
        raise ValueError(f"need at least 3 oscillators, got n={n}")
    if k <= 0:
        raise ValueError(f"stiffness k must be positive, got {k}")
    if c < 0 or kappa < 0:
        raise ValueError(f"damping and coupling must be non-negative, got c={c}, kappa={kappa}")
    A = torch.zeros((2 * n, 2 * n), dtype=_DTYPE)
    for i in range(n):
        q_idx, v_idx = 2 * i, 2 * i + 1
        A[q_idx, v_idx] = 1.0
        A[v_idx, q_idx] = -k - 2.0 * kappa
        A[v_idx, v_idx] = -c
        A[v_idx, 2 * ((i - 1) % n)] = kappa
        A[v_idx, 2 * ((i + 1) % n)] = kappa
    return A


def qr_lyapunov(jacobians: torch.Tensor | np.ndarray, n_steps: int) -> np.ndarray:
    """Lyapunov spectrum via the Benettin QR algorithm (eager loop).

    PyTorch contrast: the JAX companion fuses this recurrence with ``jax.lax.scan``;
    here it is a define-by-run ``for`` loop with ``torch.linalg.qr`` and inline
    ``jacobians[t % T]`` cyclic indexing.

    Parameters
    ----------
    jacobians : tensor of shape (T, N, N)
        Per-step Jacobians, reused cyclically if ``T < n_steps``.
    n_steps : int
        Number of iterations.

    Returns
    -------
    ndarray of shape (N,)
        Lyapunov exponents, sorted descending.

    Raises
    ------
    ValueError
        If ``jacobians`` is not (T, N, N) or ``n_steps < 1``.
    """
    jacobians = torch.as_tensor(jacobians, dtype=_DTYPE)
    if jacobians.ndim != 3 or jacobians.shape[1] != jacobians.shape[2]:
        raise ValueError(f"jacobians must have shape (T, N, N), got {tuple(jacobians.shape)}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    T, N, _ = jacobians.shape
    Q = torch.eye(N, dtype=_DTYPE)
    acc = torch.zeros(N, dtype=_DTYPE)
    for t in range(n_steps):
        Q, R = torch.linalg.qr(jacobians[t % T] @ Q)
        signs = torch.sign(torch.diag(R))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        Q = Q * signs.unsqueeze(0)
        R = signs.unsqueeze(1) * R
        acc = acc + torch.log(torch.abs(torch.diag(R)) + 1e-300)
    return torch.sort(acc / n_steps, descending=True).values.numpy()


def autonomous_lyapunov_reference(A: torch.Tensor | np.ndarray) -> np.ndarray:
    """Closed-form spectrum $\\Re(\\text{eigvals}(A))$, sorted descending."""
    A = torch.as_tensor(A, dtype=_DTYPE)
    return np.sort(torch.linalg.eigvals(A).real.numpy())[::-1]


def main() -> None:
    print("Chapter 2 (torch) — lyapunov_qr.py")
    print("=" * 60)
    dt, n_steps = 0.05, 2000
    A = ring_state_matrix(n=8, c=0.2)
    J = torch.linalg.matrix_exp(A * dt)
    spec = qr_lyapunov(J.unsqueeze(0), n_steps) / dt
    print(f"  damped ring: max λ = {spec.max():.4f} (should be < 0), Σλ = {spec.sum():.4f}")


if __name__ == "__main__":
    main()
