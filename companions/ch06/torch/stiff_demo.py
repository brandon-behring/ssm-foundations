"""Chapter 6 (PyTorch companion) — stiff systems: explicit RK4 versus backward Euler.

Mirrors ``companions/ch06/jax/stiff_demo.py`` for the JAX↔PyTorch contrast (this
is the **compute + parity** half of the Ch 6 companion — no figures). Same
pedagogy as §6.1: on a stiff ODE an explicit Runge-Kutta method needs step sizes
inversely proportional to the largest eigenvalue magnitude (here the van der Pol
stiffness $\\mu$), while an implicit method like backward Euler stays stable for
*every* positive step size.

Test problem: van der Pol oscillator
$$\\dot q = p, \\qquad \\dot p = \\mu (1 - q^2) p - q,$$
with $\\mu = 10$ (mildly stiff but well-behaved on float64).

JAX↔PyTorch contrast
--------------------
* **Jacobian.** The JAX companion differentiates ``vdp_rhs`` with ``jax.jacfwd``
  (the teaching point there). The hand-coded Jacobian is mathematically identical
  (the JAX test pins ``jacfwd`` == analytic at ``atol=1e-12``), so this port uses
  the closed-form :func:`vdp_jacobian` directly — fewer moving parts, and bit-for-bit
  with the JAX values the parity tests demand.
* **Implicit solve.** ``jnp.linalg.solve`` → ``torch.linalg.solve`` for the
  per-iteration Newton step $[I - \\Delta J]\\,\\delta = g$.
* **Time loop / control flow.** JAX uses ``jax.lax.scan`` over steps and
  ``jax.lax.while_loop`` for the damped-Newton iteration; PyTorch is define-by-run,
  so both are plain eager Python loops. The damped Newton still iterates *to
  tolerance* (not a fixed count), because a fixed count under-converges during the
  van der Pol fast jumps at coarse dt.
* **Divergence masking.** ``jnp.where(|h| > 1e8, nan)`` → a NumPy ``np.where`` on
  the assembled trajectory: explicit-RK blowup produces ``inf``/``nan`` which we
  mask to ``nan`` so a plot would truncate at the same visual point.
* **Precision.** float64 throughout (per-tensor in PyTorch, vs JAX's global flag).

Usage
-----
::

    PYTHONPATH=. python companions/ch06/torch/stiff_demo.py
"""

from __future__ import annotations

import numpy as np
import torch

_DTYPE = torch.float64

_MU: float = 10.0  # Stiffness parameter (mild; visible on float64).
_BLOWUP_THRESHOLD: float = 1e8  # |state| above this is treated as RK4 divergence.
_NEWTON_TOL: float = 1e-10  # backward-Euler Newton residual tolerance.
_NEWTON_MAX_ITER: int = 60  # safeguard cap on damped-Newton iterations.
# Backtracking step fractions (1, 1/2, ..., ~3e-5) for the line search: at a
# coarse dt the forward-Euler warm start sits outside the pure-Newton basin
# during the van der Pol fast jumps, so each iteration takes the
# residual-minimizing fraction of the Newton step rather than the full step.
_NEWTON_DAMPING = torch.tensor([0.5**k for k in range(16)], dtype=_DTYPE)


def vdp_rhs(h: torch.Tensor) -> torch.Tensor:
    """Van der Pol right-hand side $f(q, p) = (p, \\mu(1-q^2)p - q)$."""
    q, p = h[0], h[1]
    return torch.stack([p, _MU * (1.0 - q * q) * p - q])


def vdp_jacobian(h: torch.Tensor) -> torch.Tensor:
    """Analytic van der Pol Jacobian $\\partial f/\\partial h$.

    Identical to the JAX companion's ``jax.jacfwd(vdp_rhs)`` (the JAX test pins
    them equal at ``atol=1e-12``); used here directly so the implicit solve is
    bit-for-bit with the JAX values.
    """
    q, p = h[0], h[1]
    zero = torch.zeros((), dtype=_DTYPE)
    one = torch.ones((), dtype=_DTYPE)
    row0 = torch.stack([zero, one])
    row1 = torch.stack([-2.0 * _MU * q * p - one, _MU * (1.0 - q * q)])
    return torch.stack([row0, row1])


# ---------------------------------------------------------------------------
# Explicit RK4
# ---------------------------------------------------------------------------


def rk4_step(h: torch.Tensor, dt: float) -> torch.Tensor:
    """One step of classical Runge-Kutta 4 on the van der Pol system (autonomous)."""
    k1 = vdp_rhs(h)
    k2 = vdp_rhs(h + 0.5 * dt * k1)
    k3 = vdp_rhs(h + 0.5 * dt * k2)
    k4 = vdp_rhs(h + dt * k3)
    return h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# Backward Euler — damped Newton on g(x) = x - h - dt f(x) = 0, iterated to tol
# ---------------------------------------------------------------------------


def backward_euler_step(
    h: torch.Tensor, dt: float, tol: float = _NEWTON_TOL, max_iter: int = _NEWTON_MAX_ITER
) -> torch.Tensor:
    """Backward Euler via a damped Newton solve iterated to tolerance.

    Solves $g(x) = x - h - \\Delta f(x) = 0$ with damped Newton steps
    $x \\leftarrow x - \\alpha\\,[I - \\Delta J(x)]^{-1} g(x)$ from a forward-Euler
    warm start. Each iteration runs a backtracking line search — it evaluates the
    residual at a fixed set of step fractions $\\alpha$ (``_NEWTON_DAMPING``) and
    keeps the smallest-residual candidate (``argmin``). PyTorch contrast: the JAX
    companion uses ``jax.lax.while_loop`` + ``jax.vmap``; here both are eager loops.
    The iteration count is adaptive (loop until $\\|g\\| < \\text{tol}$, capped at
    ``max_iter``) because a fixed count under-converges during the van der Pol fast
    jumps at coarse dt.
    """
    n = h.shape[0]
    Id = torch.eye(n, dtype=_DTYPE)

    def resnorm(x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(x - h - dt * vdp_rhs(x))

    x = h + dt * vdp_rhs(h)  # forward-Euler warm start
    it = 0
    while bool(resnorm(x) > tol) and it < max_iter:
        delta = torch.linalg.solve(Id - dt * vdp_jacobian(x), x - h - dt * vdp_rhs(x))
        candidates = x[None, :] - _NEWTON_DAMPING[:, None] * delta[None, :]  # (n_alpha, n)
        res = torch.stack([resnorm(c) for c in candidates])
        x = candidates[int(torch.argmin(res))]
        it += 1
    return x


# ---------------------------------------------------------------------------
# Simulate up to t_end and return trajectories (eager time stepping)
# ---------------------------------------------------------------------------


def simulate_rk4(h0: np.ndarray, dt: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    """Explicit RK4 trajectory; diverged tail is masked to NaN. Raises on dt/t_end <= 0."""
    if dt <= 0 or t_end <= 0:
        raise ValueError(f"dt and t_end must be positive, got dt={dt}, t_end={t_end}")
    n_steps = int(round(t_end / dt)) + 1
    h = torch.as_tensor(h0, dtype=_DTYPE)
    hs = [h.clone()]
    for _ in range(n_steps - 1):
        h = rk4_step(h, dt)
        hs.append(h.clone())
    stacked = torch.stack(hs).numpy()
    # Explicit-RK divergence -> sentinel to NaN so a plot truncates (replaces
    # NumPy's seterr/try-except; |nan| > thr is False so NaNs stay NaN).
    masked = np.where(np.abs(stacked) > _BLOWUP_THRESHOLD, np.nan, stacked)
    return np.arange(n_steps) * dt, masked


def simulate_be(h0: np.ndarray, dt: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    """Backward-Euler trajectory (stable at any dt). Raises on dt/t_end <= 0."""
    if dt <= 0 or t_end <= 0:
        raise ValueError(f"dt and t_end must be positive, got dt={dt}, t_end={t_end}")
    n_steps = int(round(t_end / dt)) + 1
    h = torch.as_tensor(h0, dtype=_DTYPE)
    hs = [h.clone()]
    for _ in range(n_steps - 1):
        h = backward_euler_step(h, dt)
        hs.append(h.clone())
    return np.arange(n_steps) * dt, torch.stack(hs).numpy()


def main() -> None:
    print("Chapter 6 (torch) — stiff_demo.py")
    print("=" * 60)
    h0 = np.array([2.0, 0.0])
    t_end = 50.0  # ~2.5 limit-cycle periods at mu=10

    for dt in (0.005, 0.05, 0.2):
        _, hs_rk4 = simulate_rk4(h0, dt, t_end)
        _, hs_be = simulate_be(h0, dt, t_end)
        rk4_diverged = bool(np.any(np.isnan(hs_rk4)))
        be_qmax = float(np.max(np.abs(hs_be[:, 0])))
        print(
            f"  Δ={dt:<6g} RK4 diverged: {str(rk4_diverged):5s} | "
            f"BE bounded (max|q|={be_qmax:.3f})"
        )


if __name__ == "__main__":
    main()
