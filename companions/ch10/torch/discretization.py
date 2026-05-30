r"""Chapter 10 torch companion — discretization order + stability (Mamba-3 integrator).

Mirrors ``companions/ch10/jax/discretization.py`` in PyTorch. The discretizers are
pure elementwise functions of ``(A, dt)``; the forced integration is an eager loop
(torch has no parallel scan — the same loop also serves as the cross-framework
oracle). float64 throughout so a slope-2 convergence study is not masked by
roundoff, and so parity against the JAX companion is meaningful.

The §10.2 subtlety is identical to the JAX module: the transition $\alpha =
e^{A\Delta}$ is exact, so on a homogeneous system ZOH and exp-trapezoidal coincide
and the order is invisible; the order-2 gain is only measurable on a forced
system. See ``tests/test_mamba3_torch.py`` for the parity pins.

Port credit
-----------
Mirrors ``companions/ch10/jax/discretization.py``; Mamba-3: Lahoti et al.,
arXiv:2603.15569.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "discretize_zoh",
    "discretize_bilinear",
    "discretize_exp_trapezoidal",
    "forced_exact",
    "integrate",
    "global_error",
    "order_sweep",
    "amplification",
]

# Match JAX float64 so cross-framework parity is meaningful.
torch.set_default_dtype(torch.float64)


def _as_tensor(A) -> Tensor:
    if isinstance(A, Tensor):
        return A
    if isinstance(A, complex):
        return torch.tensor(A, dtype=torch.complex128)
    return torch.tensor(A, dtype=torch.float64)


def discretize_zoh(A, dt: float) -> tuple[Tensor, Tensor]:
    r"""Zero-order hold (first-order): ``alpha = e^{A dt}``, ``beta = (alpha-1)/A``."""
    A = _as_tensor(A)
    alpha = torch.exp(A * dt)
    if torch.abs(A) < 1e-12:
        beta = torch.as_tensor(dt, dtype=alpha.dtype)
    else:
        beta = (alpha - 1.0) / A
    return alpha, beta


def discretize_bilinear(A, dt: float) -> tuple[Tensor, Tensor]:
    r"""Bilinear / Tustin (second-order, A-stable). ``x_k = alpha x_{k-1} + beta (u_{k-1}+u_k)``."""
    A = _as_tensor(A)
    half = dt / 2.0
    denom = 1.0 - half * A
    alpha = (1.0 + half * A) / denom
    beta = half / denom
    return alpha, beta


def discretize_exp_trapezoidal(A, dt: float, lam: float = 0.5) -> tuple[Tensor, Tensor, Tensor]:
    r"""Exponential-trapezoidal (second-order). ``x_k = alpha x_{k-1} + beta u_{k-1} + gamma u_k``.

    $\alpha = e^{A\Delta}$, $\beta = (1-\lambda)\Delta\alpha$, $\gamma = \lambda\Delta$.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    A = _as_tensor(A)
    alpha = torch.exp(A * dt)
    beta = (1.0 - lam) * dt * alpha
    gamma = torch.as_tensor(lam * dt, dtype=alpha.dtype)
    return alpha, beta, gamma


def forced_exact(A, omega: float, t: Tensor, x0=0.0) -> Tensor:
    r"""Exact solution of $x' = A x + \sin(\omega t)$, $x(0)=x_0$ (variation of constants)."""
    A = _as_tensor(A)
    t = t.to(A.dtype) if torch.is_complex(A) else t.to(torch.float64)
    denom = A * A + omega * omega
    c = x0 + omega / denom
    return c * torch.exp(A * t) - (A * torch.sin(omega * t) + omega * torch.cos(omega * t)) / denom


def integrate(scheme: str, A, dt: float, n_steps: int, omega, x0=0.0, lam: float = 0.5) -> Tensor:
    r"""Integrate $x' = A x + u(t)$ for ``n_steps`` eager steps. ``omega=None`` -> homogeneous."""
    A = _as_tensor(A)
    cdtype = torch.complex128 if torch.is_complex(A) else torch.float64
    ks = torch.arange(n_steps + 1, dtype=torch.float64)
    t = ks * dt
    if omega is None:
        u_grid = torch.zeros(n_steps + 1, dtype=torch.float64)
    else:
        u_grid = torch.sin(omega * t)

    x = torch.as_tensor(x0, dtype=cdtype)
    xs = [x]
    if scheme == "zoh":
        alpha, beta = discretize_zoh(A, dt)
        for k in range(1, n_steps + 1):
            x = alpha * x + beta * u_grid[k - 1]
            xs.append(x)
    elif scheme == "bilinear":
        alpha, beta = discretize_bilinear(A, dt)
        for k in range(1, n_steps + 1):
            x = alpha * x + beta * (u_grid[k - 1] + u_grid[k])
            xs.append(x)
    elif scheme == "exp_trapezoidal":
        alpha, beta, gamma = discretize_exp_trapezoidal(A, dt, lam=lam)
        for k in range(1, n_steps + 1):
            x = alpha * x + beta * u_grid[k - 1] + gamma * u_grid[k]
            xs.append(x)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    return torch.stack(xs)


def global_error(scheme: str, A, dt: float, T: float, omega, x0=0.0, lam: float = 0.5) -> float:
    """Max absolute error over [0, T] vs the exact solution."""
    n_steps = int(round(T / dt))
    xs = integrate(scheme, A, dt, n_steps, omega, x0=x0, lam=lam)
    t = torch.arange(n_steps + 1, dtype=torch.float64) * dt
    if omega is None:
        exact = torch.as_tensor(x0, dtype=xs.dtype) * torch.exp(_as_tensor(A) * t)
    else:
        exact = forced_exact(A, omega, t, x0=x0)
    return float(torch.max(torch.abs(xs - exact)))


def order_sweep(scheme: str, A, dts, T: float, omega, x0=0.0, lam: float = 0.5) -> tuple[list[float], float]:
    """Errors at each step size + the fitted convergence slope (the empirical order)."""
    import math

    errors = [global_error(scheme, A, float(dt), T, omega, x0=x0, lam=lam) for dt in dts]
    slope = (math.log(errors[-1]) - math.log(errors[-2])) / (math.log(float(dts[-1])) - math.log(float(dts[-2])))
    return errors, slope


def amplification(scheme: str, z: Tensor) -> Tensor:
    r"""Amplification factor $\alpha(z)$, $z = A\Delta$ (see the JAX docstring)."""
    if scheme in ("zoh", "exp_trapezoidal"):
        return torch.exp(z)
    if scheme == "bilinear":
        return (1.0 + z / 2.0) / (1.0 - z / 2.0)
    if scheme == "forward_euler":
        return 1.0 + z
    raise ValueError(f"unknown scheme {scheme!r}")
