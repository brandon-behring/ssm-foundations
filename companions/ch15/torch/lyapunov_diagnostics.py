r"""Chapter 15 torch companion — the Lyapunov diagnostic core (parity mirror).

Mirrors the JAX module's engine (``companions/ch15/jax/lyapunov_diagnostics.py``):
the Benettin QR Lyapunov estimator, the closed-form spectrum, and the effective
state size. Eager, float64, in-process parity against JAX ``< 1e-9`` (pinned in the
shared torch test file). The instrument must be framework-agnostic because pilot B
runs it on trained *torch* models — this mirror is that guarantee.

Port credit
-----------
Mirrors the JAX module; the QR engine mirrors Chapter 2's ``qr_lyapunov``.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "qr_lyapunov",
    "closed_form_log_growth",
    "log_det_rate",
    "effective_state_size",
    "marginal_mode_count",
]

torch.set_default_dtype(torch.float64)


def qr_lyapunov(jacobians: Tensor, n_steps: int) -> Tensor:
    r"""Benettin QR Lyapunov spectrum (descending) of a per-step Jacobian sequence.

    ``jacobians`` is ``(T, N, N)`` (cycled if ``T < n_steps``); mirrors the JAX core.
    """
    if jacobians.ndim != 3 or jacobians.shape[1] != jacobians.shape[2]:
        raise ValueError(f"jacobians must be (T, N, N); got {tuple(jacobians.shape)}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1; got {n_steps}")
    T, N, _ = jacobians.shape
    Q = torch.eye(N)
    acc = torch.zeros(N)
    for t in range(n_steps):
        Jt = jacobians[t % T]
        Qn, R = torch.linalg.qr(Jt @ Q)
        signs = torch.sign(torch.diag(R))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        Qn = Qn * signs.unsqueeze(0)
        R = signs.unsqueeze(1) * R
        acc = acc + torch.log(torch.abs(torch.diag(R)) + 1e-300)
        Q = Qn
    return torch.sort(acc / n_steps, descending=True).values


def closed_form_log_growth(J: Tensor) -> Tensor:
    r"""Closed-form Lyapunov spectrum of an autonomous system: $\log|\lambda_i(J)|$ descending."""
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(f"J must be square (N, N); got {tuple(J.shape)}")
    mags = torch.abs(torch.linalg.eigvals(J))
    return torch.sort(torch.log(mags), descending=True).values


def log_det_rate(jacobians: Tensor) -> float:
    r"""The divergence identity's right side $\langle \log|\det J_t|\rangle$."""
    if jacobians.ndim != 3 or jacobians.shape[1] != jacobians.shape[2]:
        raise ValueError(f"jacobians must be (T, N, N); got {tuple(jacobians.shape)}")
    return float(torch.linalg.slogdet(jacobians).logabsdet.mean())


def effective_state_size(magnitudes: Tensor) -> float:
    r"""Participation ratio of $|\lambda_i|^2$: $(\sum p)^2 / \sum p^2$."""
    p = magnitudes.to(torch.float64) ** 2
    denom = float((p**2).sum())
    if denom == 0.0:
        raise ValueError("all magnitudes are zero; effective state size undefined")
    return float(p.sum() ** 2 / denom)


def marginal_mode_count(values: Tensor, tol: float, mode: str = "magnitude") -> int:
    r"""Count marginal modes: $|\lambda_i| \ge 1-\delta$ (magnitude) or $\lambda_i \ge -\delta$ (exponent)."""
    if tol < 0.0:
        raise ValueError(f"tol must be >= 0; got {tol}")
    if mode == "magnitude":
        return int(torch.sum(torch.abs(values) >= 1.0 - tol))
    if mode == "exponent":
        return int(torch.sum(values >= -tol))
    raise ValueError(f"mode must be 'magnitude' or 'exponent'; got {mode!r}")
