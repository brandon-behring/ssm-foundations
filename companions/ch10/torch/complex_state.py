r"""Chapter 10 torch companion — complex state + RoPE equivalence.

Mirrors ``companions/ch10/jax/complex_state.py`` in PyTorch. Demonstrates that a
complex diagonal mode $A = \log\rho + i\theta$ both decays and rotates, and that
the production realization — a real 2-D RoPE rotation on $B, C$ — is identical to
complex multiplication. The ``ComplexMode`` ``nn.Module`` shows the
buffers-vs-Parameters distinction: $\log\rho$ and $\theta$ are learned
``nn.Parameter``s.

Port credit
-----------
Mirrors ``companions/ch10/jax/complex_state.py``; Mamba-3: Lahoti et al.,
arXiv:2603.15569.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = [
    "complex_scalar_recurrence",
    "rope_matrix",
    "complex_to_real2",
    "rope_equivalence_residual",
    "decay_rate",
    "ComplexMode",
]

torch.set_default_dtype(torch.float64)


def complex_scalar_recurrence(rho: float, theta: float, n_steps: int, x0: complex = 1.0 + 0.0j) -> Tensor:
    r"""Homogeneous complex trajectory $x_k = (\rho e^{i\theta})^k x_0$ (complex128)."""
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must be in (0, 1], got {rho}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}")
    alpha = rho * torch.exp(1j * torch.tensor(theta, dtype=torch.float64))
    ks = torch.arange(n_steps + 1, dtype=torch.float64)
    return torch.as_tensor(x0, dtype=torch.complex128) * alpha**ks


def rope_matrix(theta: float) -> Tensor:
    r"""The $2\times2$ rotation $R(\theta)$ — the real form of $e^{i\theta}$."""
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)


def complex_to_real2(z: Tensor) -> Tensor:
    r"""Stack a complex tensor as $[\operatorname{Re}, \operatorname{Im}]$ on a new last axis."""
    return torch.stack([z.real, z.imag], dim=-1)


def rope_equivalence_residual(rho: float, theta: float, drive: Tensor) -> float:
    r"""Max discrepancy between the complex recurrence and the real 2-D RoPE recurrence.

    Both compute $x_k = (\rho e^{i\theta}) x_{k-1} + d_k$; the real version uses
    $\rho R(\theta)$ on $[\operatorname{Re}, \operatorname{Im}]$. Algebraically
    identical (`ch10:rope-complex-equivalence`), so the residual is at roundoff.
    """
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must be in (0, 1], got {rho}")
    if drive.ndim != 1:
        raise ValueError(f"drive must be 1-D (L,), got {tuple(drive.shape)}")

    alpha = rho * torch.exp(1j * torch.tensor(theta, dtype=torch.float64))
    drive = drive.to(torch.complex128)

    x = torch.zeros((), dtype=torch.complex128)
    xs_complex = []
    for d in drive:
        x = alpha * x + d
        xs_complex.append(x)
    xs_complex = torch.stack(xs_complex)

    R = rho * rope_matrix(theta)
    drive_real = complex_to_real2(drive)  # (L, 2)
    s = torch.zeros(2, dtype=torch.float64)
    ss = []
    for d2 in drive_real:
        s = R @ s + d2
        ss.append(s)
    ss = torch.stack(ss)  # (L, 2)
    recombined = ss[:, 0] + 1j * ss[:, 1]
    return float(torch.max(torch.abs(xs_complex - recombined)))


def decay_rate(xs: Tensor) -> float:
    r"""Mean per-step log-magnitude increment (should equal $\log\rho$)."""
    mags = torch.abs(xs)
    if torch.any(mags <= 0):
        raise ValueError("trajectory contains a non-positive magnitude")
    return float(torch.mean(torch.diff(torch.log(mags))))


class ComplexMode(nn.Module):
    r"""A single learnable complex mode realized via real RoPE rotation.

    Stores $\log\rho$ (so $\rho = e^{\log\rho} \le 1$ when $\log\rho \le 0$) and the
    angle $\theta$ as ``nn.Parameter``s; applies the real 2-D rotation-scaling each
    step. This is the buffers-vs-Parameters teaching point: nothing here is a fixed
    buffer — both knobs are trained.
    """

    def __init__(self, log_rho: float = -0.05, theta: float = math.pi / 9) -> None:
        super().__init__()
        self.log_rho = nn.Parameter(torch.tensor(float(log_rho)))
        self.theta = nn.Parameter(torch.tensor(float(theta)))

    def forward(self, drive_real: Tensor) -> Tensor:
        r"""Run the real 2-D recurrence $s_k = \rho R(\theta) s_{k-1} + d_k$.

        Parameters
        ----------
        drive_real : Tensor, shape (L, 2)
            Per-step real 2-vectors (the $[\operatorname{Re}, \operatorname{Im}]$
            of a complex drive).

        Returns
        -------
        Tensor, shape (L, 2)
            State trajectory.
        """
        if drive_real.ndim != 2 or drive_real.shape[1] != 2:
            raise ValueError(f"drive_real must be (L, 2), got {tuple(drive_real.shape)}")
        rho = torch.exp(self.log_rho)
        c = torch.cos(self.theta)
        s_ = torch.sin(self.theta)
        R = rho * torch.stack([torch.stack([c, -s_]), torch.stack([s_, c])])
        s = torch.zeros(2, dtype=drive_real.dtype)
        out = []
        for d2 in drive_real:
            s = R @ s + d2
            out.append(s)
        return torch.stack(out)
