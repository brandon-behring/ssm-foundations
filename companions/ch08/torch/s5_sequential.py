r"""Chapter 8 §8.6 (PyTorch companion) — S5 as a *sequential* diagonal MIMO scan.

Mirrors ``companions/ch08/jax/s5_scan.py``, with one essential difference: PyTorch
has **no native parallel associative scan**, so the recurrence

.. math::

    h_k = \bar A \odot h_{k-1} + \bar B u_k

runs as an eager Python ``for`` loop (the honest define-by-run spelling, just like
the Chapter 7 torch encoder). The states it produces match the JAX
``associative_scan`` *bit-for-bit* (``tests/test_s4d_torch.py``) — the math is the
same linear recurrence. What torch cannot reproduce here is the parallel
**critical-path depth** $O(\log L)$: that O(log L) story is JAX-only (see the
``s5-scan-depth`` figure) and, in production, needs a custom CUDA/Triton scan.

Diagonal modes use the S4D-Lin init (§8.5); the $P$ modes are complex and the
real output is $2\,\mathrm{Re}(C h_k)$. complex128 throughout, matching the JAX
companion.

Port credit
-----------
Associative-scan structure (here serialized) follows
``post_transformers/experiments/jax/week06/s5_scan.py`` and Smith et al., S5
(arXiv:2208.04933).

Usage
-----
::

    PYTHONPATH=. python companions/ch08/torch/s5_sequential.py
"""

from __future__ import annotations

import math

import torch
from torch import nn

_RDTYPE = torch.float64
_CDTYPE = torch.complex128

__all__ = ["discretize_s5", "s5_sequential_scan", "s5_apply", "S5Layer"]


def discretize_s5(A: torch.Tensor, B: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Diagonal ZOH: $\bar A = e^{A\Delta}$, $\bar B = \frac{\bar A - 1}{A} B$."""
    if torch.any(A == 0):
        raise ValueError(
            "diagonal modes A must be nonzero (a zero mode has no ZOH input (Abar-1)/A)"
        )
    Abar = torch.exp(A * dt)
    Bbar = ((Abar - 1.0) / A).unsqueeze(-1) * B
    return Abar, Bbar


def s5_sequential_scan(Abar: torch.Tensor, Bu: torch.Tensor) -> torch.Tensor:
    r"""States $h_k$ via an eager sequential loop (torch has no parallel scan).

    Parameters
    ----------
    Abar : torch.Tensor, shape (P,), complex
        Diagonal discrete state matrix.
    Bu : torch.Tensor, shape (L, P), complex
        Per-step driving terms $\bar B u_k$.

    Returns
    -------
    hs : torch.Tensor, shape (L, P), complex
        States $h_0, \ldots, h_{L-1}$ from $h_{-1} = 0$.
    """
    L, P = Bu.shape
    h = torch.zeros(P, dtype=Bu.dtype)
    hs = []
    for k in range(L):
        h = Abar * h + Bu[k]
        hs.append(h)
    return torch.stack(hs)


def s5_apply(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: float,
    u: torch.Tensor,
) -> torch.Tensor:
    r"""Run the diagonal MIMO S5 SSM on ``u`` (shape ``(L, H)``); returns ``(L, H)`` real."""
    Abar, Bbar = discretize_s5(A, B, dt)
    Bu = u.to(Bbar.dtype) @ Bbar.T  # (L, P)
    hs = s5_sequential_scan(Abar, Bu)
    return 2.0 * (hs @ C.T).real


class S5Layer(nn.Module):
    """A diagonal MIMO S5 layer (sequential scan) as an ``nn.Module``.

    Diagonal ``A`` (S4D-Lin), ``B``, ``C`` are registered as buffers (fixed init for
    this companion). ``forward(u)`` maps ``(L, H) -> (L, H)``.
    """

    def __init__(self, n_modes: int, h_dim: int, dt: float = 0.1, seed: int = 0) -> None:
        super().__init__()
        if n_modes < 1 or h_dim < 1:
            raise ValueError(f"n_modes and h_dim must be >= 1, got {n_modes}, {h_dim}")
        gen = torch.Generator().manual_seed(seed)
        A = (-0.5 + 1j * math.pi * torch.arange(n_modes, dtype=_RDTYPE)).to(_CDTYPE)
        B = torch.randn(n_modes, h_dim, dtype=_RDTYPE, generator=gen) + 1j * torch.randn(
            n_modes, h_dim, dtype=_RDTYPE, generator=gen
        )
        C = torch.randn(h_dim, n_modes, dtype=_RDTYPE, generator=gen) + 1j * torch.randn(
            h_dim, n_modes, dtype=_RDTYPE, generator=gen
        )
        self.dt = dt
        self.register_buffer("A", A)
        self.register_buffer("B", B.to(_CDTYPE))
        self.register_buffer("C", C.to(_CDTYPE))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return s5_apply(self.A, self.B, self.C, self.dt, u)


def main() -> None:
    print("Chapter 8 (torch) — s5_sequential.py")
    print("=" * 60)
    layer = S5Layer(n_modes=8, h_dim=4)
    L = 256
    z = torch.linspace(0.0, 1.0, L, dtype=_RDTYPE)
    u = torch.stack([torch.sin(2 * math.pi * (k + 1) * z) for k in range(4)], dim=1)
    with torch.no_grad():
        y = layer(u)
    print(f"  L={L}: output shape {tuple(y.shape)}, real dtype {y.dtype}")
    print(
        f"  S5Layer learnable params: {sum(p.numel() for p in layer.parameters())} "
        f"(sequential-only; no torch associative_scan)"
    )


if __name__ == "__main__":
    main()
