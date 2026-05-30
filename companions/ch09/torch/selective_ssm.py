r"""Chapter 9 §9.1-9.3 — selective (input-dependent) SSM in PyTorch.

Mirrors ``companions/ch09/jax/selective_ssm.py`` (same SISO-simplified selective
SSM, same $A = -e^{a_\mathrm{log}}$ stability convention, same $\Delta_t B_t$
discretization). The functional core is pinned **bit-for-bit against the JAX
companion** in the tests (both run float64), the cross-framework lesson Chapter 7
introduced.

Two PyTorch-specific teaching points:

* **No native parallel scan.** JAX has ``lax.associative_scan``; PyTorch does
  not, so :func:`selective_scan_sequential` runs the recurrence as an eager
  Python loop — correct and identical to the JAX states, but $O(L)$ sequential
  (the same situation as Chapter 8's ``s5_sequential.py``; in production a custom
  CUDA/Triton kernel supplies the scan, §9.4).
* **Buffers vs Parameters.** In a selective SSM the modes *are learned*:
  :class:`SelectiveSSM` stores ``A_log`` as an ``nn.Parameter`` and the selection
  projections as ``nn.Linear``. Contrast Chapter 7's HiPPO encoder, where the
  fixed transition matrix was a registered *buffer*. There are no buffers here —
  nothing is held fixed.

Port credit
-----------
SISO-simplified from ``post_transformers/experiments/jax/week07/mamba1.py`` and
Gu & Dao, *Mamba* (arXiv:2312.00752).

Usage
-----
::

    PYTHONPATH=. python companions/ch09/torch/selective_ssm.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "stable_A",
    "discretize_selective",
    "selective_scan_sequential",
    "selective_apply",
    "SelectiveSSM",
]

_DTYPE = torch.float64


def stable_A(a_log: torch.Tensor) -> torch.Tensor:
    r"""Stable diagonal modes $A = -e^{a_\mathrm{log}}$ (real part $< 0$ for any input)."""
    return -torch.exp(a_log)


def discretize_selective(
    A: torch.Tensor, delta: torch.Tensor, B: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Time-varying ZOH: $\bar A_t = e^{\Delta_t A}$, $\bar B_t = \Delta_t B_t$ (shapes ``(L, N)``)."""
    if A.ndim != 1 or delta.ndim != 1:
        raise ValueError(f"A must be (N,) and delta (L,), got {tuple(A.shape)} and {tuple(delta.shape)}")
    if B.shape != (delta.shape[0], A.shape[0]):
        raise ValueError(f"B must be ({delta.shape[0]}, {A.shape[0]}), got {tuple(B.shape)}")
    abar = torch.exp(delta[:, None] * A[None, :])
    bbar = delta[:, None] * B
    return abar, bbar


def selective_scan_sequential(Abar: torch.Tensor, Bu: torch.Tensor) -> torch.Tensor:
    r"""States $h_k$ of the LTV recurrence via an eager loop (PyTorch has no parallel scan)."""
    if Abar.shape != Bu.shape:
        raise ValueError(f"Abar {tuple(Abar.shape)} must match Bu {tuple(Bu.shape)}")
    length, n = Abar.shape
    h = torch.zeros(n, dtype=Abar.dtype)
    states = []
    for t in range(length):
        h = Abar[t] * h + Bu[t]
        states.append(h)
    return torch.stack(states, dim=0)


def selective_apply(
    A: torch.Tensor,
    delta: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | float,
    u: torch.Tensor,
) -> torch.Tensor:
    r"""Run the SISO selective SSM on ``u`` (shape ``(L,)``); torch mirror of the JAX ``selective_apply``."""
    if u.ndim != 1:
        raise ValueError(f"u must be 1-D (L,), got {tuple(u.shape)}")
    if C.shape != B.shape:
        raise ValueError(f"C {tuple(C.shape)} must match B {tuple(B.shape)}")
    abar, bbar = discretize_selective(A, delta, B)
    bu = bbar * u[:, None]
    hs = selective_scan_sequential(abar, bu)
    d_feed = torch.as_tensor(D, dtype=hs.dtype)
    return (C * hs).sum(dim=1) + d_feed * u


class SelectiveSSM(nn.Module):
    r"""A SISO selective SSM layer: selection projections + eager scan.

    The selection mechanism (§9.1): $\Delta_t, B_t, C_t$ are projections of the
    feature input ``x``, and the SSM channel ``u`` is a learned projection too. The
    modes ``A_log`` are an ``nn.Parameter`` (learned), so $A = -e^{a\_log}$ stays
    stable for any training value (§8.5 sign trap). No registered buffers — unlike
    the fixed HiPPO buffer of Chapter 7.

    Parameters
    ----------
    d_model : int
        Feature dimension of the input tokens.
    n_state : int
        State dimension $N$.
    seed : int
        PRNG seed for reproducible initialization.
    """

    def __init__(self, d_model: int, n_state: int, seed: int = 0) -> None:
        super().__init__()
        if d_model < 1 or n_state < 1:
            raise ValueError(f"d_model and n_state must be >= 1, got {d_model}, {n_state}")
        torch.manual_seed(seed)
        self.d_model = d_model
        self.n_state = n_state
        # Init modes -1, ..., -N (the mamba-minimal convention), stored as log.
        n = torch.arange(1, n_state + 1, dtype=_DTYPE)
        self.A_log = nn.Parameter(torch.log(n))
        self.W_delta = nn.Linear(d_model, 1, dtype=_DTYPE)
        self.W_B = nn.Linear(d_model, n_state, bias=False, dtype=_DTYPE)
        self.W_C = nn.Linear(d_model, n_state, bias=False, dtype=_DTYPE)
        self.W_u = nn.Linear(d_model, 1, bias=False, dtype=_DTYPE)
        self.D = nn.Parameter(torch.zeros((), dtype=_DTYPE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Map a feature sequence ``x`` of shape ``(L, d_model)`` to ``(L,)`` output."""
        if x.ndim != 2 or x.shape[1] != self.d_model:
            raise ValueError(f"x must be (L, d_model={self.d_model}), got {tuple(x.shape)}")
        A = stable_A(self.A_log)  # (N,) negative
        delta = F.softplus(self.W_delta(x).squeeze(-1))  # (L,)
        B = self.W_B(x)  # (L, N)
        C = self.W_C(x)  # (L, N)
        u = self.W_u(x).squeeze(-1)  # (L,)
        return selective_apply(A, delta, B, C, self.D, u)


def main() -> None:
    torch.manual_seed(0)
    layer = SelectiveSSM(d_model=4, n_state=8)
    x = torch.randn(20, 4, dtype=_DTYPE)
    y = layer(x)
    n_params = sum(p.numel() for p in layer.parameters())
    print("Chapter 9 — selective_ssm.py (torch)")
    print("=" * 60)
    print(f"  output shape {tuple(y.shape)} (expect (20,)), dtype {y.dtype}")
    print(f"  finite: {bool(torch.all(torch.isfinite(y)))}")
    print(f"  learnable params (A_log, projections, D): {n_params}  (modes ARE learned; no buffers)")


if __name__ == "__main__":
    main()
