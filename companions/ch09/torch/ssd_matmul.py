r"""Chapter 9 §9.5-9.6 — the SSD semiseparable matrix and attention dual in PyTorch.

Mirrors ``companions/ch09/jax/ssd_semiseparable.py``: the same ``segsum`` decay,
the same semiseparable matrix $M$, and the same scalar-$A$ masked-attention form
$M = L \circ (C B^\top)\,\mathrm{diag}(\Delta)$. The matrices are pinned
bit-for-bit against the JAX companion in the tests.

The point of the *matmul* (quadratic) mode is that it is exactly the computation
GPUs are fastest at — dense matrix multiplies on tensor cores — which is why
Mamba-2 uses it within chunks. :class:`SSDAttention` packages the scalar-$A$ dual
form as an ``nn.Module``: it is masked linear attention with $C$ as queries, $B$
as keys, and a learned decay mask $L$ in place of softmax (the bridge to
Chapter 11).

Port credit
-----------
``segsum`` and the explicit-matrix framing from
``post_transformers/experiments/jax/week08/mamba2_ssd.py`` and Dao & Gu,
*Transformers are SSMs* (arXiv:2405.21060).

Usage
-----
::

    PYTHONPATH=. python companions/ch09/torch/ssd_matmul.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "segsum",
    "build_ssm_matrix",
    "ssd_apply_matmul",
    "masked_attention_form",
    "SSDAttention",
]

_DTYPE = torch.float64


def segsum(x: torch.Tensor) -> torch.Tensor:
    r"""Stable cumulative segment sum: $y_{ij} = \sum_{k=j+1}^i x_k$ for $i\ge j$, else $-\infty$."""
    if x.ndim < 1:
        raise ValueError(f"x must have at least one axis, got shape {tuple(x.shape)}")
    t = x.shape[-1]
    cum = torch.cumsum(x, dim=-1)
    diff = cum[..., :, None] - cum[..., None, :]
    mask = torch.tril(torch.ones(t, t, dtype=torch.bool))
    neg_inf = torch.tensor(float("-inf"), dtype=x.dtype)
    return torch.where(mask, diff, neg_inf)


def build_ssm_matrix(
    A: torch.Tensor, delta: torch.Tensor, B: torch.Tensor, C: torch.Tensor
) -> torch.Tensor:
    r"""The $L\times L$ semiseparable matrix $M_{kj}=\sum_n C_{k,n}e^{A_n\,\mathrm{seg}_{kj}}\Delta_j B_{j,n}$."""
    if A.ndim != 1 or delta.ndim != 1:
        raise ValueError(f"A must be (N,) and delta (L,), got {tuple(A.shape)} and {tuple(delta.shape)}")
    n, length = A.shape[0], delta.shape[0]
    if B.shape != (length, n) or C.shape != (length, n):
        raise ValueError(f"B and C must be ({length}, {n}), got {tuple(B.shape)} and {tuple(C.shape)}")
    adt = A[:, None] * delta[None, :]  # (N, L), already signed
    decay = torch.exp(segsum(adt))  # (N, L, L); strict upper -> 0
    return torch.einsum("kn,nkj,j,jn->kj", C, decay, delta, B)


def ssd_apply_matmul(M: torch.Tensor, D: torch.Tensor | float, u: torch.Tensor) -> torch.Tensor:
    r"""The dense (quadratic) SSD mode: $y = M u + D u$."""
    if u.ndim != 1 or M.shape != (u.shape[0], u.shape[0]):
        raise ValueError(f"need square M (L,L) and u (L,), got {tuple(M.shape)} and {tuple(u.shape)}")
    return M @ u + torch.as_tensor(D, dtype=M.dtype) * u


def masked_attention_form(
    a: float | torch.Tensor, delta: torch.Tensor, B: torch.Tensor, C: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Scalar-$A$ duality: $M = L \circ (C B^\top)\,\mathrm{diag}(\Delta)$, returning $(M, L, G)$."""
    a_t = torch.as_tensor(a, dtype=delta.dtype)
    l_mask = torch.exp(segsum(a_t * delta))  # (L, L), 1-semiseparable
    g = C @ B.t()  # (L, L) attention scores
    matrix = l_mask * g * delta[None, :]
    return matrix, l_mask, g


class SSDAttention(nn.Module):
    r"""Scalar-$A$ SSD as masked linear attention (the quadratic / dual mode, §9.6).

    Computes $y = (L \circ (C B^\top))\,\mathrm{diag}(\Delta)\,u + D u$ with the
    decay mask $L$ replacing softmax. ``a_log`` (scalar decay), the selection
    projections, and ``D`` are all ``nn.Parameter``.
    """

    def __init__(self, d_model: int, n_state: int, seed: int = 0) -> None:
        super().__init__()
        if d_model < 1 or n_state < 1:
            raise ValueError(f"d_model and n_state must be >= 1, got {d_model}, {n_state}")
        torch.manual_seed(seed)
        self.d_model = d_model
        self.n_state = n_state
        self.a_log = nn.Parameter(torch.zeros((), dtype=_DTYPE))  # scalar a = -exp(a_log)
        self.W_delta = nn.Linear(d_model, 1, dtype=_DTYPE)
        self.W_B = nn.Linear(d_model, n_state, bias=False, dtype=_DTYPE)
        self.W_C = nn.Linear(d_model, n_state, bias=False, dtype=_DTYPE)
        self.W_u = nn.Linear(d_model, 1, bias=False, dtype=_DTYPE)
        self.D = nn.Parameter(torch.zeros((), dtype=_DTYPE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Map a feature sequence ``x`` of shape ``(L, d_model)`` to ``(L,)`` via the dual form."""
        if x.ndim != 2 or x.shape[1] != self.d_model:
            raise ValueError(f"x must be (L, d_model={self.d_model}), got {tuple(x.shape)}")
        a = -torch.exp(self.a_log)
        delta = F.softplus(self.W_delta(x).squeeze(-1))
        B = self.W_B(x)
        C = self.W_C(x)
        u = self.W_u(x).squeeze(-1)
        matrix, _l, _g = masked_attention_form(a, delta, B, C)
        return ssd_apply_matmul(matrix, self.D, u)


def main() -> None:
    torch.manual_seed(0)
    layer = SSDAttention(d_model=4, n_state=6)
    x = torch.randn(24, 4, dtype=_DTYPE)
    y = layer(x)
    print("Chapter 9 — ssd_matmul.py (torch)")
    print("=" * 60)
    print(f"  output shape {tuple(y.shape)} (expect (24,)), dtype {y.dtype}")
    print(f"  finite: {bool(torch.all(torch.isfinite(y)))}  (masked-attention dual form)")


if __name__ == "__main__":
    main()
