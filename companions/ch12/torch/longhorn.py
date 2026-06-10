r"""Chapter 12 torch companion — Longhorn's implicit one-step solve.

Mirrors ``companions/ch12/jax/longhorn.py``: the closed-form implicit step is
the delta rule evaluated at the self-limiting rate
$\beta^{\mathrm{eff}} = 1/(\alpha + \|k\|^2)$. Eager loop, float64, parity
against JAX ``< 1e-9``.

Port credit
-----------
Mirrors the JAX module (arXiv:2407.14207).
"""

from __future__ import annotations

import torch
from torch import Tensor

from companions.ch12.torch.delta_rule import delta_rule_step

__all__ = [
    "longhorn_effective_beta",
    "longhorn_step",
    "longhorn_recurrent",
]

torch.set_default_dtype(torch.float64)


def longhorn_effective_beta(key: Tensor, alpha: float) -> float:
    r"""Longhorn's effective rate $\beta^{\mathrm{eff}} = 1/(\alpha + \|k\|^2)$, capped at $1/\alpha$."""
    return float(1.0 / (alpha + key @ key))


def longhorn_step(state: Tensor, key: Tensor, value: Tensor, alpha: float) -> Tensor:
    r"""One implicit-step update — the delta rule at the self-limiting rate."""
    return delta_rule_step(state, key, value, longhorn_effective_beta(key, alpha))


def longhorn_recurrent(
    q: Tensor, k: Tensor, v: Tensor, alphas: Tensor
) -> tuple[Tensor, Tensor]:
    r"""Eager Longhorn over a sequence; post-update read $o_t = S_t q_t$."""
    if q.shape != k.shape or v.shape[0] != q.shape[0] or alphas.shape != (q.shape[0],):
        raise ValueError(
            f"inconsistent stream shapes: q {tuple(q.shape)}, k {tuple(k.shape)}, "
            f"v {tuple(v.shape)}, alphas {tuple(alphas.shape)}"
        )
    d_k, d_v = q.shape[1], v.shape[1]
    state = torch.zeros(d_v, d_k)
    outputs = []
    for t in range(q.shape[0]):
        beta_eff = 1.0 / (alphas[t] + k[t] @ k[t])
        state = state + beta_eff * torch.outer(v[t] - state @ k[t], k[t])
        outputs.append(state @ q[t])
    return torch.stack(outputs), state
