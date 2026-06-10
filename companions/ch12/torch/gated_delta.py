r"""Chapter 12 torch companion — the gated delta rule (Gated DeltaNet's update).

Mirrors ``companions/ch12/jax/gated_delta.py``: scalar decay $\gamma_t$ on the
erased state plus the ungated delta-rule write,

.. math::

    S_t = \gamma_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top.

Eager loop, float64, parity against JAX ``< 1e-9``. The $\gamma \equiv 1$ and
$\beta \equiv 0$ reductions are pinned in the shared torch test file.

Port credit
-----------
Mirrors the JAX module (greenfield from arXiv:2412.06464 §3).
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "gated_delta_step",
    "gated_delta_recurrent",
]

torch.set_default_dtype(torch.float64)


def gated_delta_step(
    state: Tensor, key: Tensor, value: Tensor, beta: float, gamma: float
) -> Tensor:
    r"""One gated-delta update $S \leftarrow \gamma S(I - \beta kk^\top) + \beta vk^\top$."""
    if state.shape != (value.shape[0], key.shape[0]):
        raise ValueError(
            f"state must have shape (d_v, d_k) = ({value.shape[0]}, {key.shape[0]}); "
            f"got {tuple(state.shape)}"
        )
    erased = state - beta * torch.outer(state @ key, key)
    return gamma * erased + beta * torch.outer(value, key)


def gated_delta_recurrent(
    q: Tensor, k: Tensor, v: Tensor, betas: Tensor, gammas: Tensor
) -> tuple[Tensor, Tensor]:
    r"""Eager gated DeltaNet over a sequence; post-update read $o_t = S_t q_t$."""
    if q.shape != k.shape or v.shape[0] != q.shape[0]:
        raise ValueError(
            f"inconsistent q/k/v shapes: {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    if betas.shape != (q.shape[0],) or gammas.shape != (q.shape[0],):
        raise ValueError(
            f"betas and gammas must have shape (L,) = ({q.shape[0]},); "
            f"got {tuple(betas.shape)}, {tuple(gammas.shape)}"
        )
    d_k, d_v = q.shape[1], v.shape[1]
    state = torch.zeros(d_v, d_k)
    outputs = []
    for t in range(q.shape[0]):
        erased = state - betas[t] * torch.outer(state @ k[t], k[t])
        state = gammas[t] * erased + betas[t] * torch.outer(v[t], k[t])
        outputs.append(state @ q[t])
    return torch.stack(outputs), state
