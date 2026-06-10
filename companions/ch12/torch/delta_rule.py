r"""Chapter 12 torch companion — the delta rule (DeltaNet's state update).

Mirrors ``companions/ch12/jax/delta_rule.py`` in PyTorch: the recurrent form is
an eager loop over the matrix state (torch has no parallel scan — the loop is
also the cross-framework oracle). float64 throughout so parity against the JAX
companion is meaningful (``< 1e-9``).

Buffers vs Parameters: in these operator functions the per-step rates
$\beta_t$ are *data*; in a trained DeltaNet layer they are produced by a
learned projection of the input — an ``nn.Parameter``-bearing module — while a
fixed identity-like mask would be a ``register_buffer``. The distinction is
exercised in ``tests/test_delta_rule_lineage_torch.py``.

Port credit
-----------
Mirrors the JAX module (ported from ``post_transformers`` week12;
arXiv:2406.06484).
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "delta_rule_step",
    "delta_rule_recurrent",
    "delta_rule_fixed_point",
    "additive_state",
]

torch.set_default_dtype(torch.float64)


def delta_rule_step(state: Tensor, key: Tensor, value: Tensor, beta: float) -> Tensor:
    r"""One delta-rule update $S \leftarrow S + \beta (v - Sk)k^\top$ (rank-one form)."""
    if state.shape != (value.shape[0], key.shape[0]):
        raise ValueError(
            f"state must have shape (d_v, d_k) = ({value.shape[0]}, {key.shape[0]}); "
            f"got {tuple(state.shape)}"
        )
    return state + beta * torch.outer(value - state @ key, key)


def delta_rule_recurrent(
    q: Tensor, k: Tensor, v: Tensor, betas: Tensor, initial_state: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    r"""Eager DeltaNet over a sequence; post-update read $o_t = S_t q_t$.

    Same contract as the JAX ``delta_rule_recurrent``: q/k ``(L, d_k)``,
    v ``(L, d_v)``, betas ``(L,)``; returns outputs ``(L, d_v)`` and the final
    state ``(d_v, d_k)``.
    """
    if q.shape != k.shape or v.shape[0] != q.shape[0] or betas.shape != (q.shape[0],):
        raise ValueError(
            f"inconsistent stream shapes: q {tuple(q.shape)}, k {tuple(k.shape)}, "
            f"v {tuple(v.shape)}, betas {tuple(betas.shape)}"
        )
    d_k, d_v = q.shape[1], v.shape[1]
    state = torch.zeros(d_v, d_k) if initial_state is None else initial_state.clone()
    if state.shape != (d_v, d_k):
        raise ValueError(
            f"initial_state must have shape (d_v, d_k) = ({d_v}, {d_k}); got {tuple(state.shape)}"
        )
    outputs = []
    for t in range(q.shape[0]):
        state = state + betas[t] * torch.outer(v[t] - state @ k[t], k[t])
        outputs.append(state @ q[t])
    return torch.stack(outputs), state


def delta_rule_fixed_point(key: Tensor, value: Tensor) -> Tensor:
    r"""The fixed point $S^\star = vk^\top/\|k\|^2$ (exact retrieval, invariant under any beta)."""
    k_sq = float(key @ key)
    if k_sq == 0.0:
        raise ValueError("key must be nonzero: the fixed point v k^T / ||k||^2 is undefined")
    return torch.outer(value, key) / k_sq


def additive_state(keys: Tensor, values: Tensor) -> Tensor:
    r"""Chapter 11's additive state $S = \sum_i v_i k_i^\top$ (no erasure; interference lingers)."""
    if keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != values.shape[0]:
        raise ValueError(
            f"keys (K, d_k) and values (K, d_v) must share K; "
            f"got {tuple(keys.shape)} and {tuple(values.shape)}"
        )
    return values.T @ keys
