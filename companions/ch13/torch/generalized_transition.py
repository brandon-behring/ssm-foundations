r"""Chapter 13 torch companion — the generalized (diagonal-plus-rank-one) transition.

Mirrors ``companions/ch13/jax/generalized_transition.py``: the symmetric transition
$A = \mathrm{Diag}(w) - c\,a a^\top$, its real spectrum, the generalized delta-rule
recurrence $S_t = S_{t-1}(\mathrm{Diag}(w_t) - c_t a_t a_t^\top) + u_t b_t^\top$, and
the reduction to Chapter 12's gated DeltaNet. Eager loop, float64, parity against
JAX ``< 1e-9`` (pinned in the shared torch test file).

Port credit
-----------
Mirrors the JAX module (greenfield from RWKV-7, arXiv:2503.14456).
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "dplr_transition",
    "transition_spectrum",
    "generalized_delta_step",
    "generalized_delta_recurrent",
    "gated_delta_reduction",
]

torch.set_default_dtype(torch.float64)


def dplr_transition(w: Tensor, a: Tensor, c: float) -> Tensor:
    r"""The symmetric transition $A = \mathrm{Diag}(w) - c\,a a^\top$."""
    if w.shape != a.shape or w.ndim != 1:
        raise ValueError(f"w and a must share shape (d,); got {tuple(w.shape)}, {tuple(a.shape)}")
    if c < 0.0:
        raise ValueError(f"removal coefficient c must be >= 0; got {c}")
    return torch.diag(w) - c * torch.outer(a, a)


def transition_spectrum(w: Tensor, a: Tensor, c: float) -> Tensor:
    r"""Ascending real eigenvalues of $\mathrm{Diag}(w) - c\,a a^\top$ via ``eigvalsh``."""
    return torch.linalg.eigvalsh(dplr_transition(w, a, c))


def generalized_delta_step(
    state: Tensor, w: Tensor, a: Tensor, c: float, u: Tensor, b: Tensor
) -> Tensor:
    r"""One step $S \leftarrow S\,\mathrm{Diag}(w) - c\,(S a)a^\top + u b^\top$."""
    if state.shape != (u.shape[0], w.shape[0]):
        raise ValueError(
            f"state must have shape (d_v, d_k) = ({u.shape[0]}, {w.shape[0]}); got {tuple(state.shape)}"
        )
    return state * w - c * torch.outer(state @ a, a) + torch.outer(u, b)


def generalized_delta_recurrent(
    q: Tensor, w: Tensor, a: Tensor, c: Tensor, u: Tensor, b: Tensor
) -> tuple[Tensor, Tensor]:
    r"""Eager generalized delta rule over a sequence; post-update read $o_t = S_t q_t$."""
    length, d_k = q.shape
    d_v = u.shape[1]
    if w.shape != (length, d_k) or a.shape != (length, d_k) or b.shape != (length, d_k):
        raise ValueError("w, a, b must each have shape (L, d_k)")
    if c.shape != (length,):
        raise ValueError(f"c must have shape (L,) = ({length},); got {tuple(c.shape)}")
    state = torch.zeros(d_v, d_k)
    outputs = []
    for t in range(length):
        state = state * w[t] - c[t] * torch.outer(state @ a[t], a[t]) + torch.outer(u[t], b[t])
        outputs.append(state @ q[t])
    return torch.stack(outputs), state


def gated_delta_reduction(
    q: Tensor, k: Tensor, v: Tensor, betas: Tensor, gammas: Tensor
) -> tuple[Tensor, Tensor]:
    r"""Generalized rule with the parameters reproducing Chapter 12's gated DeltaNet.

    $w_t = \gamma_t\mathbf 1$, $a_t = k_t/\|k_t\|$, $c_t = \gamma_t\beta_t\|k_t\|^2$,
    write $u_t b_t^\top = \beta_t v_t k_t^\top$.
    """
    length, d_k = q.shape
    k_norm = torch.linalg.norm(k, dim=1, keepdim=True)
    a = k / k_norm
    w = gammas[:, None].expand(length, d_k)
    c = gammas * betas * (k_norm[:, 0] ** 2)
    u = betas[:, None] * v
    return generalized_delta_recurrent(q, w, a, c, u, k)
