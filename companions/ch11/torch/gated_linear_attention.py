r"""Chapter 11 torch companion — gated linear attention (GLA / RetNet).

Mirrors ``companions/ch11/jax/gated_linear_attention.py``. The gated recurrence
$S_t = \mathrm{diag}(\gamma_t)S_{t-1} + \phi(k_t)v_t^\top$ is an eager loop (the
oracle); the masked-parallel twin uses the GLA $e^{\pm g_t}$ rescaling. float64
for ``< 1e-9`` parity with JAX. Unnormalized (RetNet/GLA replace the normalizer
with an output GroupNorm).

Port credit
-----------
Mirrors the JAX module (greenfield; RetNet arXiv:2307.08621, GLA arXiv:2312.06635).
"""

from __future__ import annotations

import torch
from torch import Tensor

from companions.ch11.torch.linear_attention import _resolve_phi

__all__ = [
    "gated_recurrent",
    "gated_masked",
    "retnet_decay_mask",
    "gla_scalar_decay_mask",
]

torch.set_default_dtype(torch.float64)


def _broadcast_log_gamma(log_gamma: Tensor, length: int, d_k: int) -> Tensor:
    log_gamma = torch.as_tensor(log_gamma)
    if log_gamma.ndim == 1:
        if log_gamma.shape[0] != length:
            raise ValueError(f"log_gamma (L,) must have L={length}, got {log_gamma.shape[0]}")
        return log_gamma[:, None].expand(length, d_k)
    if tuple(log_gamma.shape) != (length, d_k):
        raise ValueError(f"log_gamma must be (L,) or (L, d_k)=({length}, {d_k}); got {tuple(log_gamma.shape)}")
    return log_gamma


def gated_recurrent(q: Tensor, k: Tensor, v: Tensor, log_gamma: Tensor, feature_map="elu") -> Tensor:
    r"""Eager gated recurrence $S_t = \mathrm{diag}(\gamma_t)S_{t-1} + \phi(k_t)v_t^\top$."""
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError(f"q, k, v must be 2D; got {q.shape}, {k.shape}, {v.shape}")
    if q.shape != k.shape or v.shape[0] != q.shape[0]:
        raise ValueError(f"need q.shape==k.shape and matching L; got {q.shape}, {k.shape}, {v.shape}")
    phi = _resolve_phi(feature_map)
    qf, kf = phi(q), phi(k)
    length, d_k = qf.shape
    d_v = v.shape[1]
    gamma = torch.exp(_broadcast_log_gamma(log_gamma, length, d_k))
    s = torch.zeros(d_k, d_v, dtype=v.dtype)
    ys = []
    for t in range(length):
        s = gamma[t][:, None] * s + torch.outer(kf[t], v[t])
        ys.append(s.t() @ qf[t])
    return torch.stack(ys)


def gated_masked(q: Tensor, k: Tensor, v: Tensor, log_gamma: Tensor, feature_map="elu") -> Tensor:
    r"""Masked-parallel gated form via the $e^{\pm g_t}$ rescaling (equals the recurrence)."""
    phi = _resolve_phi(feature_map)
    qf, kf = phi(q), phi(k)
    length, d_k = qf.shape
    g = torch.cumsum(_broadcast_log_gamma(log_gamma, length, d_k), dim=0)
    q_tilde = qf * torch.exp(g)
    k_tilde = kf * torch.exp(-g)
    scores = q_tilde @ k_tilde.t()
    causal = torch.tril(torch.ones(length, length, dtype=scores.dtype))
    return (scores * causal) @ v


def retnet_decay_mask(gamma: float, length: int) -> Tensor:
    r"""RetNet constant-decay mask $L_{tj} = \gamma^{t-j}$ for $j\le t$, else $0$ (Toeplitz)."""
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    idx = torch.arange(length)
    diff = idx[:, None] - idx[None, :]
    return torch.where(diff >= 0, torch.as_tensor(float(gamma)) ** diff, torch.zeros((), dtype=torch.float64))


def gla_scalar_decay_mask(log_gamma_vec: Tensor) -> Tensor:
    r"""GLA scalar-gate decay mask $L_{tj} = e^{g_t - g_j}$, $j \le t$ (not Toeplitz when varying)."""
    log_gamma_vec = torch.as_tensor(log_gamma_vec)
    if log_gamma_vec.ndim != 1:
        raise ValueError(f"log_gamma_vec must be 1D (L,), got {tuple(log_gamma_vec.shape)}")
    length = log_gamma_vec.shape[0]
    g = torch.cumsum(log_gamma_vec, dim=0)
    diff = g[:, None] - g[None, :]
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool))
    return torch.where(causal, torch.exp(diff), torch.zeros((), dtype=torch.float64))
