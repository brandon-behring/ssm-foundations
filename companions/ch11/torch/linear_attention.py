r"""Chapter 11 torch companion — linear attention as a matrix-state recurrence.

Mirrors ``companions/ch11/jax/linear_attention.py`` in PyTorch. The recurrent form
is an eager loop over the matrix state $S_t$ (torch has no parallel scan — the
loop is also the cross-framework oracle); the parallel form is a masked matmul.
float64 throughout so parity against the JAX companion is meaningful (``< 1e-9``).

Buffers vs Parameters: in these functions the causal mask is data, not a learned
weight — in a real ``nn.Module`` it would be a ``register_buffer`` (moves with the
module, not optimized), while a *learned* feature projection would be an
``nn.Parameter``. See ``tests/test_linear_attention_hyena_torch.py`` for the
concrete distinction.

Port credit
-----------
Mirrors the JAX module (greenfield; Katharopoulos et al., arXiv:2006.16236).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "feature_map_elu",
    "feature_map_relu",
    "resolve_phi",
    "linear_attention_recurrent",
    "linear_attention_parallel",
    "linear_attention_state",
]

torch.set_default_dtype(torch.float64)


def feature_map_elu(x: Tensor) -> Tensor:
    r"""Katharopoulos feature map $\phi(x) = \mathrm{elu}(x) + 1$ (strictly positive)."""
    return F.elu(x) + 1.0


def feature_map_relu(x: Tensor) -> Tensor:
    r"""ReLU feature map $\phi(x) = \max(x, 0)$ (non-negative)."""
    return F.relu(x)


_PHI = {"elu": feature_map_elu, "relu": feature_map_relu}


def resolve_phi(feature_map):
    if callable(feature_map):
        return feature_map
    if feature_map not in _PHI:
        raise ValueError(f"unknown feature_map {feature_map!r}; expected one of {sorted(_PHI)}")
    return _PHI[feature_map]


def _check_qkv(q: Tensor, k: Tensor, v: Tensor) -> None:
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError(f"q, k, v must each be 2D (L, d); got {q.shape}, {k.shape}, {v.shape}")
    if q.shape != k.shape:
        raise ValueError(f"q and k must share shape; got {q.shape} and {k.shape}")
    if v.shape[0] != q.shape[0]:
        raise ValueError(f"v must have the same length L as q/k; got {v.shape[0]} vs {q.shape[0]}")


def linear_attention_recurrent(q: Tensor, k: Tensor, v: Tensor, feature_map="elu", normalize: bool = True) -> Tensor:
    r"""Eager matrix-state recurrence $S_t = S_{t-1} + \phi(k_t)v_t^\top$, $y_t = S_t^\top\phi(q_t)$."""
    _check_qkv(q, k, v)
    phi = resolve_phi(feature_map)
    qf, kf = phi(q), phi(k)
    length, d_k = qf.shape
    d_v = v.shape[1]
    s = torch.zeros(d_k, d_v, dtype=v.dtype)
    z = torch.zeros(d_k, dtype=qf.dtype)
    ys = []
    for t in range(length):
        s = s + torch.outer(kf[t], v[t])
        z = z + kf[t]
        num = s.t() @ qf[t]
        ys.append(num / (z @ qf[t]) if normalize else num)
    return torch.stack(ys)


def linear_attention_parallel(q: Tensor, k: Tensor, v: Tensor, feature_map="elu", normalize: bool = True) -> Tensor:
    r"""Masked-parallel form $Y = (L \circ (Q_\phi K_\phi^\top))V$ with $L$ the all-ones causal mask."""
    _check_qkv(q, k, v)
    phi = resolve_phi(feature_map)
    qf, kf = phi(q), phi(k)
    length = qf.shape[0]
    scores = qf @ kf.t()
    causal = torch.tril(torch.ones(length, length, dtype=scores.dtype))
    masked = scores * causal
    num = masked @ v
    if normalize:
        return num / masked.sum(dim=1, keepdim=True)
    return num


def linear_attention_state(k: Tensor, v: Tensor, feature_map="elu") -> Tensor:
    r"""The final matrix state $S = \sum_i \phi(k_i)v_i^\top = \Phi_K^\top V$ (rank $\le \min(K, d_k)$)."""
    if k.ndim != 2 or v.ndim != 2 or k.shape[0] != v.shape[0]:
        raise ValueError(f"k (K, d) and v (K, d_v) must share K; got {k.shape} and {v.shape}")
    kf = resolve_phi(feature_map)(k)
    return kf.t() @ v
