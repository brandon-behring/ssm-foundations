r"""Chapter 13 torch companion — mLSTM matrix memory and the exponential-gate stabilizer.

Mirrors ``companions/ch13/jax/xlstm.py``: the naive (overflow-prone) mLSTM recurrence
with raw exponential gates and the log-domain max-state stabilizer. Eager loops,
float64, parity against JAX ``< 1e-9`` in the safe regime (pinned in the shared torch
test file).

Port credit
-----------
Mirrors the JAX module (greenfield from xLSTM, arXiv:2405.04517 §4).
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "log_sigmoid",
    "mlstm_naive",
    "mlstm_stabilized",
]

torch.set_default_dtype(torch.float64)


def log_sigmoid(x: Tensor) -> Tensor:
    r"""$\log\sigma(x) = -\mathrm{softplus}(-x) \le 0$ (forget log-gate)."""
    return -torch.nn.functional.softplus(-x)


def _check(q: Tensor, k: Tensor, v: Tensor, log_f: Tensor, log_i: Tensor) -> None:
    length = q.shape[0]
    if q.shape != k.shape or v.shape[0] != length:
        raise ValueError("q, k must share (L, d_k); v must be (L, d_v)")
    if log_f.shape != (length,) or log_i.shape != (length,):
        raise ValueError(f"log_f, log_i must have shape (L,) = ({length},)")


def mlstm_naive(q: Tensor, k: Tensor, v: Tensor, log_f: Tensor, log_i: Tensor) -> Tensor:
    r"""mLSTM readouts with raw exponential gates (overflows for large $\log i$)."""
    _check(q, k, v, log_f, log_i)
    length, d_k = q.shape
    d_v = v.shape[1]
    cell = torch.zeros(d_v, d_k)
    norm = torch.zeros(d_k)
    one = torch.tensor(1.0)
    outputs = []
    for t in range(length):
        f = torch.exp(log_f[t])
        i = torch.exp(log_i[t])
        cell = f * cell + i * torch.outer(v[t], k[t])
        norm = f * norm + i * k[t]
        denom = torch.maximum(torch.abs(norm @ q[t]), one)
        outputs.append(cell @ q[t] / denom)
    return torch.stack(outputs)


def mlstm_stabilized(
    q: Tensor, k: Tensor, v: Tensor, log_f: Tensor, log_i: Tensor
) -> tuple[Tensor, Tensor]:
    r"""mLSTM readouts via the log-domain max-state stabilizer (never overflows).

    Returns the readouts and the stabilizer-state trajectory $m_t$.
    """
    _check(q, k, v, log_f, log_i)
    length, d_k = q.shape
    d_v = v.shape[1]
    cell = torch.zeros(d_v, d_k)
    norm = torch.zeros(d_k)
    m = torch.tensor(-float("inf"))
    outputs = []
    ms = []
    for t in range(length):
        m_new = torch.maximum(log_f[t] + m, log_i[t])
        f_p = torch.exp(log_f[t] + m - m_new)
        i_p = torch.exp(log_i[t] - m_new)
        cell = f_p * cell + i_p * torch.outer(v[t], k[t])
        norm = f_p * norm + i_p * k[t]
        denom = torch.maximum(torch.abs(norm @ q[t]), torch.exp(-m_new))
        outputs.append(cell @ q[t] / denom)
        ms.append(m_new)
        m = m_new
    return torch.stack(outputs), torch.stack(ms)
