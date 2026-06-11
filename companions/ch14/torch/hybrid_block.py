r"""Chapter 14 torch companion — hybrid composition patterns.

Mirrors ``companions/ch14/jax/hybrid_block.py`` in PyTorch: the windowed
attention is the same vectorised band mask; the gated-decay EMA is an eager
loop (torch has no parallel scan — the loop is also the cross-framework
oracle). float64 throughout so parity against the JAX companion is
meaningful (``< 1e-9``). Decode-cost accounting stays JAX-side (it is
arithmetic, not compute).

Buffers vs Parameters: :class:`TinyGatedMixLayer` carries its mixing gate as
an ``nn.Parameter`` (a trained hybrid learns the blend) and its per-channel
decay rates as a ``register_buffer`` (fixed data moved with the module, not
optimised) — the distinction is exercised in the tests.

Port credit
-----------
Mirrors the JAX module (greenfield for this chapter; see its docstring for
the cited architecture papers).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = [
    "full_causal_attention",
    "sliding_window_attention",
    "gated_decay_ssm",
    "parallel_gated_hybrid",
    "interleave_schedule",
    "interleave_hybrid",
    "TinyGatedMixLayer",
]

torch.set_default_dtype(torch.float64)


def _validate_sequence(x: Tensor) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must have shape (L, d); got {tuple(x.shape)}")


def full_causal_attention(x: Tensor) -> Tensor:
    r"""Single-head causal self-attention with $q = k = v = x$ (no projections)."""
    _validate_sequence(x)
    length, d = x.shape
    scores = (x @ x.T) / torch.sqrt(torch.tensor(float(d)))
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool))
    scores = scores.masked_fill(~causal, float("-inf"))
    return torch.softmax(scores, dim=-1) @ x


def sliding_window_attention(x: Tensor, window: int) -> Tensor:
    r"""Causal self-attention restricted to the last ``window`` positions (band mask)."""
    _validate_sequence(x)
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}")
    length, d = x.shape
    scores = (x @ x.T) / torch.sqrt(torch.tensor(float(d)))
    rows = torch.arange(length)[:, None]
    cols = torch.arange(length)[None, :]
    band = (cols <= rows) & (cols > rows - window)
    scores = scores.masked_fill(~band, float("-inf"))
    return torch.softmax(scores, dim=-1) @ x


def gated_decay_ssm(x: Tensor, gates: Tensor) -> Tensor:
    r"""Per-channel gated EMA $h_t = g_t \odot h_{t-1} + (1-g_t) \odot x_t$ (eager loop)."""
    _validate_sequence(x)
    if gates.ndim == 1:
        gates = gates.expand(x.shape[0], -1)
    if gates.shape != x.shape:
        raise ValueError(
            f"gates must have shape {tuple(x.shape)} or ({x.shape[1]},); "
            f"got {tuple(gates.shape)}"
        )
    if bool(torch.any((gates < 0.0) | (gates > 1.0))):
        raise ValueError("gates must lie in [0, 1]")
    h = torch.zeros(x.shape[1])
    outputs = []
    for t in range(x.shape[0]):
        h = gates[t] * h + (1.0 - gates[t]) * x[t]
        outputs.append(h)
    return torch.stack(outputs)


def parallel_gated_hybrid(x: Tensor, gate: float | Tensor, gates_ssm: Tensor, window: int) -> Tensor:
    r"""Gated parallel mix $g \odot \mathrm{attn}(x) + (1-g) \odot \mathrm{ssm}(x)$."""
    _validate_sequence(x)
    g = torch.as_tensor(gate, dtype=x.dtype)
    if g.ndim not in (0, 1) or (g.ndim == 1 and g.shape != (x.shape[1],)):
        raise ValueError(f"gate must be a scalar or shape ({x.shape[1]},); got {tuple(g.shape)}")
    if bool(torch.any((g < 0.0) | (g > 1.0))):
        raise ValueError("gate must lie in [0, 1]")
    return g * sliding_window_attention(x, window) + (1.0 - g) * gated_decay_ssm(x, gates_ssm)


def interleave_schedule(n_blocks: int, ratio: int) -> tuple[str, ...]:
    r"""The $r\!:\!1$ layer schedule (``ratio`` SSM blocks, then one attention block)."""
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1; got {n_blocks}")
    if ratio < 0:
        raise ValueError(f"ratio must be >= 0; got {ratio}")
    period = ratio + 1
    return tuple("attn" if (i + 1) % period == 0 else "ssm" for i in range(n_blocks))


def interleave_hybrid(
    x: Tensor, schedule: tuple[str, ...], gates_ssm: Tensor, window: int
) -> Tensor:
    r"""Residual stack following ``schedule``: $y \leftarrow y + \mathrm{layer}(y)$."""
    _validate_sequence(x)
    if not schedule:
        raise ValueError("schedule must be non-empty")
    bad = sorted(set(schedule) - {"ssm", "attn"})
    if bad:
        raise ValueError(f'schedule entries must be "ssm" or "attn"; got {bad}')
    y = x
    for layer in schedule:
        if layer == "ssm":
            y = y + gated_decay_ssm(y, gates_ssm)
        else:
            y = y + sliding_window_attention(y, window)
    return y


class TinyGatedMixLayer(nn.Module):
    r"""A minimal parallel-gated hybrid layer: learned gate, fixed decays.

    The per-channel mixing gate (``gate_logit``, sigmoid-squashed) is an
    ``nn.Parameter`` — in a trained hybrid the blend is learned. The decay
    rates of the SSM branch are a ``register_buffer`` — fixed data that
    moves with the module but is not optimised. (In production both are
    typically input-conditioned projections; this layer isolates the
    storage-semantics distinction.)
    """

    def __init__(self, d: int, window: int, decays: Tensor) -> None:
        super().__init__()
        if decays.shape != (d,):
            raise ValueError(f"decays must have shape ({d},); got {tuple(decays.shape)}")
        self.window = window
        self.gate_logit = nn.Parameter(torch.zeros(d))
        self.register_buffer("decays", decays.clone())

    def forward(self, x: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate_logit)
        return gate * sliding_window_attention(x, self.window) + (1.0 - gate) * gated_decay_ssm(
            x, self.decays
        )
