r"""Chapter 11 torch companion — Hyena's FFT causal long convolution.

Mirrors ``companions/ch11/jax/fftconv.py`` with ``torch.fft.rfft`` / ``irfft``.
float64 for ``< 1e-9`` parity with JAX and a ``< 1e-12`` match to the explicit
Toeplitz oracle.

Buffers vs Parameters: here the filter ``k`` is an *input*. In a real Hyena layer
the implicit filter is produced by an MLP whose weights are ``nn.Parameter``s
(learned), while a fixed positional grid would be a ``register_buffer``. The
test file makes the distinction concrete.

Port credit
-----------
Mirrors the JAX module, itself ported from
``post_transformers/experiments/jax/week11/hyena_lineage.py``. Hyena: Poli et al.,
arXiv:2302.10866.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "fftconv",
    "causal_conv1d_naive",
    "cyclic_conv_unpadded",
]

torch.set_default_dtype(torch.float64)


def _check_shapes(u: Tensor, k: Tensor, bias: Tensor) -> tuple[int, int, int]:
    if u.ndim != 3:
        raise ValueError(f"u must be 3D (B, L, D), got shape {tuple(u.shape)}")
    if k.ndim != 2:
        raise ValueError(f"k must be 2D (D, L), got shape {tuple(k.shape)}")
    if bias.ndim != 1:
        raise ValueError(f"bias must be 1D (D,), got shape {tuple(bias.shape)}")
    batch, seqlen, channels = u.shape
    if tuple(k.shape) != (channels, seqlen):
        raise ValueError(f"k shape {tuple(k.shape)} != (D, L) = ({channels}, {seqlen})")
    if tuple(bias.shape) != (channels,):
        raise ValueError(f"bias shape {tuple(bias.shape)} != (D,) = ({channels},)")
    return batch, seqlen, channels


def fftconv(u: Tensor, k: Tensor, bias: Tensor) -> Tensor:
    r"""FFT-based causal long convolution along the sequence axis ($2L$ padding)."""
    _, seqlen, _ = _check_shapes(u, k, bias)
    fft_size = 2 * seqlen
    u_t = u.transpose(1, 2)  # (B, D, L)
    u_f = torch.fft.rfft(u_t, n=fft_size, dim=-1)
    k_f = torch.fft.rfft(k, n=fft_size, dim=-1)
    y = torch.fft.irfft(u_f * k_f[None, :, :], n=fft_size, dim=-1)[..., :seqlen]
    y = y + bias[None, :, None] * u_t
    return y.transpose(1, 2)


def causal_conv1d_naive(u: Tensor, k: Tensor, bias: Tensor) -> Tensor:
    r"""$O(L^2)$ explicit lower-triangular Toeplitz reference."""
    _, seqlen, _ = _check_shapes(u, k, bias)
    idx = torch.arange(seqlen)
    diff = idx[:, None] - idx[None, :]
    valid = diff >= 0
    diff_safe = torch.where(valid, diff, torch.zeros_like(diff))
    big_k = torch.where(valid[None, :, :], k[:, diff_safe], torch.zeros((), dtype=k.dtype))
    u_t = u.transpose(1, 2)  # (B, D, L)
    y = torch.einsum("dts,bds->bdt", big_k, u_t) + bias[None, :, None] * u_t
    return y.transpose(1, 2)


def cyclic_conv_unpadded(u: Tensor, k: Tensor, bias: Tensor) -> Tensor:
    r"""Length-$L$ (un-padded) cyclic FFT conv — breaks causality (demonstrates $2L$ necessity)."""
    _, seqlen, _ = _check_shapes(u, k, bias)
    u_t = u.transpose(1, 2)
    u_f = torch.fft.rfft(u_t, n=seqlen, dim=-1)
    k_f = torch.fft.rfft(k, n=seqlen, dim=-1)
    y = torch.fft.irfft(u_f * k_f[None, :, :], n=seqlen, dim=-1)[..., :seqlen]
    y = y + bias[None, :, None] * u_t
    return y.transpose(1, 2)
