r"""Cross-framework parity: torch linear-attention / GLA / Hyena-FFTConv vs JAX.

Two layers (mirroring the ch09/ch10 torch suites):

* **standalone torch assertions** — the recurrent==parallel and gated==masked
  identities, FFTConv==Toeplitz, the capacity rank, and the buffers-vs-Parameters
  distinction — meaningful even without JAX present;
* **cross-framework parity** — recompute the JAX companion in-process and pin the
  torch outputs to it (``< 1e-9``, both float64). Skipped if JAX is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float64)

from companions.ch11.torch.fftconv import (  # noqa: E402
    causal_conv1d_naive,
    cyclic_conv_unpadded,
    fftconv,
)
from companions.ch11.torch.gated_linear_attention import (  # noqa: E402
    gated_masked,
    gated_recurrent,
    gla_scalar_decay_mask,
    retnet_decay_mask,
)
from companions.ch11.torch.linear_attention import (  # noqa: E402
    linear_attention_parallel,
    linear_attention_recurrent,
    linear_attention_state,
)


def _qkv(length=32, d=8, d_v=6, seed=0):
    rng = np.random.default_rng(seed)
    q = torch.tensor(rng.standard_normal((length, d)))
    k = torch.tensor(rng.standard_normal((length, d)))
    v = torch.tensor(rng.standard_normal((length, d_v)))
    return q, k, v


def _conv_inputs(length=64, channels=3, batch=2, seed=0):
    rng = np.random.default_rng(seed)
    u = torch.tensor(rng.standard_normal((batch, length, channels)))
    taps = np.arange(length)
    k = torch.tensor(rng.standard_normal((channels, length)) * np.exp(-0.05 * taps)[None, :])
    bias = torch.tensor(rng.standard_normal(channels))
    return u, k, bias


# --- standalone torch checks -----------------------------------------------


def test_linear_recurrent_equals_parallel():
    q, k, v = _qkv()
    y_rec = linear_attention_recurrent(q, k, v, feature_map="elu")
    y_par = linear_attention_parallel(q, k, v, feature_map="elu")
    assert torch.max(torch.abs(y_rec - y_par)) < 1e-12


def test_linattn_state_rank():
    rng = np.random.default_rng(11)
    for n_kv, expected in [(4, 4), (16, 16), (48, 32)]:
        k = torch.tensor(rng.standard_normal((n_kv, 32)))
        v = torch.tensor(rng.standard_normal((n_kv, 64)))
        assert int(torch.linalg.matrix_rank(linear_attention_state(k, v))) == expected


def test_gated_recurrent_equals_masked():
    q, k, v = _qkv()
    length, d_k = q.shape
    rng = np.random.default_rng(1)
    log_gamma = torch.log(0.9 + 0.09 * torch.sigmoid(torch.tensor(rng.standard_normal((length, d_k)))))
    assert torch.max(torch.abs(gated_recurrent(q, k, v, log_gamma) - gated_masked(q, k, v, log_gamma))) < 1e-12


def test_retnet_toeplitz_and_gla_constant_match():
    mask = retnet_decay_mask(0.9, 20)
    assert torch.max(torch.abs(mask[1:, 1:] - mask[:-1, :-1])) == 0.0  # Toeplitz
    # A constant GLA gate reproduces the RetNet mask.
    gla = gla_scalar_decay_mask(torch.full((20,), float(np.log(0.9))))
    assert torch.max(torch.abs(gla - mask)) < 1e-12


def test_fftconv_equals_naive_and_padding_necessary():
    u, k, bias = _conv_inputs()
    assert torch.max(torch.abs(fftconv(u, k, bias) - causal_conv1d_naive(u, k, bias))) < 1e-12
    nopad_err = float(torch.max(torch.abs(cyclic_conv_unpadded(u, k, bias) - causal_conv1d_naive(u, k, bias))))
    assert nopad_err > 1e-1  # un-padded breaks causality


def test_buffers_vs_parameters():
    """A fixed causal mask is a buffer; a learned kernel is a Parameter (the Hyena distinction)."""

    class TinyHyena(torch.nn.Module):
        def __init__(self, length: int, channels: int):
            super().__init__()
            self.register_buffer("causal", torch.tril(torch.ones(length, length)))  # fixed -> buffer
            self.kernel = torch.nn.Parameter(torch.zeros(channels, length))  # learned -> Parameter

    m = TinyHyena(8, 3)
    param_names = {n for n, _ in m.named_parameters()}
    buffer_names = {n for n, _ in m.named_buffers()}
    assert "kernel" in param_names and "causal" not in param_names
    assert "causal" in buffer_names and "kernel" not in buffer_names


def test_invalid_inputs_raise():
    q, k, v = _qkv(length=8)
    with pytest.raises(ValueError):
        linear_attention_recurrent(q, k[:-1], v)
    with pytest.raises(ValueError):
        retnet_decay_mask(1.5, 8)
    with pytest.raises(ValueError):
        fftconv(*(_conv_inputs(length=8, channels=2)[:1]), torch.zeros(2, 7), torch.zeros(2))


# --- cross-framework parity (recompute JAX in-process) ---------------------


def test_linear_attention_parity_against_jax():
    pytest.importorskip("jax")
    from companions.ch11.jax.linear_attention import linear_attention_parallel as jax_par

    rng = np.random.default_rng(7)
    q, k, v = (rng.standard_normal((40, 8)), rng.standard_normal((40, 8)), rng.standard_normal((40, 6)))
    y_torch = linear_attention_parallel(torch.tensor(q), torch.tensor(k), torch.tensor(v)).numpy()
    y_jax = np.asarray(jax_par(__import__("jax.numpy", fromlist=["a"]).asarray(q),
                               __import__("jax.numpy", fromlist=["a"]).asarray(k),
                               __import__("jax.numpy", fromlist=["a"]).asarray(v)))
    np.testing.assert_allclose(y_torch, y_jax, rtol=0, atol=1e-9)


def test_gated_masked_parity_against_jax():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from companions.ch11.jax.gated_linear_attention import gated_masked as jax_gated

    rng = np.random.default_rng(2)
    q, k, v = (rng.standard_normal((32, 8)), rng.standard_normal((32, 8)), rng.standard_normal((32, 6)))
    log_gamma = np.log(0.9 + 0.09 * (1 / (1 + np.exp(-rng.standard_normal((32, 8))))))
    y_torch = gated_masked(torch.tensor(q), torch.tensor(k), torch.tensor(v), torch.tensor(log_gamma)).numpy()
    y_jax = np.asarray(jax_gated(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), jnp.asarray(log_gamma)))
    np.testing.assert_allclose(y_torch, y_jax, rtol=0, atol=1e-9)


def test_fftconv_parity_against_jax():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from companions.ch11.jax.fftconv import fftconv as jax_fft

    rng = np.random.default_rng(5)
    u = rng.standard_normal((2, 128, 4))
    k = rng.standard_normal((4, 128))
    bias = rng.standard_normal(4)
    y_torch = fftconv(torch.tensor(u), torch.tensor(k), torch.tensor(bias)).numpy()
    y_jax = np.asarray(jax_fft(jnp.asarray(u), jnp.asarray(k), jnp.asarray(bias)))
    np.testing.assert_allclose(y_torch, y_jax, rtol=0, atol=1e-9)
