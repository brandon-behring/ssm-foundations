r"""Cross-framework parity: torch generalized transition / mLSTM vs JAX.

Two layers (mirroring the ch12/ch16 torch suites):

* **standalone torch assertions** — the transition's symmetry, the P2 stabilizer
  exactness reproduced in torch, the overflow cliff — meaningful without JAX;
* **cross-framework parity** — recompute the JAX companions in-process on the same
  numpy-drawn inputs and pin the torch outputs to them (``< 1e-9``, both float64).
  Skipped if JAX is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float64)

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch13.jax import generalized_transition as jgt  # noqa: E402
from companions.ch13.jax import xlstm as jxl  # noqa: E402
from companions.ch13.torch import generalized_transition as tgt  # noqa: E402
from companions.ch13.torch import xlstm as txl  # noqa: E402

_ATOL = 1e-9


def _t(x):
    return torch.tensor(np.asarray(x, dtype=np.float64))


def _gen_stream(length=40, d_k=6, d_v=5, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((length, d_k))
    w = rng.uniform(0.7, 1.0, size=(length, d_k))
    a = rng.standard_normal((length, d_k))
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    c = rng.uniform(0.0, 0.5, size=length)
    u = rng.standard_normal((length, d_v))
    b = rng.standard_normal((length, d_k))
    return q, w, a, c, u, b


def _ch12_stream(length=40, d_k=6, d_v=5, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((length, d_k))
    k = rng.standard_normal((length, d_k))
    k = k / np.linalg.norm(k, axis=1, keepdims=True)
    v = rng.standard_normal((length, d_v))
    betas = rng.uniform(0.1, 0.9, size=length)
    gammas = rng.uniform(0.7, 1.0, size=length)
    return q, k, v, betas, gammas


def _gate_stream(length=24, d_k=6, d_v=5, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((length, d_k))
    k = rng.standard_normal((length, d_k))
    k = k / np.linalg.norm(k, axis=1, keepdims=True)
    v = rng.standard_normal((length, d_v))
    log_f_pre = rng.uniform(0.0, 2.0, size=length)
    log_i = rng.uniform(-2.0, 2.0, size=length)
    return q, k, v, log_f_pre, log_i


# --- standalone torch assertions -------------------------------------------


def test_transition_symmetric_torch():
    rng = np.random.default_rng(0)
    w = _t(rng.uniform(0.5, 1.0, size=6))
    a = _t(rng.standard_normal(6))
    a = a / torch.linalg.norm(a)
    A = tgt.dplr_transition(w, a, 0.7)
    assert torch.allclose(A, A.T, rtol=0, atol=1e-15)


def test_p2_exactness_torch():
    """The stabilizer is exact in torch too: stabilized == naive in the safe regime."""
    q, k, v, log_f_pre, log_i = _gate_stream(seed=1)
    log_f = txl.log_sigmoid(_t(log_f_pre))
    h_naive = txl.mlstm_naive(_t(q), _t(k), _t(v), log_f, _t(log_i))
    h_stab, _ = txl.mlstm_stabilized(_t(q), _t(k), _t(v), log_f, _t(log_i))
    assert bool(torch.all(torch.isfinite(h_naive)))
    assert torch.allclose(h_naive, h_stab, rtol=0, atol=1e-12)


def test_overflow_cliff_torch():
    """torch naive overflows at large log input-gate; stabilized stays finite."""
    rng = np.random.default_rng(0)
    length = 16
    q = _t(rng.standard_normal((length, 4)))
    k = _t(rng.standard_normal((length, 4)))
    k = k / torch.linalg.norm(k, dim=1, keepdim=True)
    v = _t(rng.standard_normal((length, 3)))
    log_f = txl.log_sigmoid(_t(rng.uniform(0.0, 2.0, size=length)))
    log_i = _t(rng.uniform(-1.0, 1.0, size=length))
    log_i[length // 2] = 760.0
    h_naive = txl.mlstm_naive(q, k, v, log_f, log_i)
    h_stab, _ = txl.mlstm_stabilized(q, k, v, log_f, log_i)
    assert not bool(torch.all(torch.isfinite(h_naive)))
    assert bool(torch.all(torch.isfinite(h_stab)))


# --- cross-framework parity (torch vs JAX) ---------------------------------


@pytest.mark.parametrize("seed", [0, 2])
def test_generalized_recurrent_parity(seed):
    q, w, a, c, u, b = _gen_stream(seed=seed)
    y_j, s_j = jgt.generalized_delta_recurrent(
        jnp.asarray(q), jnp.asarray(w), jnp.asarray(a), jnp.asarray(c), jnp.asarray(u), jnp.asarray(b)
    )
    y_t, s_t = tgt.generalized_delta_recurrent(_t(q), _t(w), _t(a), _t(c), _t(u), _t(b))
    assert np.max(np.abs(np.asarray(y_j) - y_t.numpy())) < _ATOL
    assert np.max(np.abs(np.asarray(s_j) - s_t.numpy())) < _ATOL


def test_gated_delta_reduction_parity():
    q, k, v, betas, gammas = _ch12_stream(seed=3)
    y_j, _ = jgt.gated_delta_reduction(
        jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), jnp.asarray(betas), jnp.asarray(gammas)
    )
    y_t, _ = tgt.gated_delta_reduction(_t(q), _t(k), _t(v), _t(betas), _t(gammas))
    assert np.max(np.abs(np.asarray(y_j) - y_t.numpy())) < _ATOL


@pytest.mark.parametrize("c", [0.3, 0.8, 1.5])
def test_transition_spectrum_parity(c):
    rng = np.random.default_rng(4)
    w = rng.uniform(0.55, 1.0, size=6)
    a = rng.standard_normal(6)
    a = a / np.linalg.norm(a)
    spec_j = jgt.transition_spectrum(jnp.asarray(w), jnp.asarray(a), c)
    spec_t = tgt.transition_spectrum(_t(w), _t(a), c)
    assert np.max(np.abs(np.asarray(spec_j) - spec_t.numpy())) < _ATOL


@pytest.mark.parametrize("seed", [0, 5])
def test_mlstm_stabilized_parity(seed):
    q, k, v, log_f_pre, log_i = _gate_stream(seed=seed)
    lf_j = jxl.log_sigmoid(jnp.asarray(log_f_pre))
    h_j, m_j = jxl.mlstm_stabilized(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), lf_j, jnp.asarray(log_i))
    lf_t = txl.log_sigmoid(_t(log_f_pre))
    h_t, m_t = txl.mlstm_stabilized(_t(q), _t(k), _t(v), lf_t, _t(log_i))
    assert np.max(np.abs(np.asarray(h_j) - h_t.numpy())) < _ATOL
    assert np.max(np.abs(np.asarray(m_j) - m_t.numpy())) < _ATOL


def test_mlstm_naive_parity_safe():
    q, k, v, log_f_pre, log_i = _gate_stream(seed=7)
    lf_j = jxl.log_sigmoid(jnp.asarray(log_f_pre))
    h_j = jxl.mlstm_naive(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), lf_j, jnp.asarray(log_i))
    lf_t = txl.log_sigmoid(_t(log_f_pre))
    h_t = txl.mlstm_naive(_t(q), _t(k), _t(v), lf_t, _t(log_i))
    assert np.max(np.abs(np.asarray(h_j) - h_t.numpy())) < _ATOL
