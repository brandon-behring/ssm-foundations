r"""Cross-framework parity: PyTorch Chapter 9 companions vs the JAX reference (§9.1-9.6).

The torch functional core is fed the *same* NumPy inputs as the JAX core and the
outputs are compared in float64, so the two frameworks stay in lock-step (the
cross-framework consistency goal). Module-level smoke tests confirm the
``nn.Module`` layers run and that stability holds for any parameters.

Pinned facts:

* torch ``selective_apply`` reproduces the JAX selective scan (§9.3);
* torch ``build_ssm_matrix`` / ``masked_attention_form`` reproduce the JAX SSD
  matrices (§9.5-9.6);
* $A = -e^{a\_log}$ keeps $|\bar A_t| < 1$ for any parameters (§9.1).
"""

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

torch = pytest.importorskip("torch")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch09.jax import selective_ssm as jax_sel  # noqa: E402
from companions.ch09.jax import ssd_semiseparable as jax_ssd  # noqa: E402
from companions.ch09.torch import selective_ssm as torch_sel  # noqa: E402
from companions.ch09.torch import ssd_matmul as torch_ssd  # noqa: E402

_DTYPE = torch.float64
_PARITY_TOL = 1e-9


def _numpy_system(n: int = 8, length: int = 48, d: int = 4, seed: int = 0):
    """Shared NumPy inputs both frameworks consume (so only the arithmetic differs)."""
    rng = np.random.default_rng(seed)
    a_log = rng.standard_normal(n)
    x = rng.standard_normal((length, d))
    w_delta = 0.5 * rng.standard_normal(d)
    w_B = rng.standard_normal((d, n))
    w_C = rng.standard_normal((d, n))
    delta = np.log1p(np.exp(x @ w_delta))  # softplus
    B = x @ w_B
    C = x @ w_C
    u = rng.standard_normal(length)
    A = -np.exp(a_log)
    return A, delta, B, C, u


# ---------------------------------------------------------------------------
# §9.3 — selective scan parity
# ---------------------------------------------------------------------------


def test_selective_apply_matches_jax() -> None:
    """torch selective_apply reproduces the JAX selective scan output (§9.3 parity)."""
    A, delta, B, C, u = _numpy_system()
    d_feed = 0.3
    y_jax = np.asarray(
        jax_sel.selective_apply(
            jnp.asarray(A), jnp.asarray(delta), jnp.asarray(B), jnp.asarray(C), d_feed, jnp.asarray(u)
        )
    )
    y_torch = torch_sel.selective_apply(
        torch.tensor(A), torch.tensor(delta), torch.tensor(B), torch.tensor(C), d_feed, torch.tensor(u)
    ).numpy()
    assert np.max(np.abs(y_jax - y_torch)) < _PARITY_TOL


# ---------------------------------------------------------------------------
# §9.5-9.6 — SSD matrix parity
# ---------------------------------------------------------------------------


def test_build_ssm_matrix_matches_jax() -> None:
    """torch build_ssm_matrix reproduces the JAX semiseparable matrix (§9.5 parity)."""
    A, delta, B, C, _ = _numpy_system(n=6, length=40, seed=1)
    m_jax = np.asarray(
        jax_ssd.build_ssm_matrix(jnp.asarray(A), jnp.asarray(delta), jnp.asarray(B), jnp.asarray(C))
    )
    m_torch = torch_ssd.build_ssm_matrix(
        torch.tensor(A), torch.tensor(delta), torch.tensor(B), torch.tensor(C)
    ).numpy()
    assert np.max(np.abs(m_jax - m_torch)) < _PARITY_TOL


def test_masked_attention_matches_jax() -> None:
    """torch masked_attention_form reproduces the JAX scalar-A SSD matrix (§9.6 parity)."""
    _A, delta, B, C, _ = _numpy_system(n=6, length=40, seed=2)
    a = -0.7
    m_jax, l_jax, _g = jax_ssd.masked_attention_form(a, jnp.asarray(delta), jnp.asarray(B), jnp.asarray(C))
    m_torch, l_torch, _gt = torch_ssd.masked_attention_form(
        a, torch.tensor(delta), torch.tensor(B), torch.tensor(C)
    )
    assert np.max(np.abs(np.asarray(m_jax) - m_torch.numpy())) < _PARITY_TOL
    assert np.max(np.abs(np.asarray(l_jax) - l_torch.numpy())) < _PARITY_TOL


def test_recurrent_equals_matmul_torch() -> None:
    """torch side: the recurrent scan and the dense M@u agree (§9.5, the two SSD modes)."""
    A, delta, B, C, u = _numpy_system(n=6, length=40, seed=3)
    d_feed = 0.5
    y_scan = torch_sel.selective_apply(
        torch.tensor(A), torch.tensor(delta), torch.tensor(B), torch.tensor(C), d_feed, torch.tensor(u)
    )
    matrix = torch_ssd.build_ssm_matrix(
        torch.tensor(A), torch.tensor(delta), torch.tensor(B), torch.tensor(C)
    )
    y_mat = torch_ssd.ssd_apply_matmul(matrix, d_feed, torch.tensor(u))
    assert float(torch.max(torch.abs(y_scan - y_mat))) < 1e-12


# ---------------------------------------------------------------------------
# §9.1 — stability, and the nn.Module smoke tests
# ---------------------------------------------------------------------------


def test_selective_stable_for_any_parameters_torch() -> None:
    r"""Stability by construction (§9.1): $A = -e^{a\_log} < 0$, so $|\bar A_t| \le 1$ for any params.

    Matches the JAX claim: the exact guarantee is $A < 0$ (float-safe), giving
    $|\bar A_t| \le 1$; strict $< 1$ holds for finite modes / moderate steps.
    """
    torch.manual_seed(0)
    for _ in range(50):
        A = torch_sel.stable_A(torch.randn(16, dtype=_DTYPE) * 10)
        delta = torch.nn.functional.softplus(torch.randn(8, dtype=_DTYPE) * 10)
        B = torch.randn(8, 16, dtype=_DTYPE)
        abar, _ = torch_sel.discretize_selective(A, delta, B)
        assert bool(torch.all(A < 0.0))
        assert bool(torch.all(torch.abs(abar) <= 1.0))
    # Moderate params -> strictly contractive.
    A = torch_sel.stable_A(torch.randn(8, dtype=_DTYPE))
    delta = torch.nn.functional.softplus(torch.randn(6, dtype=_DTYPE))
    B = torch.randn(6, 8, dtype=_DTYPE)
    abar, _ = torch_sel.discretize_selective(A, delta, B)
    assert bool(torch.all(torch.abs(abar) < 1.0))


def test_selective_module_forward() -> None:
    """SelectiveSSM runs and the modes are learnable Parameters (no buffers)."""
    layer = torch_sel.SelectiveSSM(d_model=4, n_state=8)
    y = layer(torch.randn(20, 4, dtype=_DTYPE))
    assert y.shape == (20,)
    assert bool(torch.all(torch.isfinite(y)))
    assert sum(p.numel() for p in layer.parameters()) > 0
    assert sum(1 for _ in layer.buffers()) == 0  # nothing is frozen


def test_ssd_attention_module_forward() -> None:
    """SSDAttention (the masked-attention dual) runs end to end."""
    layer = torch_ssd.SSDAttention(d_model=4, n_state=6)
    y = layer(torch.randn(24, 4, dtype=_DTYPE))
    assert y.shape == (24,)
    assert bool(torch.all(torch.isfinite(y)))
