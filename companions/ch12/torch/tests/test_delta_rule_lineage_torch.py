r"""Cross-framework parity: torch delta rule / Longhorn / gated delta vs JAX.

Two layers (mirroring the ch09-ch11 torch suites):

* **standalone torch assertions** — overwrite semantics, the fixed point, the
  Longhorn structural identity and self-limiting rate, the gated reductions,
  and the buffers-vs-Parameters distinction — meaningful without JAX present;
* **cross-framework parity** — recompute the JAX companions in-process on the
  same inputs and pin the torch outputs to them (``< 1e-9``, both float64).
  Skipped if JAX is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float64)

from companions.ch12.torch.delta_rule import (  # noqa: E402
    additive_state,
    delta_rule_fixed_point,
    delta_rule_recurrent,
    delta_rule_step,
)
from companions.ch12.torch.gated_delta import (  # noqa: E402
    gated_delta_recurrent,
    gated_delta_step,
)
from companions.ch12.torch.longhorn import (  # noqa: E402
    longhorn_effective_beta,
    longhorn_recurrent,
    longhorn_step,
)


def _stream(length=40, d_k=8, d_v=6, seed=0):
    """Unit-norm keys + beta < 1: the stable regime (beta * ||k||^2 < 2)."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((length, d_k))
    k = rng.standard_normal((length, d_k))
    k = k / np.linalg.norm(k, axis=1, keepdims=True)
    v = rng.standard_normal((length, d_v))
    betas = rng.uniform(0.1, 0.9, size=length)
    return q, k, v, betas


# ---------------------------------------------------------------------------
# Standalone torch layer
# ---------------------------------------------------------------------------


def test_overwrite_replaces_additive_lingers():
    """Re-storing a key: the delta rule retrieves v2 exactly; additive keeps v1 + v2."""
    rng = np.random.default_rng(0)
    key = rng.standard_normal(8)
    key = torch.tensor(key / np.linalg.norm(key))
    v1, v2 = torch.tensor(rng.standard_normal(4)), torch.tensor(rng.standard_normal(4))
    s = delta_rule_step(torch.zeros(4, 8), key, v1, 1.0)
    s = delta_rule_step(s, key, v2, 1.0)
    assert float(torch.max(torch.abs(s @ key - v2))) < 1e-12
    s_add = additive_state(torch.stack([key, key]), torch.stack([v1, v2]))
    assert float(torch.max(torch.abs(s_add @ key - v2))) > 0.1  # the stale v1 residue


def test_fixed_point_invariant():
    rng = np.random.default_rng(1)
    key = torch.tensor(rng.standard_normal(8))
    value = torch.tensor(rng.standard_normal(6))
    s_star = delta_rule_fixed_point(key, value)
    for beta in (0.1, 1.0, 2.5):
        drift = float(torch.max(torch.abs(delta_rule_step(s_star, key, value, beta) - s_star)))
        assert drift < 1e-12, f"beta={beta}: fixed point drifted by {drift:.2e}"


def test_longhorn_structural_identity_and_cap():
    """longhorn_step == delta_rule_step at beta_eff; beta_eff * ||k||^2 < 1 always."""
    rng = np.random.default_rng(2)
    state = torch.tensor(rng.standard_normal((6, 8)))
    key = torch.tensor(rng.standard_normal(8))
    value = torch.tensor(rng.standard_normal(6))
    for alpha in (0.1, 0.7, 5.0):
        lh = longhorn_step(state, key, value, alpha)
        dn = delta_rule_step(state, key, value, longhorn_effective_beta(key, alpha))
        assert float(torch.max(torch.abs(lh - dn))) < 1e-15
    for scale in (1.0, 1e2, 1e4):
        kk = key * scale
        assert 0.0 < longhorn_effective_beta(kk, 0.7) * float(kk @ kk) < 1.0


def test_gated_reductions():
    """gamma == 1 recovers the plain delta rule; beta == 0 is pure decay by gamma."""
    q, k, v, betas = _stream(seed=3)
    q_t, k_t, v_t = torch.tensor(q), torch.tensor(k), torch.tensor(v)
    betas_t = torch.tensor(betas)
    y_gated, s_gated = gated_delta_recurrent(q_t, k_t, v_t, betas_t, torch.ones(len(betas)))
    y_plain, s_plain = delta_rule_recurrent(q_t, k_t, v_t, betas_t)
    assert float(torch.max(torch.abs(y_gated - y_plain))) < 1e-12
    assert float(torch.max(torch.abs(s_gated - s_plain))) < 1e-12

    rng = np.random.default_rng(4)
    state = torch.tensor(rng.standard_normal((6, 8)))
    stepped = gated_delta_step(state, k_t[0], v_t[0], 0.0, 0.9)
    assert float(torch.max(torch.abs(stepped - 0.9 * state))) < 1e-15


def test_buffers_vs_parameters():
    """A learned rate projection is an nn.Parameter (optimized); a fixed floor is a buffer."""

    class TinyDeltaLayer(torch.nn.Module):
        def __init__(self, d_k: int) -> None:
            super().__init__()
            self.rate_proj = torch.nn.Parameter(torch.zeros(d_k))  # learned: beta_t = sigmoid(w.k_t)
            self.register_buffer("beta_floor", torch.tensor(0.05))  # fixed data: moves, not optimized

        def rate(self, key: torch.Tensor) -> torch.Tensor:
            return self.beta_floor + (1.0 - self.beta_floor) * torch.sigmoid(self.rate_proj @ key)

    layer = TinyDeltaLayer(8)
    param_names = {name for name, _ in layer.named_parameters()}
    assert param_names == {"rate_proj"}  # the buffer is NOT a parameter
    assert set(layer.state_dict()) == {"rate_proj", "beta_floor"}  # but both persist
    rate = layer.rate(torch.ones(8)).detach()
    assert 0.05 <= float(rate) <= 1.0  # the floor bounds the learned rate


def test_argument_validation():
    q, k, v, betas = _stream(length=8)
    q_t, k_t, v_t, b_t = torch.tensor(q), torch.tensor(k), torch.tensor(v), torch.tensor(betas)
    with pytest.raises(ValueError):
        delta_rule_recurrent(q_t, k_t[:-1], v_t, b_t)
    with pytest.raises(ValueError):
        longhorn_recurrent(q_t, k_t, v_t, b_t[:-1])
    with pytest.raises(ValueError):
        gated_delta_recurrent(q_t, k_t, v_t, b_t, b_t[:-1])
    with pytest.raises(ValueError):
        delta_rule_fixed_point(torch.zeros(8), torch.ones(6))


# ---------------------------------------------------------------------------
# Cross-framework parity (skipped without JAX)
# ---------------------------------------------------------------------------


def test_delta_rule_parity_against_jax():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from companions.ch12.jax.delta_rule import delta_rule_recurrent as jax_delta

    q, k, v, betas = _stream(seed=7)
    y_torch, s_torch = delta_rule_recurrent(
        torch.tensor(q), torch.tensor(k), torch.tensor(v), torch.tensor(betas)
    )
    y_jax, s_jax = jax_delta(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), jnp.asarray(betas))
    np.testing.assert_allclose(y_torch.numpy(), np.asarray(y_jax), rtol=0, atol=1e-9)
    np.testing.assert_allclose(s_torch.numpy(), np.asarray(s_jax), rtol=0, atol=1e-9)


def test_longhorn_parity_against_jax():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from companions.ch12.jax.longhorn import longhorn_recurrent as jax_longhorn

    rng = np.random.default_rng(8)
    q, k, v = (rng.standard_normal((40, 8)), rng.standard_normal((40, 8)),
               rng.standard_normal((40, 6)))
    alphas = rng.uniform(0.5, 2.0, size=40)
    y_torch, s_torch = longhorn_recurrent(
        torch.tensor(q), torch.tensor(k), torch.tensor(v), torch.tensor(alphas)
    )
    y_jax, s_jax = jax_longhorn(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), jnp.asarray(alphas))
    np.testing.assert_allclose(y_torch.numpy(), np.asarray(y_jax), rtol=0, atol=1e-9)
    np.testing.assert_allclose(s_torch.numpy(), np.asarray(s_jax), rtol=0, atol=1e-9)


def test_gated_delta_parity_against_jax():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from companions.ch12.jax.gated_delta import gated_delta_recurrent as jax_gated

    q, k, v, betas = _stream(seed=9)
    rng = np.random.default_rng(10)
    gammas = rng.uniform(0.7, 1.0, size=len(betas))
    y_torch, s_torch = gated_delta_recurrent(
        torch.tensor(q), torch.tensor(k), torch.tensor(v),
        torch.tensor(betas), torch.tensor(gammas),
    )
    y_jax, s_jax = jax_gated(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v),
                             jnp.asarray(betas), jnp.asarray(gammas))
    np.testing.assert_allclose(y_torch.numpy(), np.asarray(y_jax), rtol=0, atol=1e-9)
    np.testing.assert_allclose(s_torch.numpy(), np.asarray(s_jax), rtol=0, atol=1e-9)
