r"""Tests for ch12 longhorn: the implicit step and its unconditional stability.

Pins the §12.3-§12.4 claims:

* ``lax.scan`` == materialised-projector oracle to ``< 1e-12`` (float64);
* **structural identity** — ``longhorn_step`` IS ``delta_rule_step`` evaluated
  at $\beta^{\mathrm{eff}} = 1/(\alpha + \|k\|^2)$, to machine zero;
* the implicit rate is self-limiting: $\beta^{\mathrm{eff}}\|k\|^2 < 1$ for
  every key magnitude (so Longhorn never reaches the explicit boundary 2),
  with equality $\beta^{\mathrm{eff}} = 1/\alpha$ at $k = 0$;
* under a repeated pair the error decays *exactly* geometrically with the
  analytic ratio: DeltaNet diverges past $\beta\|k\|^2 = 2$, Longhorn
  contracts for every $\alpha > 0$.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from companions.ch12.jax.delta_rule import delta_rule_step  # noqa: E402
from companions.ch12.jax.longhorn import (  # noqa: E402
    error_trajectory,
    longhorn_effective_beta,
    longhorn_naive,
    longhorn_recurrent,
    longhorn_step,
    longhorn_step_via_solve,
)


def _stream(length=48, d_k=8, d_v=6, seed=0):
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((length, d_k)))
    k = jnp.asarray(rng.standard_normal((length, d_k)))
    v = jnp.asarray(rng.standard_normal((length, d_v)))
    alphas = jnp.asarray(rng.uniform(0.5, 2.0, size=length))
    return q, k, v, alphas


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_scan_equals_naive_oracle(seed):
    """Longhorn needs no stable-regime guard: beta_eff self-limits even for raw keys."""
    q, k, v, alphas = _stream(seed=seed)
    y_scan, s_scan = longhorn_recurrent(q, k, v, alphas)
    y_naive, s_naive = longhorn_naive(q, k, v, alphas)
    assert_allclose(np.asarray(y_scan), np.asarray(y_naive), rtol=0, atol=1e-12)
    assert_allclose(np.asarray(s_scan), np.asarray(s_naive), rtol=0, atol=1e-12)


def test_structural_identity_with_delta_rule():
    """longhorn_step(S, k, v, alpha) == delta_rule_step(S, k, v, beta_eff) exactly."""
    rng = np.random.default_rng(2)
    state = jnp.asarray(rng.standard_normal((6, 8)))
    key = jnp.asarray(rng.standard_normal(8))
    value = jnp.asarray(rng.standard_normal(6))
    for alpha in (0.1, 0.7, 5.0):
        lh = longhorn_step(state, key, value, alpha)
        dn = delta_rule_step(state, key, value, longhorn_effective_beta(key, alpha))
        assert_allclose(np.asarray(lh), np.asarray(dn), rtol=0, atol=1e-15)


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("alpha", [0.1, 0.7, 5.0])
def test_closed_form_equals_dense_implicit_solve(alpha, seed):
    """The independent certificate of the closed form (Theorem 12.3): solving the
    stationarity system S_t (alpha I + k k^T) = alpha S_{t-1} + v k^T with a dense
    linear solve — no shared code with the rank-one form — agrees < 1e-12, and the
    closed form zeroes the stationarity residual."""
    rng = np.random.default_rng(seed)
    state = jnp.asarray(rng.standard_normal((6, 8)))
    key = jnp.asarray(rng.standard_normal(8))
    value = jnp.asarray(rng.standard_normal(6))
    closed = longhorn_step(state, key, value, alpha)
    solved = longhorn_step_via_solve(state, key, value, alpha)
    assert_allclose(np.asarray(closed), np.asarray(solved), rtol=0, atol=1e-12)
    residual = alpha * (closed - state) + jnp.outer(closed @ key - value, key)
    assert float(jnp.max(jnp.abs(residual))) < 1e-12


def test_effective_rate_is_self_limiting():
    """beta_eff * ||k||^2 = ||k||^2/(alpha + ||k||^2) < 1 for every key magnitude.

    Float-safe form of the guarantee: the product is strictly below 1 while the
    gap alpha/(alpha + ||k||^2) is representable (here up to ||k|| ~ 1e6, gap
    ~1e-13); at ||k|| ~ 1e8 the gap (~1e-17) falls below half an ulp and the
    product rounds to exactly 1.0 — never above it.
    """
    rng = np.random.default_rng(4)
    base = jnp.asarray(rng.standard_normal(8))
    alpha = 0.7
    for scale in (1.0, 1e2, 1e4, 1e6):
        key = base * scale
        product = float(longhorn_effective_beta(key, alpha) * (key @ key))
        assert 0.0 < product < 1.0, f"scale {scale:g}: beta_eff*||k||^2 = {product} not in (0, 1)"
    product = float(longhorn_effective_beta(base * 1e8, alpha) * ((base * 1e8) @ (base * 1e8)))
    assert product <= 1.0, f"even at float saturation the product never exceeds 1; got {product}"
    # At k = 0 the rate saturates at its cap 1/alpha exactly.
    assert float(longhorn_effective_beta(jnp.zeros(8), alpha)) == pytest.approx(1.0 / alpha)


def test_error_decay_ratio_equals_analytic_rho():
    """Repeated pair: the per-step Frobenius-error ratio IS the spectral radius."""
    rng = np.random.default_rng(0)
    key = rng.standard_normal(8)
    key = jnp.asarray(key / np.linalg.norm(key))  # unit key: beta*||k||^2 = beta
    value = jnp.asarray(rng.standard_normal(6))
    cases = [
        (dict(beta=0.5), 0.5),
        (dict(beta=1.9), 0.9),
        (dict(beta=2.5), 1.5),  # past the boundary: exact geometric DIVERGENCE
        (dict(alpha=1.0), 0.5),  # rho = alpha/(alpha + 1)
        (dict(alpha=0.25), 0.2),
    ]
    for kwargs, rho in cases:
        traj = np.asarray(error_trajectory(key, value, 12, **kwargs))
        # Early steps sit far above the float noise floor: pin them at 1e-12
        # (the precision the §12.4 captions quote).
        assert_allclose(traj[1:4] / traj[:3], rho, rtol=0, atol=1e-12)
        # Once ||S_t - S*|| shrinks toward eps * ||S*||, the subtraction noise
        # dominates the ratio — restrict the full-trajectory pin to steps above
        # that floor, at a correspondingly looser tolerance.
        mask = traj[:-1] >= 1e-4 * traj[0]
        ratios = (traj[1:] / traj[:-1])[mask]
        assert ratios.size >= 4, f"{kwargs}: too few steps above the noise floor"
        assert_allclose(ratios, rho, rtol=0, atol=1e-10)


def test_explicit_diverges_implicit_contracts():
    """The headline: past the boundary DeltaNet's error grows; Longhorn's always shrinks."""
    rng = np.random.default_rng(1)
    key = rng.standard_normal(8)
    key = jnp.asarray(key / np.linalg.norm(key))
    value = jnp.asarray(rng.standard_normal(6))
    dn = np.asarray(error_trajectory(key, value, 20, beta=2.5))
    lh = np.asarray(error_trajectory(key, value, 20, alpha=1.0))
    assert dn[-1] > 1e3 * dn[0], f"DeltaNet at beta=2.5 should diverge; got {dn[-1] / dn[0]:.1f}x"
    assert lh[-1] < 1e-5 * lh[0], f"Longhorn should contract; got {lh[-1] / lh[0]:.2e}x"


def test_argument_validation():
    q, k, v, alphas = _stream(length=8)
    with pytest.raises(ValueError):
        longhorn_recurrent(q, k, v, alphas[:-1])  # mismatched rates
    with pytest.raises(ValueError):
        longhorn_recurrent(q, k[:-1], v, alphas)  # mismatched keys
    key = jnp.ones(4)
    value = jnp.ones(3)
    with pytest.raises(ValueError):
        error_trajectory(key, value, 5)  # neither beta nor alpha
    with pytest.raises(ValueError):
        error_trajectory(key, value, 5, beta=0.5, alpha=1.0)  # both
