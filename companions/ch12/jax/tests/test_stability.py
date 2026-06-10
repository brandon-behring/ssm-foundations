r"""Tests for ch12 stability: the closed-form spectral radii and the drift guard.

Pins the §12.4 claims:

* the analytic $k$-direction eigenvalue $1 - \beta^{\mathrm{eff}}\|k\|^2$
  matches the Rayleigh quotient of the materialised iteration matrix to
  ``< 1e-12`` across the parameter grid (derivation-drift guard);
* DeltaNet is stable exactly on $\beta\|k\|^2 \in (0, 2)$, with the named
  boundary constant 2;
* Longhorn's radius $\alpha/(\alpha + \|k\|^2)$ is strictly below 1 for every
  positive $\alpha$ and key magnitude, and the complement identity
  $\rho_{\mathrm{LH}} + \beta^{\mathrm{eff}}\|k\|^2 = 1$ holds exactly.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from companions.ch12.jax.stability import (  # noqa: E402
    deltanet_a_stability_boundary,
    deltanet_spectral_radius,
    iteration_eigenvalue_along_k,
    longhorn_effective_beta_k_product,
    longhorn_spectral_radius,
)


def test_analytic_matches_rayleigh_eigenvalue():
    """The drift guard: closed form == eigenvalue of the materialised matrix."""
    rng = np.random.default_rng(0)
    for bk in np.linspace(0.05, 2.95, 25):
        key = jnp.asarray(rng.standard_normal(8))
        beta_eff = bk / float(key @ key)  # so beta_eff * ||k||^2 = bk
        lam_numerical = float(iteration_eigenvalue_along_k(key, beta_eff))
        assert lam_numerical == pytest.approx(1.0 - bk, rel=0, abs=1e-12)


def test_deltanet_stability_interval():
    """rho < 1 exactly on beta*||k||^2 in (0, 2); the boundary constant is 2."""
    assert deltanet_a_stability_boundary() == 2.0
    bk = np.linspace(0.05, 3.0, 60)
    rho = np.asarray(deltanet_spectral_radius(bk, 1.0))
    inside = bk < 2.0
    assert np.all(rho[inside] < 1.0)
    assert np.all(rho[~inside] >= 1.0)
    # The boundary itself: rho = 1 exactly (marginal, sign-alternating).
    assert float(deltanet_spectral_radius(2.0, 1.0)) == pytest.approx(1.0, rel=0, abs=0)


def test_longhorn_unconditionally_stable():
    """rho = alpha/(alpha + ||k||^2) < 1 for every alpha > 0 and every key magnitude."""
    alphas = np.asarray([1e-3, 0.1, 1.0, 10.0])
    k_sq = np.logspace(-4, 8, 30)
    for alpha in alphas:
        rho = np.asarray(longhorn_spectral_radius(alpha, k_sq))
        assert np.all(rho < 1.0)
        assert np.all(rho > 0.0)
        # Monotone: larger keys are corrected harder (smaller rho).
        assert np.all(np.diff(rho) < 0)


def test_complement_identity():
    """rho_LH + beta_eff*||k||^2 = 1 exactly: the stable mass splits between the two."""
    alphas = np.asarray([0.1, 1.0, 10.0])
    k_sq = np.logspace(-2, 6, 17)
    for alpha in alphas:
        rho = np.asarray(longhorn_spectral_radius(alpha, k_sq))
        prod = np.asarray(longhorn_effective_beta_k_product(alpha, k_sq))
        assert_allclose(rho + prod, 1.0, rtol=0, atol=1e-15)
        assert np.all(prod < 1.0)  # never reaches the explicit boundary (let alone 2)


def test_rayleigh_argument_validation():
    with pytest.raises(ValueError):
        iteration_eigenvalue_along_k(jnp.zeros(8), 0.5)  # zero key
    with pytest.raises(ValueError):
        iteration_eigenvalue_along_k(jnp.ones((2, 4)), 0.5)  # not 1D
