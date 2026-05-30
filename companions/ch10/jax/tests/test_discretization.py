r"""Tests for ch10 discretization: order of accuracy, homogeneous-blindness, stability.

These pin the §10.2-10.3 claims the prose cites:

* exp-trapezoidal is globally second-order on a FORCED system (slope ~2), ZOH is
  first-order (slope ~1);
* on the HOMOGENEOUS system the exponential transition is exact for both, so the
  order is invisible (both at roundoff) — the chapter's load-bearing subtlety;
* ZOH and exp-trapezoidal are A-stable with $\alpha = e^{A\Delta} \to 0$ on stiff
  modes; bilinear is A-stable but $\alpha \to -1$ (undamped); forward Euler is not
  A-stable.

All ``assert_allclose`` use ``rtol=0`` so the absolute tolerance is the real gate
(numpy's default ``rtol=1e-7`` would silently loosen the homogeneous check ~1e5x).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from companions.ch10.jax.discretization import (  # noqa: E402
    amplification,
    discretize_bilinear,
    discretize_exp_trapezoidal,
    discretize_zoh,
    forced_exact,
    global_error,
    integrate,
    order_sweep,
)

# A complex decaying-oscillating mode: a faithful single Mamba-3 eigenvalue.
_A = -0.5 + 2.0j
_OMEGA = 1.3
_T = 6.0
_X0 = 1.0 + 0.0j
_DTS = jnp.asarray([0.2, 0.1, 0.05, 0.025, 0.0125])


# --- §10.2 order of accuracy (the headline) --------------------------------


def test_zoh_first_order_forced():
    _, slope = order_sweep("zoh", _A, _DTS, _T, _OMEGA, x0=_X0)
    assert 0.9 < slope < 1.15, f"ZOH forced slope {slope} not ~1"


def test_exp_trapezoidal_second_order_forced():
    _, slope = order_sweep("exp_trapezoidal", _A, _DTS, _T, _OMEGA, x0=_X0)
    assert 1.9 < slope < 2.1, f"exp-trap forced slope {slope} not ~2"


def test_bilinear_second_order_forced():
    _, slope = order_sweep("bilinear", _A, _DTS, _T, _OMEGA, x0=_X0)
    assert 1.9 < slope < 2.1, f"bilinear forced slope {slope} not ~2"


def test_exp_trapezoidal_beats_zoh_on_forced():
    """At a fixed step size the second-order scheme is strictly more accurate."""
    e_zoh = global_error("zoh", _A, 0.05, _T, _OMEGA, x0=_X0)
    e_trap = global_error("exp_trapezoidal", _A, 0.05, _T, _OMEGA, x0=_X0)
    assert e_trap < e_zoh, f"exp-trap ({e_trap:.2e}) should beat ZOH ({e_zoh:.2e})"


# --- §10.2 THE load-bearing subtlety: homogeneous-blindness ----------------


def test_homogeneous_order_is_invisible():
    """On u=0 the exponential transition is exact, so ZOH == exp-trap to roundoff.

    This is why Mamba-3's order-2 claim can only be verified on a forced system
    (cf. Ch 4 Ex 4.3). Both schemes must hit machine precision regardless of order.
    """
    e_zoh = global_error("zoh", _A, 0.1, _T, None, x0=_X0)
    e_trap = global_error("exp_trapezoidal", _A, 0.1, _T, None, x0=_X0)
    assert e_zoh < 1e-12, f"ZOH homogeneous error {e_zoh:.2e} not at roundoff"
    assert e_trap < 1e-12, f"exp-trap homogeneous error {e_trap:.2e} not at roundoff"


def test_homogeneous_transition_identical_zoh_exptrap():
    """The alpha coefficients themselves are bit-for-bit the same (both e^{A dt})."""
    a_zoh, _ = discretize_zoh(_A, 0.1)
    a_trap, _, _ = discretize_exp_trapezoidal(_A, 0.1)
    assert_allclose(np.asarray(a_zoh), np.asarray(a_trap), rtol=0, atol=0)


def test_homogeneous_integration_matches_exact_state():
    """Direct check: integrate(u=0) reproduces x0 e^{A t} for both exponential schemes."""
    n_steps = 60
    dt = 0.1
    t = jnp.arange(n_steps + 1) * dt
    exact = _X0 * jnp.exp(_A * t)
    for scheme in ("zoh", "exp_trapezoidal"):
        xs = integrate(scheme, _A, dt, n_steps, None, x0=_X0)
        assert_allclose(np.asarray(xs), np.asarray(exact), rtol=0, atol=1e-12)


# --- §10.2 coefficient identities ------------------------------------------


def test_exp_trapezoidal_lambda_one_is_shifted_zoh():
    """lam=1 degenerates to a first-order shifted ZOH (gamma=dt, beta=0)."""
    alpha, beta, gamma = discretize_exp_trapezoidal(_A, 0.1, lam=1.0)
    assert_allclose(np.asarray(beta), 0.0, rtol=0, atol=1e-15)
    assert_allclose(np.asarray(gamma), 0.1, rtol=0, atol=1e-15)


def test_exp_trapezoidal_symmetric_weights_at_half():
    """lam=1/2: beta = (dt/2) alpha, gamma = dt/2 — the symmetric trapezoid."""
    dt = 0.1
    alpha, beta, gamma = discretize_exp_trapezoidal(_A, dt, lam=0.5)
    assert_allclose(np.asarray(gamma), dt / 2, rtol=0, atol=1e-15)
    assert_allclose(np.asarray(beta), np.asarray((dt / 2) * alpha), rtol=0, atol=1e-15)


def test_schemes_agree_as_dt_to_zero():
    """All three discretizations converge to the same map as dt -> 0."""
    dt = 1e-6
    a_zoh, b_zoh = discretize_zoh(_A, dt)
    a_bl, b_bl = discretize_bilinear(_A, dt)
    a_tr, b_tr, g_tr = discretize_exp_trapezoidal(_A, dt)
    # transitions agree to O(dt^2); input weights (total) agree to O(dt).
    assert_allclose(np.asarray(a_zoh), np.asarray(a_tr), rtol=0, atol=1e-10)
    assert abs(complex(a_bl) - complex(a_zoh)) < 1e-10
    assert abs((complex(b_tr) + complex(g_tr)) - complex(b_zoh)) < 1e-8


def test_zoh_beta_limit_at_zero_mode():
    """beta -> dt as A -> 0 (L'Hopital guard)."""
    _, beta = discretize_zoh(jnp.asarray(0.0 + 0.0j), 0.3)
    assert_allclose(np.asarray(beta).real, 0.3, rtol=0, atol=1e-12)


def test_forced_exact_solves_the_ode():
    """forced_exact satisfies x' = A x + sin(omega t) by finite-difference check."""
    A = -0.7 + 1.1j
    t = jnp.linspace(0.0, 3.0, 2001)
    x = forced_exact(A, _OMEGA, t, x0=0.3 + 0.0j)
    dxdt = jnp.gradient(x, t)
    residual = dxdt - (A * x + jnp.sin(_OMEGA * t))
    # central differences are O(h^2); interior residual should be tiny.
    assert float(jnp.max(jnp.abs(residual[5:-5]))) < 1e-4


# --- §10.3 stability --------------------------------------------------------


def test_exponential_schemes_a_stable_whole_lhp():
    """|e^z| <= 1 for every z with Re z <= 0 (A-stability of ZOH / exp-trap)."""
    rng = np.random.default_rng(0)
    re = -rng.random(500) * 50  # Re z in [-50, 0]
    im = (rng.random(500) - 0.5) * 100
    z = jnp.asarray(re + 1j * im)
    assert bool(jnp.all(jnp.abs(amplification("exp_trapezoidal", z)) <= 1.0 + 1e-12))


def test_stiff_mode_damping_contrast():
    """Stiff mode: exp schemes -> 0, bilinear -> 1, forward Euler blows up."""
    z = jnp.asarray(-50.0)
    assert float(jnp.abs(amplification("exp_trapezoidal", z))) < 1e-12
    assert float(jnp.abs(amplification("bilinear", z))) > 0.9  # ~ (1-25)/(1+25)=0.92
    assert float(jnp.abs(amplification("forward_euler", z))) > 1.0  # |1-50| = 49, unstable


def test_forward_euler_not_a_stable():
    """Forward Euler is unstable outside its disk |1+z| <= 1."""
    z = jnp.asarray(-3.0)  # Re z < 0 but outside the disk
    assert float(jnp.abs(amplification("forward_euler", z))) > 1.0


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        amplification("midpoint", jnp.asarray(-1.0))
    with pytest.raises(ValueError):
        integrate("midpoint", _A, 0.1, 4, _OMEGA)


def test_exp_trapezoidal_rejects_bad_lambda():
    with pytest.raises(ValueError):
        discretize_exp_trapezoidal(_A, 0.1, lam=1.5)
