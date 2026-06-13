r"""Tests for ch13 generalized_transition: the diagonal-plus-rank-one transition.

Pins the §13.2-13.3 claims:

* **P1 spectrum.** $A = \mathrm{Diag}(w) - c\,a a^\top$ is symmetric; ``eigvalsh``
  matches the scalar-diagonal closed form $\{w_0, w_0 - c\}$ exactly, the secular
  function zeroes at the general-diagonal eigenvalues, and the spectrum interlaces
  the diagonal. At $w_0 = 1$ the moving eigenvalue $1 - c$ reproduces Chapter 12's
  $k$-direction value with $c = \beta\|k\|^2$.
* **P3 reduction.** The generalized rule with $(w,a,c,u,b)$ from
  ``gated_delta_reduction`` reproduces ``gated_delta_recurrent`` to ``< 1e-12``.
* **scan == naive oracle.** The rank-one ``lax.scan`` form equals the
  materialised-transition Python loop to ``< 1e-12``.
* **Decoupled eviction.** Targeted removal evicts an old key as exactly
  $(1-c)^T\|v_A\|$; the key-locked policy leaves it flat at $\|v_A\|$.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from companions.ch12.jax.gated_delta import gated_delta_recurrent  # noqa: E402
from companions.ch12.jax.stability import deltanet_spectral_radius  # noqa: E402
from companions.ch13.jax.generalized_transition import (  # noqa: E402
    decoupled_eviction,
    dplr_transition,
    gated_delta_reduction,
    generalized_delta_naive,
    generalized_delta_recurrent,
    generalized_delta_step,
    scalar_diagonal_spectrum,
    secular_function,
    transition_spectrum,
)


def _unit(rng, d):
    a = rng.standard_normal(d)
    return jnp.asarray(a / np.linalg.norm(a))


def _gen_stream(length=40, d_k=6, d_v=5, seed=0):
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((length, d_k)))
    w = jnp.asarray(rng.uniform(0.7, 1.0, size=(length, d_k)))
    a_raw = rng.standard_normal((length, d_k))
    a = jnp.asarray(a_raw / np.linalg.norm(a_raw, axis=1, keepdims=True))
    c = jnp.asarray(rng.uniform(0.0, 0.5, size=length))
    u = jnp.asarray(rng.standard_normal((length, d_v)))
    b = jnp.asarray(rng.standard_normal((length, d_k)))
    return q, w, a, c, u, b


def _ch12_stream(length=40, d_k=6, d_v=5, seed=0):
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((length, d_k)))
    k_raw = rng.standard_normal((length, d_k))
    k = jnp.asarray(k_raw / np.linalg.norm(k_raw, axis=1, keepdims=True))
    v = jnp.asarray(rng.standard_normal((length, d_v)))
    betas = jnp.asarray(rng.uniform(0.1, 0.9, size=length))
    gammas = jnp.asarray(rng.uniform(0.7, 1.0, size=length))
    return q, k, v, betas, gammas


# --- §13.3 scan == materialised-transition oracle ---------------------------


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_scan_equals_naive_oracle(seed):
    q, w, a, c, u, b = _gen_stream(seed=seed)
    y_scan, s_scan = generalized_delta_recurrent(q, w, a, c, u, b)
    y_naive, s_naive = generalized_delta_naive(q, w, a, c, u, b)
    assert_allclose(np.asarray(y_scan), np.asarray(y_naive), rtol=0, atol=1e-12)
    assert_allclose(np.asarray(s_scan), np.asarray(s_naive), rtol=0, atol=1e-12)


def test_step_matches_single_step_recurrent():
    """One generalized_delta_step equals an L=1 recurrent pass (read = S_1 q_0)."""
    q, w, a, c, u, b = _gen_stream(length=1, seed=3)
    s1 = generalized_delta_step(
        jnp.zeros((u.shape[1], w.shape[1])), w[0], a[0], float(c[0]), u[0], b[0]
    )
    _, s_rec = generalized_delta_recurrent(q, w, a, c, u, b)
    assert_allclose(np.asarray(s1), np.asarray(s_rec), rtol=0, atol=1e-12)


# --- §13.3 P3: reduction to Chapter 12's gated DeltaNet ---------------------


@pytest.mark.parametrize("seed", [0, 2, 5])
def test_p3_reduction_to_gated_delta(seed):
    q, k, v, betas, gammas = _ch12_stream(seed=seed)
    y_gen, s_gen = gated_delta_reduction(q, k, v, betas, gammas)
    y_ch12, s_ch12 = gated_delta_recurrent(q, k, v, betas, gammas)
    # Tight pin backing the prose's 8.9e-16 (measured ~8.88e-16): an exact
    # algebraic identity computed two ways, so float roundoff only.
    assert_allclose(np.asarray(y_gen), np.asarray(y_ch12), rtol=0, atol=1e-13)
    assert_allclose(np.asarray(s_gen), np.asarray(s_ch12), rtol=0, atol=1e-13)


# --- §13.2 P1: the spectrum ------------------------------------------------


def test_transition_is_symmetric():
    rng = np.random.default_rng(0)
    w = jnp.asarray(rng.uniform(0.5, 1.0, size=6))
    a = _unit(rng, 6)
    A = dplr_transition(w, a, 0.7)
    assert_allclose(np.asarray(A), np.asarray(A).T, rtol=0, atol=1e-15)


@pytest.mark.parametrize("c", [0.0, 0.5, 1.0, 2.0, 2.5])
def test_scalar_diagonal_spectrum_matches_eigvalsh(c):
    """Scalar diagonal: eigvalsh == closed form {w0-c, w0 (x d-1)}, any a direction."""
    rng = np.random.default_rng(1)
    d = 6
    a = _unit(rng, d)
    spec = transition_spectrum(jnp.ones(d), a, c)
    closed = scalar_diagonal_spectrum(1.0, c, d)
    # Tight pin backing the caption's measured 6.7e-16 (eigvalsh roundoff).
    assert_allclose(np.asarray(spec), np.asarray(closed), rtol=0, atol=1e-13)


def test_scalar_diagonal_moving_eigenvalue_pins():
    """The moving eigenvalue is exactly w0 - c; the rest sit at w0 (mult d-1)."""
    d = 6
    for c, moving in ((0.5, 0.5), (1.0, 0.0), (2.0, -1.0)):
        spec = np.asarray(scalar_diagonal_spectrum(1.0, c, d))
        assert_allclose(spec.min(), moving, rtol=0, atol=1e-12)
        # exactly one eigenvalue differs from w0 = 1 (when c != 0)
        at_w0 = np.isclose(spec, 1.0, rtol=0, atol=1e-12).sum()
        assert at_w0 == (d if c == 0.0 else d - 1)


def test_c_equals_two_is_the_stability_boundary():
    """w0 = 1, c = 2: the moving eigenvalue hits exactly -1 (rho = 1)."""
    spec = np.asarray(transition_spectrum(jnp.ones(6), _unit(np.random.default_rng(4), 6), 2.0))
    assert_allclose(spec.min(), -1.0, rtol=0, atol=1e-12)


def test_ch12_correspondence_at_w0_one():
    """At w0 = 1, |1 - c| == DeltaNet's |1 - beta||k||^2| with c = beta*||k||^2."""
    for beta, ksq in ((0.3, 1.0), (0.5, 1.7), (1.2, 0.9)):
        c = beta * ksq
        moving = np.asarray(scalar_diagonal_spectrum(1.0, c, 5)).min()
        assert_allclose(abs(moving), float(deltanet_spectral_radius(beta, ksq)), rtol=0, atol=1e-12)


@pytest.mark.parametrize("c", [0.3, 0.8, 1.5])
def test_secular_zeroes_at_eigenvalues(c):
    """Each computed eigenvalue (not equal to a diagonal entry) zeroes the secular fn."""
    rng = np.random.default_rng(2)
    d = 6
    w = jnp.asarray(np.sort(rng.uniform(0.55, 1.0, size=d)))
    a = _unit(rng, d)
    spec = np.asarray(transition_spectrum(w, a, c))
    w_np = np.asarray(w)
    worst = max(
        abs(float(secular_function(float(lam), w, a, c)))
        for lam in spec
        if np.min(np.abs(w_np - lam)) > 1e-6
    )
    assert worst < 1e-9


@pytest.mark.parametrize("c", [0.3, 0.8, 1.5, 2.2])
def test_spectrum_interlaces_diagonal(c):
    """A rank-one downdate keeps eigenvalues in [min(w) - c, max(w)]."""
    rng = np.random.default_rng(3)
    d = 6
    w = jnp.asarray(rng.uniform(0.55, 1.0, size=d))
    a = _unit(rng, d)
    spec = np.asarray(transition_spectrum(w, a, c))
    assert spec.max() <= float(jnp.max(w)) + 1e-12
    assert spec.min() >= float(jnp.min(w)) - c - 1e-12


# --- §13.3 decoupled eviction (the learned-direction payoff + figure pins) --


@pytest.mark.parametrize("c,T", [(0.1, 30), (0.2, 20), (0.05, 50)])
def test_decoupled_eviction_analytic(c, T):
    """Targeted removal decays the old key as exactly (1-c)^T ||v_A||."""
    targeted, analytic, _ = decoupled_eviction(T, c)
    assert_allclose(targeted, analytic, rtol=0, atol=1e-12)


def test_locked_policy_is_flat():
    """Chapter 12's key-locked removal leaves k_A untouched: flat at ||v_A|| = T=0 norm."""
    base_t, _, base_l = decoupled_eviction(0, 0.1)
    assert_allclose(base_t, base_l, rtol=0, atol=1e-12)  # at T=0 both = ||v_A||
    for T in (10, 30, 90):
        _, _, locked = decoupled_eviction(T, 0.1)
        assert_allclose(locked, base_l, rtol=0, atol=1e-12)


def test_eviction_figure_caption_pins():
    """The exact numbers the learned-direction figure caption quotes (seed 0)."""
    targeted, analytic, locked = decoupled_eviction(30, 0.1)
    assert_allclose(targeted, 0.028576, rtol=0, atol=1e-6)
    assert_allclose(analytic, 0.028576, rtol=0, atol=1e-6)
    assert_allclose(locked, 0.674096, rtol=0, atol=1e-6)  # ||v_A||, the flat baseline


# --- validation -------------------------------------------------------------


def test_dplr_rejects_bad_shapes():
    with pytest.raises(ValueError):
        dplr_transition(jnp.ones(4), jnp.ones(5), 0.1)
    with pytest.raises(ValueError):
        dplr_transition(jnp.ones(4), jnp.ones(4), -0.1)


def test_scalar_diagonal_spectrum_rejects_bad_d():
    with pytest.raises(ValueError):
        scalar_diagonal_spectrum(1.0, 0.5, 0)


def test_generalized_recurrent_rejects_shape_mismatch():
    q, w, a, c, u, b = _gen_stream(seed=0)
    with pytest.raises(ValueError):
        generalized_delta_recurrent(q, w, a, c[:-1], u, b)  # c wrong length


def test_decoupled_eviction_rejects_bad_c():
    with pytest.raises(ValueError):
        decoupled_eviction(10, 1.5)
    with pytest.raises(ValueError):
        decoupled_eviction(-1, 0.1)
