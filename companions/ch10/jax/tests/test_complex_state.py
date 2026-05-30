r"""Tests for ch10 complex_state: RoPE<->complex equivalence and spiral decay.

Pins the §10.4 claims:

* a 2-D RoPE rotation on the real state equals complex multiplication
  (`ch10:rope-complex-equivalence`) to machine precision, driven and homogeneous;
* a homogeneous complex mode decays geometrically at rate $\log\rho$ and rotates
  by $\theta$ per step.
"""

from __future__ import annotations

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from companions.ch10.jax.complex_state import (  # noqa: E402
    complex_scalar_recurrence,
    complex_to_real2,
    decay_rate,
    rope_equivalence_residual,
    rope_matrix,
)

_RHO = 0.95
_THETA = math.pi / 9.0


def test_rope_equivalence_driven():
    """RoPE 2-D recurrence == complex recurrence under a random complex drive."""
    rng = np.random.default_rng(1)
    drive = jnp.asarray(rng.standard_normal(50) + 1j * rng.standard_normal(50))
    assert rope_equivalence_residual(_RHO, _THETA, drive) < 1e-12


def test_rope_equivalence_homogeneous():
    """With zero drive the equivalence still holds (pure transition)."""
    drive = jnp.zeros(30, dtype=jnp.complex128).at[0].set(1.0 + 1.0j)
    assert rope_equivalence_residual(_RHO, _THETA, drive) < 1e-12


def test_rope_matrix_is_rotation():
    """R(theta) is orthogonal with det 1 (a proper rotation)."""
    R = rope_matrix(_THETA)
    assert_allclose(np.asarray(R.T @ R), np.eye(2), rtol=0, atol=1e-14)
    assert_allclose(float(jnp.linalg.det(R)), 1.0, rtol=0, atol=1e-14)


def test_rope_matrix_matches_complex_multiply():
    """R(theta) @ [a,b] == complex (cos+isin)(a+bi), elementwise identity."""
    z = 0.7 - 0.4j
    R = rope_matrix(_THETA)
    got = R @ complex_to_real2(jnp.asarray(z))
    want = jnp.exp(1j * _THETA) * z
    assert_allclose(np.asarray(got), np.asarray([want.real, want.imag]), rtol=0, atol=1e-14)


def test_spiral_decay_rate_matches_log_rho():
    """Homogeneous mode magnitude decays at exactly log(rho) per step."""
    xs = complex_scalar_recurrence(_RHO, _THETA, 80)
    assert_allclose(decay_rate(xs), math.log(_RHO), rtol=0, atol=1e-12)


def test_spiral_phase_advances_by_theta():
    """Consecutive points differ in phase by theta (the rotation)."""
    xs = complex_scalar_recurrence(_RHO, _THETA, 20)
    phases = jnp.angle(xs)
    dphase = jnp.diff(phases)
    # unwrap-safe: compare cos/sin to avoid +-2pi ambiguity
    assert_allclose(np.asarray(jnp.cos(dphase)), math.cos(_THETA), rtol=0, atol=1e-12)
    assert_allclose(np.asarray(jnp.sin(dphase)), math.sin(_THETA), rtol=0, atol=1e-12)


def test_undamped_mode_preserves_magnitude():
    """rho=1 is a pure rotation: |x_k| constant."""
    xs = complex_scalar_recurrence(1.0, _THETA, 40)
    assert_allclose(np.asarray(jnp.abs(xs)), 1.0, rtol=0, atol=1e-13)


def test_recurrence_rejects_unstable_rho():
    with pytest.raises(ValueError):
        complex_scalar_recurrence(1.01, _THETA, 10)
    with pytest.raises(ValueError):
        complex_scalar_recurrence(0.0, _THETA, 10)
