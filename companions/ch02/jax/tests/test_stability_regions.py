"""Assertion-based tests for the Chapter 2 stability-regions companion (0527-F26).

Pins the §2.4 closed-form stability functions $R(z)$ at analytic points, the
bilinear/ZOH "imaginary-axis → unit-circle" property, and the JAX pole-masking
(``jnp.where(isfinite, ...)``) that replaces NumPy's ``np.errstate``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from companions.ch02.jax import stability_regions as sr


def test_forward_euler_unit_disk() -> None:
    assert float(jnp.abs(sr.forward_euler_R(-1.0 + 0j))) < 1e-12  # center → 0
    assert abs(float(jnp.abs(sr.forward_euler_R(-2.0 + 0j))) - 1.0) < 1e-12  # boundary |1+z|=1


def test_backward_euler_stable_at_minus_one() -> None:
    assert float(jnp.abs(sr.backward_euler_R(-1.0 + 0j))) == 0.5  # 1/(1-(-1)) = 1/2


def test_bilinear_maps_imag_axis_to_unit_circle() -> None:
    for y in (0.5, 2.0, 7.3):
        assert abs(float(jnp.abs(sr.bilinear_R(1j * y))) - 1.0) < 1e-12


def test_zoh_stable_iff_left_half_plane() -> None:
    assert float(jnp.abs(sr.zoh_R(-1.0 + 0j))) < 1.0  # Re(z) < 0 → stable
    assert float(jnp.abs(sr.zoh_R(1.0 + 0j))) > 1.0  # Re(z) > 0 → unstable
    assert abs(float(jnp.abs(sr.zoh_R(2j))) - 1.0) < 1e-12  # imaginary axis → |e^z| = 1


def test_stability_magnitude_masks_pole() -> None:
    """Backward Euler has a pole at z=1; the mask sends it to +inf, not NaN."""
    mag = np.asarray(sr.stability_magnitude(sr.backward_euler_R, jnp.array([1.0 + 0j])))
    assert np.isinf(mag[0])


def test_grid_shape() -> None:
    X, Y, Z = sr.stability_grid(resolution=64)
    assert X.shape == (64, 64) and Z.shape == (64, 64)
