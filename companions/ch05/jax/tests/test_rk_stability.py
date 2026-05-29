"""Assertion-based tests for the Chapter 5 stability-regions companion (0527-F26).

Pins the §5 stability claims: the explicit-RK stability function $R(z)$ equals the
method's exact rational/polynomial form (forward Euler $1+z$; classical RK4 the
degree-4 Taylor polynomial of $e^z$), the ``jax.vmap`` grid evaluation matches an
independent NumPy per-point solve, and the ZOH/bilinear closed forms are correct.

(File named test_rk_stability rather than test_stability_regions to avoid a
pytest basename collision with the Chapter 2 stability-regions test.)
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from companions.ch05.jax import stability_regions as sr

_SAMPLE_Z = np.array([-1.0 + 0j, 0.3 - 0.5j, -2.0 + 1.0j, 0.1 + 0.0j, -0.5 - 2.0j])


def _R_numpy(tab, z_arr: np.ndarray) -> np.ndarray:
    """Independent NumPy per-point solve oracle for explicit_stab_fn (vmap target)."""
    A, b, s = np.asarray(tab.A), np.asarray(tab.b), tab.s
    ones = np.ones(s)
    out = np.zeros_like(z_arr)
    for idx, zv in enumerate(z_arr.ravel()):
        kappa = np.linalg.solve(np.eye(s) - zv * A, ones)
        out.flat[idx] = 1.0 + zv * (b @ kappa)
    return out.reshape(z_arr.shape)


def test_forward_euler_is_one_plus_z() -> None:
    R = sr.explicit_stab_fn(sr.forward_euler_tableau())
    np.testing.assert_allclose(np.asarray(R(_SAMPLE_Z)), 1.0 + _SAMPLE_Z, atol=1e-12)


def test_rk4_is_degree4_taylor() -> None:
    """Classical RK4's R(z) is exactly the degree-4 Taylor polynomial of e^z."""
    z = _SAMPLE_Z
    taylor4 = 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0
    R = sr.explicit_stab_fn(sr.classical_rk4_tableau())
    np.testing.assert_allclose(np.asarray(R(z)), taylor4, atol=1e-12)


def test_vmap_matches_numpy_loop() -> None:
    """vmap grid evaluation equals the NumPy per-point solve, on a real 2-D grid."""
    re = np.linspace(-3.0, 1.0, 17)
    im = np.linspace(-3.0, 3.0, 19)
    Z = re[None, :] + 1j * im[:, None]
    R = sr.explicit_stab_fn(sr.classical_rk4_tableau())
    np.testing.assert_allclose(np.asarray(R(Z)), _R_numpy(sr.classical_rk4_tableau(), Z), atol=1e-12)


def test_zoh_is_exp() -> None:
    np.testing.assert_allclose(np.asarray(sr.zoh_stab_fn(_SAMPLE_Z)), np.exp(_SAMPLE_Z), atol=1e-12)


def test_bilinear_stable_iff_left_half_plane() -> None:
    assert float(jnp.abs(sr.bilinear_stab_fn(-1.0 + 0j))) < 1.0
    assert float(jnp.abs(sr.bilinear_stab_fn(1.0 + 0j))) > 1.0
    assert abs(float(jnp.abs(sr.bilinear_stab_fn(3j))) - 1.0) < 1e-12  # imag axis → |R| = 1
