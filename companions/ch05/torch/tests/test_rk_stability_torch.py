r"""Cross-framework parity: PyTorch Chapter 5 stability functions vs the JAX reference.

The torch numerical core ($R(z)$ for explicit RK methods, plus the ZOH and bilinear
closed forms) is fed the *same* inputs as the JAX core and the outputs are compared in
float64 (the cross-framework consistency goal, 0527-F27).

Pinned facts (mirroring ``companions/ch05/jax/tests/test_rk_stability.py``):

* forward Euler $R(z) = 1 + z$ (exact identity);
* classical RK4 $R(z)$ is the degree-4 Taylor polynomial of $e^z$ (exact identity);
* the batched ``torch.linalg.solve`` grid evaluation matches JAX's ``vmap`` solve on a
  real 2-D complex grid;
* ZOH $R(z) = e^z$ and bilinear $R(z) = (1+z/2)/(1-z/2)$ closed forms agree;
* bilinear is stable iff $\operatorname{Re}(z) \le 0$ ($|R|=1$ on the imaginary axis).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch05.jax import stability_regions as jax_sr  # noqa: E402
from companions.ch05.torch import stability_regions as torch_sr  # noqa: E402

_PARITY_TOL = 1e-9
_EXACT_TOL = 1e-12

_SAMPLE_Z = np.array([-1.0 + 0j, 0.3 - 0.5j, -2.0 + 1.0j, 0.1 + 0.0j, -0.5 - 2.0j])


# ---------------------------------------------------------------------------
# Exact-identity claims (mirror the JAX headline claims, < 1e-12)
# ---------------------------------------------------------------------------


def test_forward_euler_is_one_plus_z() -> None:
    """torch forward-Euler R(z) is exactly 1 + z (the JAX headline identity)."""
    R = torch_sr.explicit_stab_fn(torch_sr.forward_euler_tableau())
    got = R(_SAMPLE_Z).numpy()
    assert np.max(np.abs(got - (1.0 + _SAMPLE_Z))) < _EXACT_TOL


def test_rk4_is_degree4_taylor() -> None:
    """torch classical-RK4 R(z) is exactly the degree-4 Taylor polynomial of e^z."""
    z = _SAMPLE_Z
    taylor4 = 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0
    R = torch_sr.explicit_stab_fn(torch_sr.classical_rk4_tableau())
    got = R(z).numpy()
    assert np.max(np.abs(got - taylor4)) < _EXACT_TOL


def test_bilinear_stable_iff_left_half_plane() -> None:
    """torch bilinear: |R(-1)| < 1, |R(+1)| > 1, |R(3i)| = 1 (imag axis boundary)."""
    assert float(torch.abs(torch_sr.bilinear_stab_fn(-1.0 + 0j))) < 1.0
    assert float(torch.abs(torch_sr.bilinear_stab_fn(1.0 + 0j))) > 1.0
    assert abs(float(torch.abs(torch_sr.bilinear_stab_fn(3j))) - 1.0) < _EXACT_TOL


# ---------------------------------------------------------------------------
# JAX↔torch parity on identical inputs (< 1e-9)
# ---------------------------------------------------------------------------


def test_forward_euler_matches_jax() -> None:
    """torch forward-Euler R(z) reproduces the JAX R(z) on the sample points."""
    z = _SAMPLE_Z
    r_jax = np.asarray(jax_sr.explicit_stab_fn(jax_sr.forward_euler_tableau())(jnp.asarray(z)))
    r_torch = torch_sr.explicit_stab_fn(torch_sr.forward_euler_tableau())(z).numpy()
    assert np.max(np.abs(r_jax - r_torch)) < _PARITY_TOL


def test_rk4_matches_jax() -> None:
    """torch classical-RK4 R(z) reproduces the JAX R(z) on the sample points."""
    z = _SAMPLE_Z
    r_jax = np.asarray(jax_sr.explicit_stab_fn(jax_sr.classical_rk4_tableau())(jnp.asarray(z)))
    r_torch = torch_sr.explicit_stab_fn(torch_sr.classical_rk4_tableau())(z).numpy()
    assert np.max(np.abs(r_jax - r_torch)) < _PARITY_TOL


def test_midpoint_rk2_matches_jax() -> None:
    """torch midpoint-RK2 R(z) reproduces the JAX R(z) on the sample points."""
    z = _SAMPLE_Z
    r_jax = np.asarray(jax_sr.explicit_stab_fn(jax_sr.midpoint_rk2_tableau())(jnp.asarray(z)))
    r_torch = torch_sr.explicit_stab_fn(torch_sr.midpoint_rk2_tableau())(z).numpy()
    assert np.max(np.abs(r_jax - r_torch)) < _PARITY_TOL


def test_rkf45_matches_jax() -> None:
    """torch RKF 4(5) R(z) reproduces the JAX R(z) (6-stage tableau, identical entries)."""
    z = _SAMPLE_Z
    r_jax = np.asarray(jax_sr.explicit_stab_fn(jax_sr.rkf45_tableau())(jnp.asarray(z)))
    r_torch = torch_sr.explicit_stab_fn(torch_sr.rkf45_tableau())(z).numpy()
    assert np.max(np.abs(r_jax - r_torch)) < _PARITY_TOL


def test_grid_evaluation_matches_jax() -> None:
    """torch batched-solve grid evaluation matches JAX's vmap solve on a 2-D complex grid.

    This is the torch mirror of the JAX ``test_vmap_matches_numpy_loop``: the JAX side is
    the oracle here, exercising the same RK4 stability function over a full complex grid.
    """
    re = np.linspace(-3.0, 1.0, 17)
    im = np.linspace(-3.0, 3.0, 19)
    Z = re[None, :] + 1j * im[:, None]
    r_jax = np.asarray(jax_sr.explicit_stab_fn(jax_sr.classical_rk4_tableau())(jnp.asarray(Z)))
    r_torch = torch_sr.explicit_stab_fn(torch_sr.classical_rk4_tableau())(Z).numpy()
    assert r_torch.shape == Z.shape
    assert np.max(np.abs(r_jax - r_torch)) < _PARITY_TOL


def test_rk4_region_boundary_crossing_matches_jax() -> None:
    """Headline stability-region claim: RK4 |R(z)| crosses 1 on the negative real axis.

    RK4's real-axis stability limit is z ≈ -2.7853; this pins |R| < 1 inside and > 1
    outside, and confirms the torch |R(z)| grid matches JAX along that axis.
    """
    x = np.linspace(-3.5, 0.0, 71)
    z = x.astype(np.complex128)
    R_jax = jax_sr.explicit_stab_fn(jax_sr.classical_rk4_tableau())
    R_torch = torch_sr.explicit_stab_fn(torch_sr.classical_rk4_tableau())
    mod_jax = np.abs(np.asarray(R_jax(jnp.asarray(z))))
    mod_torch = np.abs(R_torch(z).numpy())
    assert np.max(np.abs(mod_jax - mod_torch)) < _PARITY_TOL
    # Boundary at z ≈ -2.7853: inside (|z| smaller) stable, outside unstable.
    assert float(torch.abs(R_torch(-2.5 + 0j))) < 1.0
    assert float(torch.abs(R_torch(-3.0 + 0j))) > 1.0


def test_zoh_matches_jax() -> None:
    """torch ZOH/exp-trap R(z) = e^z reproduces the JAX closed form."""
    z = _SAMPLE_Z
    r_jax = np.asarray(jax_sr.zoh_stab_fn(jnp.asarray(z)))
    r_torch = torch_sr.zoh_stab_fn(z).numpy()
    assert np.max(np.abs(r_jax - r_torch)) < _PARITY_TOL
    # And exactly e^z.
    assert np.max(np.abs(r_torch - np.exp(z))) < _EXACT_TOL


def test_bilinear_matches_jax() -> None:
    """torch bilinear R(z) reproduces the JAX closed form on the sample points."""
    z = _SAMPLE_Z
    r_jax = np.asarray(jax_sr.bilinear_stab_fn(jnp.asarray(z)))
    r_torch = torch_sr.bilinear_stab_fn(z).numpy()
    assert np.max(np.abs(r_jax - r_torch)) < _PARITY_TOL
