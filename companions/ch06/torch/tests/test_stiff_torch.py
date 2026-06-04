r"""Cross-framework parity: PyTorch Chapter 6 stiff demo vs the JAX reference (§6.1).

The torch integrators are fed the *same* inputs as the JAX integrators and the
van der Pol trajectories are compared in float64, so the two frameworks stay in
lock-step. The headline §6.1 claims are also re-pinned on the torch side: the
analytic Jacobian matches the JAX ``jacfwd`` Jacobian, backward Euler stays
bounded at a coarse dt where explicit RK4 diverges, and the damped-Newton solve
converges along the whole trajectory.
"""

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

torch = pytest.importorskip("torch")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch06.jax import stiff_demo as jax_stiff  # noqa: E402
from companions.ch06.torch import stiff_demo as torch_stiff  # noqa: E402

_PARITY_TOL = 1e-9
_MU = torch_stiff._MU


# ---------------------------------------------------------------------------
# §6.1 — Jacobian / RHS parity
# ---------------------------------------------------------------------------


def test_vdp_rhs_matches_jax() -> None:
    """torch vdp_rhs reproduces the JAX RHS (§6.1)."""
    for h in ([2.0, 0.0], [0.5, -1.3], [-1.0, 2.0]):
        rhs_j = np.asarray(jax_stiff.vdp_rhs(jnp.array(h)))
        rhs_t = torch_stiff.vdp_rhs(torch.tensor(h, dtype=torch.float64)).numpy()
        assert np.max(np.abs(rhs_j - rhs_t)) < 1e-12


def test_vdp_jacobian_matches_jax_jacfwd() -> None:
    """torch analytic Jacobian reproduces the JAX ``jacfwd`` Jacobian exactly (§6.1).

    This is the parity counterpart of the JAX test_jacfwd_matches_handcoded: the
    closed-form Jacobian used in the torch implicit solve equals JAX's autodiff one.
    """
    for h in ([2.0, 0.0], [0.5, -1.3], [-1.0, 2.0]):
        jac_j = np.asarray(jax_stiff._vdp_jac(jnp.array(h)))
        jac_t = torch_stiff.vdp_jacobian(torch.tensor(h, dtype=torch.float64)).numpy()
        assert np.max(np.abs(jac_j - jac_t)) < 1e-12


# ---------------------------------------------------------------------------
# §6.1 — trajectory parity (identical inputs -> identical trajectories)
# ---------------------------------------------------------------------------


def test_backward_euler_trajectory_matches_jax() -> None:
    """torch backward Euler reproduces the JAX trajectory in the stable regime (§6.1).

    Compared at dt=0.05, where the van der Pol relaxation jumps are resolved and the
    integration is well-conditioned, so the two frameworks agree to float64. At the
    *coarse* dt=0.2 the dynamics sit on the fast-jump edge and amplify 1-ULP
    XLA-vs-eager roundoff (a single 2e-16 step difference blows up to O(1) within a
    couple of steps), so the JAX suite itself only pins *boundedness* there — mirrored
    in test_be_stable_at_coarse_dt_torch — not exact values.
    """
    _, hs_j = jax_stiff.simulate_be(np.array([2.0, 0.0]), 0.05, 50.0)
    _, hs_t = torch_stiff.simulate_be(np.array([2.0, 0.0]), 0.05, 50.0)
    assert np.max(np.abs(np.asarray(hs_j) - hs_t)) < _PARITY_TOL


def test_rk4_trajectory_matches_jax() -> None:
    """torch RK4 reproduces the JAX trajectory at a stable dt (§6.1 parity).

    Compared at dt=0.005 where neither framework diverges, so the comparison is a
    clean float64 equality (the divergent coarse-dt case is checked by NaN
    structure in test_rk4_blows_up_at_coarse_dt_torch).
    """
    _, hs_j = jax_stiff.simulate_rk4(np.array([2.0, 0.0]), 0.005, 50.0)
    _, hs_t = torch_stiff.simulate_rk4(np.array([2.0, 0.0]), 0.005, 50.0)
    assert np.all(np.isfinite(np.asarray(hs_j))) and np.all(np.isfinite(hs_t))
    assert np.max(np.abs(np.asarray(hs_j) - hs_t)) < _PARITY_TOL


def test_rk4_nan_mask_matches_jax() -> None:
    """torch and JAX mask the same diverged entries to NaN at coarse dt (§6.1 parity).

    The NaN footprint is identical (both diverge at the same step). The finite
    prefix is compared with a *relative* tolerance: the last few pre-blowup entries
    reach magnitude ~1e3-1e4, where XLA-fused vs torch-eager roundoff legitimately
    exceeds an absolute 1e-9 while the relative agreement stays ~1e-12.
    """
    _, hs_j = jax_stiff.simulate_rk4(np.array([2.0, 0.0]), 0.2, 50.0)
    _, hs_t = torch_stiff.simulate_rk4(np.array([2.0, 0.0]), 0.2, 50.0)
    hs_j = np.asarray(hs_j)
    # Same NaN footprint.
    assert np.array_equal(np.isnan(hs_j), np.isnan(hs_t))
    # Finite prefix agrees to float64 (relative, since values diverge toward blowup).
    finite = np.isfinite(hs_j) & np.isfinite(hs_t)
    np.testing.assert_allclose(hs_t[finite], hs_j[finite], rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# §6.1 — headline stability claims, re-pinned on the torch side
# ---------------------------------------------------------------------------


def test_backward_euler_residual_torch() -> None:
    """Mirror of the JAX headline: damped Newton converges along the whole trajectory.

    Checked at dt=0.05, the well-resolved regime that the JAX oracle test
    (``test_scan_matches_numpy_newton``) also uses — here the implicit solve reaches
    the $10^{-10}$ tolerance at every step. (At the coarse dt=0.2 the solve at the
    van der Pol fast-jump edge stalls at residual ~0.25 for *both* frameworks on
    their respective roundoff-perturbed paths, so the JAX suite pins only
    boundedness there, not residual — see test_be_stable_at_coarse_dt_torch.)
    """
    dt = 0.05
    _, hs = torch_stiff.simulate_be(np.array([2.0, 0.0]), dt, 50.0)
    hs_t = torch.tensor(hs, dtype=torch.float64)
    max_res = max(
        float(torch.linalg.norm(hs_t[k + 1] - hs_t[k] - dt * torch_stiff.vdp_rhs(hs_t[k + 1])))
        for k in range(len(hs) - 1)
    )
    assert max_res < 1e-8, f"BE max residual {max_res:.2e} at dt={dt} — Newton under-converged"


def test_be_stable_at_coarse_dt_torch() -> None:
    """Mirror of the JAX headline: BE stays on the bounded limit cycle at coarse dt.

    Position q is bounded (~2); the velocity p legitimately spikes to ~μ during the
    fast relaxation jumps, so only q is range-checked.
    """
    _, hs = torch_stiff.simulate_be(np.array([2.0, 0.0]), 0.2, 50.0)
    assert np.all(np.isfinite(hs)), "backward Euler must not diverge at coarse dt"
    assert np.max(np.abs(hs[:, 0])) < 3.0, "BE position should stay on the bounded cycle"


def test_rk4_blows_up_at_coarse_dt_torch() -> None:
    """Mirror of the JAX headline: explicit RK4 diverges (NaN-masked) at dt=0.2."""
    _, hs = torch_stiff.simulate_rk4(np.array([2.0, 0.0]), 0.2, 50.0)
    assert np.any(np.isnan(hs)), "RK4 should diverge (NaN-masked) at the coarse dt"


def test_simulate_validation_torch() -> None:
    """Input guards mirror the JAX-side validation contract."""
    with pytest.raises(ValueError):
        torch_stiff.simulate_rk4(np.array([2.0, 0.0]), 0.0, 10.0)
    with pytest.raises(ValueError):
        torch_stiff.simulate_be(np.array([2.0, 0.0]), 0.1, -1.0)
