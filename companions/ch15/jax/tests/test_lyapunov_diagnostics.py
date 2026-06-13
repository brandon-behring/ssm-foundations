r"""Tests for Chapter 15 §§15.4–15.5 — Lyapunov + regime diagnostics (P2′, P3′).

Pins, against *independent* ground truth (eigendecomposition / closed form, not the
QR iteration), every number the §15.4–15.5 figures and prose cite:

* **P2′** recovery (DPLR / LTV / S4D-Lin), the divergence identity, and the
  resolution-limit caveat on a non-normal degenerate recurrence (the ring);
* **P3′** the two-route marginal-mode count and the effective-state-size closed form.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch01.jax.coupled_oscillators import build_ring_state_matrix  # noqa: E402
from companions.ch09.jax.selective_ssm import discretize_selective  # noqa: E402
from companions.ch15.jax import lyapunov_diagnostics as ld  # noqa: E402


def _benettin_numpy(jacobians: np.ndarray, n_steps: int) -> np.ndarray:
    """Independent NumPy Benettin QR loop — the oracle for the lax.scan engine."""
    T, N, _ = jacobians.shape
    Q = np.eye(N)
    acc = np.zeros(N)
    for t in range(n_steps):
        Q, R = np.linalg.qr(jacobians[t % T] @ Q)
        s = np.sign(np.diag(R))
        s[s == 0] = 1.0
        Q = Q * s[np.newaxis, :]
        R = s[:, np.newaxis] * R
        acc += np.log(np.abs(np.diag(R)) + 1e-300)
    return np.sort(acc / n_steps)[::-1]


# --- engine sanity: the Ch 2 scan matches an independent oracle in the Ch 15 context ---


def test_scan_matches_independent_benettin() -> None:
    rng = np.random.default_rng(0)
    jacobians = rng.standard_normal((3, 4, 4)) * 0.5 + np.eye(4)
    got = ld.lyapunov_spectrum(jacobians, 80)
    np.testing.assert_allclose(got, _benettin_numpy(jacobians, 80), atol=1e-10)


def test_closed_form_log_growth_diagonal() -> None:
    J = np.diag([0.9, 0.5, 0.2, -0.8])
    np.testing.assert_allclose(ld.closed_form_log_growth(J), np.sort(np.log(np.abs(np.diag(J))))[::-1],
                               rtol=0, atol=1e-15)


# --- P2′ recovery on the three constructed systems (vs eigenvalue ground truth) ---


def test_dplr_recovery() -> None:
    a = ld._dplr_a()
    J = ld.dplr_jacobians(ld._DPLR_W, a, ld._DPLR_C, ld._RECOVER_STEPS)
    est = ld.lyapunov_spectrum(J, ld._RECOVER_STEPS)
    ref = ld.closed_form_log_growth(J[0])
    # top exponent converges fastest (largest spectral gap): pinned < 1e-4 (measured 2.3e-6)
    assert abs(est[0] - ref[0]) < 1e-4
    # full spectrum to the time-average O(1/T) tolerance (measured 2.1e-4)
    assert np.max(np.abs(est - ref)) < 1e-3


def test_ltv_recovery_exact() -> None:
    A, delta, B = ld._ltv_system()
    Abar = np.asarray(discretize_selective(jnp.asarray(A), jnp.asarray(delta), jnp.asarray(B))[0])
    J = ld.constructed_ltv_jacobians(A, delta, B)
    est = ld.lyapunov_spectrum(J, ld._LTV_LEN)
    ref = ld.ltv_closed_form_log_growth(Abar)
    # diagonal system: QR is trivial, recovery is exact
    np.testing.assert_allclose(est, ref, rtol=0, atol=1e-12)


def test_s4d_degenerate_but_decoupled_recovered_exactly() -> None:
    """S4D-Lin init: equal moduli, but decoupled (normal) -> recovered exactly (no scatter)."""
    J = ld.s4d_lin_transition(ld._S4D_MODES, ld._S4D_DT)[np.newaxis, ...]
    est = ld.lyapunov_spectrum(J, ld._RECOVER_STEPS)
    ref = ld.closed_form_log_growth(J[0])
    assert np.max(np.abs(est - ref)) < 1e-10
    # every mode decays at the same rate -dt/2
    np.testing.assert_allclose(ref, np.full_like(ref, -0.5 * ld._S4D_DT), rtol=0, atol=1e-12)


def test_divergence_identity_holds_even_when_degenerate() -> None:
    """Sum of exponents = <log|det J|> exactly, regardless of degeneracy (the robust summary)."""
    a = ld._dplr_a()
    J_dplr = ld.dplr_jacobians(ld._DPLR_W, a, ld._DPLR_C, ld._RECOVER_STEPS)
    J_s4d = ld.s4d_lin_transition(ld._S4D_MODES, ld._S4D_DT)[np.newaxis, ...]
    for J in (J_dplr, J_s4d):
        est = ld.lyapunov_spectrum(J, ld._RECOVER_STEPS)
        assert abs(est.sum() - ld.log_det_rate(J)) < 1e-10


def test_log_det_rate_autonomous() -> None:
    J = np.diag([0.9, 0.4, 0.2])
    # <log|det J|> for an autonomous tile == sum log|eigvals|
    np.testing.assert_allclose(ld.log_det_rate(J[np.newaxis, ...]),
                               float(np.sum(np.log(np.abs(np.diag(J))))), rtol=0, atol=1e-13)


def test_resolution_limit_on_non_normal_ring() -> None:
    """The caveat: distinct modes sharing a modulus in a non-normal recurrence blur.

    The individual exponents scatter by O(1e-2) (modes unresolved) while the mean is
    pinned exactly by the divergence identity.
    """
    J = ld.ring_jacobian(ld._RING_N, ld._RING_C, ld._RING_DT)[np.newaxis, ...]
    est = ld.lyapunov_spectrum(J, ld._RING_STEPS) / ld._RING_DT
    ref = np.sort(np.linalg.eigvals(np.asarray(
        build_ring_state_matrix(n=ld._RING_N, c=ld._RING_C, kappa=1.0))).real)[::-1]
    scatter = float(np.max(np.abs(est - ref)))
    assert 1e-3 < scatter < 1e-2, f"expected O(1e-2) scatter, got {scatter:.3e}"
    assert abs(est.mean() - ref.mean()) < 1e-6, "the mean rate must stay exact"


# --- P3′ regime separation + effective state size ---


def test_two_route_marginal_count_agrees() -> None:
    """Algebraic (eigvalsh magnitudes) and dynamical (QR exponents) routes agree on r."""
    A = ld.two_regime_transition(ld._REGIME_R, ld._REGIME_D, ld._REGIME_W, seed=ld._REGIME_SEED)
    spec = ld.lyapunov_spectrum(A[np.newaxis, ...], ld._REGIME_STEPS)
    mags = np.abs(np.asarray(jnp.linalg.eigvalsh(jnp.asarray(A))))
    count_alg = ld.marginal_mode_count(mags, ld._REGIME_TOL, mode="magnitude")
    count_dyn = ld.marginal_mode_count(spec, ld._REGIME_TOL, mode="exponent")
    assert count_alg == count_dyn == ld._REGIME_R


def test_effective_state_size_matches_closed_form() -> None:
    """Measured participation ratio (from eigvalsh) == the two-level closed form, machine precision."""
    for w in (0.05, 0.4, 0.95):
        A = ld.two_regime_transition(ld._REGIME_R, ld._REGIME_D, w, seed=ld._REGIME_SEED)
        mags = np.abs(np.asarray(jnp.linalg.eigvalsh(jnp.asarray(A))))
        measured = ld.effective_state_size(mags)
        closed = ld.effective_state_size_closed_form(ld._REGIME_R, ld._REGIME_D, w)
        np.testing.assert_allclose(measured, closed, rtol=0, atol=1e-12)


def test_effective_state_size_figure_values() -> None:
    """Pin the caption numbers: D_eff(w=0.05)=3.025, D_eff(w=0.95)=7.980 (r=3, d=8)."""
    assert abs(ld.effective_state_size_closed_form(3, 8, 0.05) - 3.0250) < 5e-4
    assert abs(ld.effective_state_size_closed_form(3, 8, 0.95) - 7.9798) < 5e-4


def test_effective_state_size_limits() -> None:
    # all modes equal -> D_eff = d; only r dominant (w=0) -> D_eff = r; w=1 -> d
    np.testing.assert_allclose(ld.effective_state_size(np.ones(8)), 8.0, rtol=0, atol=1e-12)
    assert ld.effective_state_size_closed_form(3, 8, 0.0) == 3.0
    assert ld.effective_state_size_closed_form(3, 8, 1.0) == 8.0


def test_marginal_mode_count_modes() -> None:
    mags = np.array([1.0, 0.99, 0.5, 0.4])
    assert ld.marginal_mode_count(mags, 0.05, mode="magnitude") == 2
    exps = np.array([0.0, -0.01, -0.7, -0.9])
    assert ld.marginal_mode_count(exps, 0.05, mode="exponent") == 2


# --- validation ---


def test_validation_raises() -> None:
    with pytest.raises(ValueError):
        ld.closed_form_log_growth(np.zeros((3, 4)))  # not square
    with pytest.raises(ValueError):
        ld.log_det_rate(np.zeros((4, 4)))  # not 3-D
    with pytest.raises(ValueError):
        ld.s4d_lin_transition(0, 0.4)
    with pytest.raises(ValueError):
        ld.s4d_lin_transition(4, 0.0)
    with pytest.raises(ValueError):
        ld.two_regime_transition(8, 8, 0.4)  # need r < d
    with pytest.raises(ValueError):
        ld.two_regime_transition(3, 8, 1.5)  # w out of range
    with pytest.raises(ValueError):
        ld.effective_state_size(np.zeros(4))  # all zero
    with pytest.raises(ValueError):
        ld.effective_state_size_closed_form(9, 8, 0.4)  # r > d
    with pytest.raises(ValueError):
        ld.marginal_mode_count(np.ones(3), 0.1, mode="bogus")
    with pytest.raises(ValueError):
        ld.marginal_mode_count(np.ones(3), -0.1, mode="magnitude")
