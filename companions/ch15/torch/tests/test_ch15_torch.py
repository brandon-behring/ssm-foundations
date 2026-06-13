r"""Cross-framework parity: torch Lyapunov diagnostics vs JAX.

Two layers (mirroring the ch13/ch16 torch suites):

* **standalone torch assertions** — diagonal recovery, the divergence identity, the
  effective-state-size closed form — meaningful without JAX;
* **cross-framework parity** — recompute the JAX diagnostics in-process on the same
  numpy-drawn inputs and pin the torch outputs to them (``< 1e-9``, both float64).
  Skipped if JAX is unavailable. This is the guarantee the instrument is
  framework-agnostic (pilot B runs it on trained torch models).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float64)

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch15.jax import lyapunov_diagnostics as jld  # noqa: E402
from companions.ch15.torch import lyapunov_diagnostics as tld  # noqa: E402

_ATOL = 1e-9


def _t(x):
    return torch.tensor(np.asarray(x, dtype=np.float64))


# --- standalone torch assertions -------------------------------------------


def test_diagonal_recovery_torch() -> None:
    J = torch.diag(_t([0.9, 0.7, 0.5, 0.3]))
    spec = tld.qr_lyapunov(J.unsqueeze(0), 2000)
    ref = torch.sort(torch.log(_t([0.9, 0.7, 0.5, 0.3])), descending=True).values
    assert torch.allclose(spec, ref, rtol=0, atol=1e-12)


def test_divergence_identity_torch() -> None:
    rng = np.random.default_rng(0)
    J = _t(rng.standard_normal((5, 5)) * 0.3 + np.eye(5))
    spec = tld.qr_lyapunov(J.unsqueeze(0), 3000)
    assert abs(float(spec.sum()) - tld.log_det_rate(J.unsqueeze(0))) < 1e-9


def test_effective_state_size_two_level_torch() -> None:
    mags = _t(np.concatenate([np.ones(3), np.full(5, 0.4)]))
    closed = (3 + 5 * 0.4**2) ** 2 / (3 + 5 * 0.4**4)
    assert abs(tld.effective_state_size(mags) - closed) < 1e-12


def test_marginal_mode_count_torch() -> None:
    assert tld.marginal_mode_count(_t([1.0, 0.99, 0.5, 0.4]), 0.05, "magnitude") == 2
    assert tld.marginal_mode_count(_t([0.0, -0.01, -0.7]), 0.05, "exponent") == 2


def test_validation_torch() -> None:
    with pytest.raises(ValueError):
        tld.qr_lyapunov(_t(np.zeros((4, 4))), 10)  # not 3-D
    with pytest.raises(ValueError):
        tld.effective_state_size(_t(np.zeros(4)))
    with pytest.raises(ValueError):
        tld.marginal_mode_count(_t(np.ones(3)), 0.1, "bogus")


# --- cross-framework parity (torch vs JAX) ---------------------------------


@pytest.mark.parametrize("seed", [0, 3])
def test_qr_lyapunov_parity(seed: int) -> None:
    rng = np.random.default_rng(seed)
    jac = rng.standard_normal((3, 4, 4)) * 0.5 + np.eye(4)
    spec_j = jld.lyapunov_spectrum(jnp.asarray(jac), 120)
    spec_t = tld.qr_lyapunov(_t(jac), 120)
    assert np.max(np.abs(np.asarray(spec_j) - spec_t.numpy())) < _ATOL


def test_closed_form_parity() -> None:
    rng = np.random.default_rng(1)
    J = rng.standard_normal((6, 6)) * 0.4 + np.eye(6)
    spec_j = jld.closed_form_log_growth(np.asarray(J))
    spec_t = tld.closed_form_log_growth(_t(J))
    assert np.max(np.abs(spec_j - spec_t.numpy())) < _ATOL


def test_dplr_spectrum_parity() -> None:
    """The DPLR Lyapunov estimate agrees across frameworks (same QR algorithm)."""
    a = jld._dplr_a()
    J = jld.dplr_jacobians(jld._DPLR_W, a, jld._DPLR_C, 1)[0]  # the (N, N) transition
    spec_j = jld.lyapunov_spectrum(J[np.newaxis, ...], 800)
    spec_t = tld.qr_lyapunov(_t(J).unsqueeze(0), 800)
    assert np.max(np.abs(np.asarray(spec_j) - spec_t.numpy())) < _ATOL


def test_effective_state_size_parity() -> None:
    rng = np.random.default_rng(2)
    mags = np.abs(rng.standard_normal(10))
    assert abs(jld.effective_state_size(mags) - tld.effective_state_size(_t(mags))) < _ATOL


def test_two_regime_spectrum_parity() -> None:
    """torch recovers the same regime spectrum as JAX on the constructed two-block system."""
    A = jld.two_regime_transition(3, 8, 0.4, seed=0)
    spec_j = jld.lyapunov_spectrum(A[np.newaxis, ...], 800)
    spec_t = tld.qr_lyapunov(_t(A).unsqueeze(0), 800)
    assert np.max(np.abs(np.asarray(spec_j) - spec_t.numpy())) < _ATOL
