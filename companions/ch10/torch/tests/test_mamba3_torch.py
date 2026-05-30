r"""Cross-framework parity tests: torch Mamba-3 discretization/complex-state vs JAX.

Two layers of checking, mirroring the ch09 torch suite:

* **standalone torch assertions** — order slopes, homogeneous-blindness, stability,
  RoPE<->complex equivalence — so the suite is meaningful even without JAX present;
* **cross-framework parity** — recompute the JAX companion in-process and pin the
  torch outputs to it (``< 1e-9``, both float64). Skipped if JAX is unavailable.
"""

from __future__ import annotations

import math

import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from companions.ch10.torch.complex_state import (
    complex_scalar_recurrence,
    decay_rate,
    rope_equivalence_residual,
    rope_matrix,
)
from companions.ch10.torch.discretization import (
    amplification,
    discretize_exp_trapezoidal,
    discretize_zoh,
    global_error,
    order_sweep,
)

_A = -0.5 + 2.0j
_OMEGA = 1.3
_T = 6.0
_X0 = 1.0 + 0.0j
_DTS = [0.2, 0.1, 0.05, 0.025, 0.0125]
_RHO = 0.95
_THETA = math.pi / 9.0


# --- standalone torch checks -----------------------------------------------


def test_order_slopes_forced():
    _, s_zoh = order_sweep("zoh", _A, _DTS, _T, _OMEGA, x0=_X0)
    _, s_trap = order_sweep("exp_trapezoidal", _A, _DTS, _T, _OMEGA, x0=_X0)
    _, s_bl = order_sweep("bilinear", _A, _DTS, _T, _OMEGA, x0=_X0)
    assert 0.9 < s_zoh < 1.15
    assert 1.9 < s_trap < 2.1
    assert 1.9 < s_bl < 2.1


def test_homogeneous_blindness():
    e_zoh = global_error("zoh", _A, 0.1, _T, None, x0=_X0)
    e_trap = global_error("exp_trapezoidal", _A, 0.1, _T, None, x0=_X0)
    assert e_zoh < 1e-12
    assert e_trap < 1e-12


def test_transition_identical_zoh_exptrap():
    a_zoh, _ = discretize_zoh(_A, 0.1)
    a_trap, _, _ = discretize_exp_trapezoidal(_A, 0.1)
    assert torch.allclose(a_zoh, a_trap, rtol=0, atol=0)


def test_stiff_mode_contrast():
    z = torch.tensor(-50.0, dtype=torch.complex128)
    assert float(torch.abs(amplification("exp_trapezoidal", z))) < 1e-12
    assert float(torch.abs(amplification("bilinear", z))) > 0.9
    assert float(torch.abs(amplification("forward_euler", z))) > 1.0


def test_rope_equivalence():
    rng = np.random.default_rng(1)
    drive = torch.tensor(rng.standard_normal(50) + 1j * rng.standard_normal(50))
    assert rope_equivalence_residual(_RHO, _THETA, drive) < 1e-12


def test_rope_matrix_matches_complex_multiply():
    z = 0.7 - 0.4j
    R = rope_matrix(_THETA)
    got = R @ torch.stack([torch.tensor(z).real, torch.tensor(z).imag])
    want = complex(math.cos(_THETA), math.sin(_THETA)) * z
    assert abs(float(got[0]) - want.real) < 1e-14
    assert abs(float(got[1]) - want.imag) < 1e-14


def test_spiral_decay_rate():
    xs = complex_scalar_recurrence(_RHO, _THETA, 80)
    assert abs(decay_rate(xs) - math.log(_RHO)) < 1e-12


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        discretize_exp_trapezoidal(_A, 0.1, lam=1.5)
    with pytest.raises(ValueError):
        amplification("midpoint", torch.tensor(-1.0))
    with pytest.raises(ValueError):
        complex_scalar_recurrence(1.01, _THETA, 5)


# --- cross-framework parity (recompute JAX in-process) ---------------------


def test_discretization_parity_against_jax():
    """torch order-sweep errors match the JAX companion to < 1e-9."""
    pytest.importorskip("jax")
    from companions.ch10.jax.discretization import order_sweep as jax_sweep

    for scheme in ("zoh", "exp_trapezoidal", "bilinear"):
        e_torch, s_torch = order_sweep(scheme, _A, _DTS, _T, _OMEGA, x0=_X0)
        e_jax, s_jax = jax_sweep(scheme, _A, __import__("jax.numpy", fromlist=["asarray"]).asarray(_DTS), _T, _OMEGA, x0=_X0)
        e_jax = np.asarray(e_jax)
        np.testing.assert_allclose(np.asarray(e_torch), e_jax, rtol=0, atol=1e-9)
        assert abs(s_torch - float(s_jax)) < 1e-9


def test_complex_recurrence_parity_against_jax():
    """torch complex trajectory matches the JAX companion to < 1e-9."""
    pytest.importorskip("jax")
    from companions.ch10.jax.complex_state import complex_scalar_recurrence as jax_rec

    xs_torch = complex_scalar_recurrence(_RHO, _THETA, 60).numpy()
    xs_jax = np.asarray(jax_rec(_RHO, _THETA, 60))
    np.testing.assert_allclose(xs_torch, xs_jax, rtol=0, atol=1e-9)
