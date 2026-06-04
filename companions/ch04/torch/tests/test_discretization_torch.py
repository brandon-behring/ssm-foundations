r"""Cross-framework parity: PyTorch Chapter 4 companions vs the JAX reference (§4.3-4.5).

Backfills JAX↔torch parity for Ch 4 (audit 0527-F27). The torch discretizers and simulators
are fed the *same* inputs as the JAX core and the outputs are compared in float64, so the two
frameworks stay in lock-step (the cross-framework-consistency goal Chapter 7 introduced). The
headline pedagogical claims of the existing JAX test (``test_discretization.py``) are mirrored
on the torch side: ZOH is first-order, bilinear is second-order, ZOH's $\bar A = e^{A\Delta}$
exactly, and the exp-trapezoidal scheme is second-order and beats ZOH.

Run: ``PYTHONPATH=. .venv/bin/pytest companions/ch04/torch -q``
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402, F401  (imported for the canonical parity-test preamble)

from companions.ch04.jax import discretization_comparison as jax_dc  # noqa: E402
from companions.ch04.jax import exp_trapezoidal as jax_et  # noqa: E402
from companions.ch04.torch import discretization_comparison as torch_dc  # noqa: E402
from companions.ch04.torch import exp_trapezoidal as torch_et  # noqa: E402

_PARITY_TOL = 1e-9
_DTS = np.array([0.1, 0.05, 0.025])


def _slope(errs: np.ndarray, dts: np.ndarray) -> float:
    """Empirical convergence order from the two finest step sizes."""
    return float(np.log(errs[-2] / errs[-1]) / np.log(dts[-2] / dts[-1]))


# ---------------------------------------------------------------------------
# §4.3-4.4 — discretizer parity (torch matrices == JAX matrices on identical inputs)
# ---------------------------------------------------------------------------


def test_discretize_forward_euler_matches_jax() -> None:
    """torch forward-Euler $(\\bar A, \\bar B)$ reproduce the JAX matrices."""
    dt = 0.1
    Ad_j, Bd_j = jax_dc.discretize_forward_euler(jax_dc._A_MAT, jax_dc._B_VEC, dt)
    Ad_t, Bd_t = torch_dc.discretize_forward_euler(torch_dc._A_MAT, torch_dc._B_VEC, dt)
    assert np.max(np.abs(np.asarray(Ad_j) - Ad_t.numpy())) < _PARITY_TOL
    assert np.max(np.abs(np.asarray(Bd_j) - Bd_t.numpy())) < _PARITY_TOL


def test_discretize_zoh_matches_jax() -> None:
    """torch ZOH $(\\bar A, \\bar B)$ via augmented matrix-exp reproduce the JAX matrices."""
    dt = 0.1
    Ad_j, Bd_j = jax_dc.discretize_zoh(jax_dc._A_MAT, jax_dc._B_VEC, dt)
    Ad_t, Bd_t = torch_dc.discretize_zoh(torch_dc._A_MAT, torch_dc._B_VEC, dt)
    assert np.max(np.abs(np.asarray(Ad_j) - Ad_t.numpy())) < _PARITY_TOL
    assert np.max(np.abs(np.asarray(Bd_j) - Bd_t.numpy())) < _PARITY_TOL


def test_discretize_bilinear_matches_jax() -> None:
    """torch bilinear $(\\bar A, \\bar B)$ via linear solve reproduce the JAX matrices."""
    dt = 0.1
    Ad_j, Bd_j = jax_dc.discretize_bilinear(jax_dc._A_MAT, jax_dc._B_VEC, dt)
    Ad_t, Bd_t = torch_dc.discretize_bilinear(torch_dc._A_MAT, torch_dc._B_VEC, dt)
    assert np.max(np.abs(np.asarray(Ad_j) - Ad_t.numpy())) < _PARITY_TOL
    assert np.max(np.abs(np.asarray(Bd_j) - Bd_t.numpy())) < _PARITY_TOL


# ---------------------------------------------------------------------------
# §4.5 — exp-trapezoidal discretization parity (the augmented φ-function blocks)
# ---------------------------------------------------------------------------


def test_discretize_exp_trap_matches_jax() -> None:
    """torch exp-trap $(\\bar A, B_0, B_1)$ reproduce the JAX augmented-matrix blocks."""
    dt = 0.1
    Ad_j, B0_j, B1_j = jax_et.discretize_exp_trap(jax_et._A_MAT, jax_et._B_VEC, dt)
    Ad_t, B0_t, B1_t = torch_et.discretize_exp_trap(torch_et._A_MAT, torch_et._B_VEC, dt)
    assert np.max(np.abs(np.asarray(Ad_j) - Ad_t.numpy())) < _PARITY_TOL
    assert np.max(np.abs(np.asarray(B0_j) - B0_t.numpy())) < _PARITY_TOL
    assert np.max(np.abs(np.asarray(B1_j) - B1_t.numpy())) < _PARITY_TOL


# ---------------------------------------------------------------------------
# Simulation parity (full trajectories agree on identical inputs)
# ---------------------------------------------------------------------------


def test_simulate_zoh_matches_jax() -> None:
    """torch ZOH trajectory reproduces the JAX ``simulate`` output (§4.3 parity)."""
    dt, t_end = 0.05, 4.0
    ts_j, ys_j = jax_dc.simulate(jax_dc.discretize_zoh, jax_dc.step_hold, dt, t_end)
    ts_t, ys_t = torch_dc.simulate(torch_dc.discretize_zoh, torch_dc.step_hold, dt, t_end)
    assert np.max(np.abs(ts_j - ts_t)) < _PARITY_TOL
    assert np.max(np.abs(ys_j - ys_t)) < _PARITY_TOL


def test_simulate_bilinear_matches_jax() -> None:
    """torch bilinear trajectory reproduces the JAX ``simulate`` output (§4.4 parity)."""
    dt, t_end = 0.05, 4.0
    ts_j, ys_j = jax_dc.simulate(jax_dc.discretize_bilinear, jax_dc.step_midpoint, dt, t_end)
    ts_t, ys_t = torch_dc.simulate(torch_dc.discretize_bilinear, torch_dc.step_midpoint, dt, t_end)
    assert np.max(np.abs(ts_j - ts_t)) < _PARITY_TOL
    assert np.max(np.abs(ys_j - ys_t)) < _PARITY_TOL


def test_simulate_exp_trap_matches_jax() -> None:
    """torch exp-trapezoidal trajectory reproduces the JAX ``simulate_exp_trap`` (§4.5 parity)."""
    dt, t_end = 0.05, 4.0
    ts_j, ys_j = jax_et.simulate_exp_trap(dt, t_end)
    ts_t, ys_t = torch_et.simulate_exp_trap(dt, t_end)
    assert np.max(np.abs(ts_j - ts_t)) < _PARITY_TOL
    assert np.max(np.abs(ys_j - ys_t)) < _PARITY_TOL


def test_measure_max_error_matches_jax() -> None:
    """torch ``measure_max_error`` reproduces the JAX scalar error (same Radau reference)."""
    for disc, step in (
        (torch_dc.discretize_zoh, torch_dc.step_hold),
        (torch_dc.discretize_bilinear, torch_dc.step_midpoint),
    ):
        # Map the torch callables to their JAX twins by name.
        jdisc = getattr(jax_dc, disc.__name__)
        jstep = getattr(jax_dc, step.__name__)
        e_j = jax_dc.measure_max_error(jdisc, jstep, 0.05)
        e_t = torch_dc.measure_max_error(disc, step, 0.05)
        assert abs(e_j - e_t) < _PARITY_TOL


# ---------------------------------------------------------------------------
# Within-framework exact identities (tighter than parity)
# ---------------------------------------------------------------------------


def test_torch_zoh_Ad_is_matrix_exponential() -> None:
    """torch ZOH's discrete dynamics matrix is exactly $e^{A\\Delta}$ (autonomous-exactness)."""
    dt = 0.1
    Ad, _ = torch_dc.discretize_zoh(torch_dc._A_MAT, torch_dc._B_VEC, dt)
    expected = torch.linalg.matrix_exp(torch_dc._A_MAT * dt)
    assert float(torch.max(torch.abs(Ad - expected))) < 1e-12


def test_torch_simulate_matches_naive_loop() -> None:
    """torch ``simulate`` matches an independent NumPy loop (the eager-scan port is correct)."""
    dt, t_end = 0.05, 2.0
    ts, ys = torch_dc.simulate(torch_dc.discretize_zoh, torch_dc.step_hold, dt, t_end)
    Ad, Bd = torch_dc.discretize_zoh(torch_dc._A_MAT, torch_dc._B_VEC, dt)
    Ad, Bd, C = Ad.numpy(), Bd.numpy(), torch_dc._C_ROW.numpy()
    h = np.zeros(2)
    ref = np.zeros(len(ts))
    u = np.sin(2.0 * ts)
    for k in range(len(ts) - 1):
        ref[k] = C @ h
        h = Ad @ h + Bd * u[k]
    ref[-1] = C @ h
    assert np.max(np.abs(ys - ref)) < 1e-12


# ---------------------------------------------------------------------------
# Headline pedagogical claims, mirrored from the JAX test on the torch side
# ---------------------------------------------------------------------------


def _torch_et_max_err(sim_fn, dt: float, t_end: float = 4.0) -> float:
    """Max output error of a torch exp_trapezoidal simulator vs the Radau reference."""
    ts, ys = sim_fn(dt, t_end)
    return float(np.max(np.abs(ys - torch_et.continuous_reference(t_end, ts))))


def test_torch_zoh_first_order() -> None:
    """Mirror of JAX ``test_zoh_first_order``: ZOH slope ≈ 1 on the torch side."""
    errs = np.array(
        [
            torch_dc.measure_max_error(torch_dc.discretize_zoh, torch_dc.step_hold, float(dt))
            for dt in _DTS
        ]
    )
    s = _slope(errs, _DTS)
    assert 0.8 <= s <= 1.3, f"ZOH slope {s:.3f} not ≈ 1"


def test_torch_bilinear_second_order() -> None:
    """Mirror of JAX ``test_bilinear_second_order``: bilinear slope ≈ 2 on the torch side."""
    errs = np.array(
        [
            torch_dc.measure_max_error(torch_dc.discretize_bilinear, torch_dc.step_midpoint, float(dt))
            for dt in _DTS
        ]
    )
    s = _slope(errs, _DTS)
    assert 1.7 <= s <= 2.3, f"bilinear slope {s:.3f} not ≈ 2"


def test_torch_exp_trap_second_order() -> None:
    """Mirror of JAX ``test_exp_trap_second_order`` (F29 regression guard): slope ≥ 1.6."""
    errs = np.array([_torch_et_max_err(torch_et.simulate_exp_trap, float(dt)) for dt in _DTS])
    s = _slope(errs, _DTS)
    assert s >= 1.6, f"exp-trap slope {s:.3f} — degraded below 2nd order?"


def test_torch_exp_trap_beats_zoh() -> None:
    """Mirror of JAX ``test_exp_trap_beats_zoh``: exp-trap beats ZOH at Δ=0.05 on torch."""
    err_exp = _torch_et_max_err(torch_et.simulate_exp_trap, 0.05)
    err_zoh = _torch_et_max_err(torch_et.simulate_zoh, 0.05)
    assert err_exp < err_zoh, f"exp-trap err {err_exp:.2e} should beat ZOH err {err_zoh:.2e}"


def test_torch_slopes_match_jax_slopes() -> None:
    """The torch empirical slopes equal the JAX slopes (the convergence story is framework-free)."""
    pairs = (("discretize_zoh", "step_hold"), ("discretize_bilinear", "step_midpoint"))
    for disc_name, step_name in pairs:
        j_errs = np.array(
            [
                jax_dc.measure_max_error(
                    getattr(jax_dc, disc_name), getattr(jax_dc, step_name), float(dt)
                )
                for dt in _DTS
            ]
        )
        t_errs = np.array(
            [
                torch_dc.measure_max_error(
                    getattr(torch_dc, disc_name), getattr(torch_dc, step_name), float(dt)
                )
                for dt in _DTS
            ]
        )
        assert abs(_slope(j_errs, _DTS) - _slope(t_errs, _DTS)) < 1e-6
