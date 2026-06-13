r"""Tests for ch13 xlstm: the mLSTM matrix memory and its exponential-gate stabilizer.

Pins the §13.4 claims:

* **P2 (stabilizer exactness).** Wherever the naive recurrence does not overflow,
  the stabilized readout equals it to ``< 1e-12`` — even at intermediate magnitudes
  near $10^{303}$ (it is a change of variables, not an approximation).
* **Gate boundedness.** The rescaled gates $f'_t, i'_t$ lie in $(0, 1]$ by
  construction of $m_t$ as a running max.
* **Overflow.** The naive recurrence genuinely overflows float64 once the
  exponential input gate exceeds $\sim e^{709}$ (non-finite readout entries); the
  stabilized recurrence stays finite and exact across the cliff.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from companions.ch13.jax.xlstm import (  # noqa: E402
    log_sigmoid,
    make_gate_stream,
    mlstm_naive,
    mlstm_stabilized,
    naive_finite_fraction,
    peak_state_magnitude,
    readout_max_abs_diff,
    single_pair_recovery,
)


def _safe_stream(length=24, d_k=6, d_v=5, seed=0):
    """Moderate gates: naive stays finite, so P2 can be checked directly."""
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((length, d_k)))
    k_raw = rng.standard_normal((length, d_k))
    k = jnp.asarray(k_raw / np.linalg.norm(k_raw, axis=1, keepdims=True))
    v = jnp.asarray(rng.standard_normal((length, d_v)))
    log_f = log_sigmoid(jnp.asarray(rng.uniform(0.0, 2.0, size=length)))
    log_i = jnp.asarray(rng.uniform(-2.0, 2.0, size=length))
    return q, k, v, log_f, log_i


# --- §13.4 P2: stabilizer exactness in the safe regime ----------------------


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_p2_exactness_safe_regime(seed):
    q, k, v, log_f, log_i = _safe_stream(seed=seed)
    h_naive = mlstm_naive(q, k, v, log_f, log_i)
    h_stab, _ = mlstm_stabilized(q, k, v, log_f, log_i)
    assert bool(jnp.all(jnp.isfinite(h_naive)))
    assert_allclose(np.asarray(h_naive), np.asarray(h_stab), rtol=0, atol=1e-12)


def test_p2_holds_at_large_but_finite_magnitude():
    """At peak log-gate 700 the intermediates reach ~1e303 yet the readout agrees."""
    assert readout_max_abs_diff(700.0) < 1e-12


# --- §13.4 gate boundedness -------------------------------------------------


@pytest.mark.parametrize("seed", [0, 3])
def test_rescaled_gates_in_unit_interval(seed):
    q, k, v, log_f, log_i = _safe_stream(seed=seed)
    _, m_traj = mlstm_stabilized(q, k, v, log_f, log_i)
    m_prev = jnp.concatenate([jnp.array([-jnp.inf]), m_traj[:-1]])
    # i'_t = exp(log_i - m_t) <= 1  <=>  log_i <= m_t
    assert bool(jnp.all(log_i <= m_traj + 1e-12))
    # f'_t = exp(log_f + m_{t-1} - m_t) <= 1  <=>  log_f + m_{t-1} <= m_t
    assert bool(jnp.all(log_f + m_prev <= m_traj + 1e-12))


def test_first_step_with_minus_inf_init_is_finite():
    """m_0 = -inf must not produce nan on the first step (exp(-inf) = 0)."""
    q, k, v, log_f, log_i = _safe_stream(length=1, seed=2)
    h_stab, m_traj = mlstm_stabilized(q, k, v, log_f, log_i)
    assert bool(jnp.all(jnp.isfinite(h_stab)))
    assert_allclose(float(m_traj[0]), float(log_i[0]), rtol=0, atol=1e-12)  # m_1 = log_i_1


# --- §13.4 overflow: naive dies, stabilized survives ------------------------


def test_naive_overflows_stabilized_finite():
    q, k, v, log_f, log_i = make_gate_stream(16, 4, 3, peak_log_i=760.0)
    h_naive = mlstm_naive(q, k, v, log_f, log_i)
    h_stab, _ = mlstm_stabilized(q, k, v, log_f, log_i)
    assert not bool(jnp.all(jnp.isfinite(h_naive)))  # naive has nan/inf
    assert bool(jnp.all(jnp.isfinite(h_stab)))  # stabilized clean


def test_overflow_cliff_location():
    """exp(705)*||outer|| is finite (~1e307); exp(710) overflows float64."""
    assert np.isfinite(peak_state_magnitude(705.0, stabilized=False))
    assert not np.isfinite(peak_state_magnitude(710.0, stabilized=False))


def test_stabilized_peak_bounded_across_cliff():
    """Stabilized peak memory entry is the same O(1) value on both sides of the cliff."""
    below = peak_state_magnitude(700.0, stabilized=True)
    above = peak_state_magnitude(760.0, stabilized=True)
    assert below < 100.0 and above < 100.0
    assert_allclose(below, above, rtol=0, atol=1e-9)  # the overflow timestep is identical


# --- figure caption pins ----------------------------------------------------


def test_overflow_figure_caption_pins():
    """The exact numbers the stabilizer-overflow figure caption quotes (seed 0)."""
    assert_allclose(peak_state_magnitude(700.0, stabilized=False), 5.941e303, rtol=1e-3, atol=0)
    assert_allclose(peak_state_magnitude(700.0, stabilized=True), 2.540, rtol=0, atol=1e-3)
    assert_allclose(readout_max_abs_diff(700.0), 8.88e-16, rtol=0, atol=5e-16)
    assert_allclose(naive_finite_fraction(760.0), 0.5, rtol=0, atol=1e-12)
    assert_allclose(naive_finite_fraction(2.0), 1.0, rtol=0, atol=1e-12)  # safe: all finite


def test_single_pair_recovery_exact():
    """The stabilizer recovers the stored value exactly at any input gate, incl. overflow."""
    for log_i in (0.0, 50.0, 800.0):
        err, _ = single_pair_recovery(log_i)
        assert err < 1e-12
    _, readout = single_pair_recovery(800.0)
    assert_allclose(np.asarray(readout), [0.3, -0.7, 1.1], rtol=0, atol=1e-12)  # = _REF_V


# --- log_sigmoid + validation ----------------------------------------------


def test_log_sigmoid_matches_definition():
    x = jnp.linspace(-10.0, 10.0, 21)
    expected = jnp.log(jax.nn.sigmoid(x))
    assert_allclose(np.asarray(log_sigmoid(x)), np.asarray(expected), rtol=0, atol=1e-12)
    assert bool(jnp.all(log_sigmoid(x) <= 0.0))  # log of a probability


def test_mlstm_rejects_shape_mismatch():
    q, k, v, log_f, log_i = _safe_stream(seed=0)
    with pytest.raises(ValueError):
        mlstm_naive(q, k, v, log_f[:-1], log_i)
    with pytest.raises(ValueError):
        mlstm_stabilized(q, k, v[:-1], log_f, log_i)
