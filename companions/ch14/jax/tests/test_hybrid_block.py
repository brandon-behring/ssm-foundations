r"""Tests for ch14 hybrid_block: composition patterns and decode-cost accounting.

Pins the §§14.3–14.4 claims:

* the band-mask windowed attention == per-position loop oracle to ``< 1e-12``;
* window $w \ge L$ **is** full causal attention (exact);
* the gated-decay EMA matches its closed form $h_t = (1 - g^t)\bar{x}$ under
  constant input, and ``lax.scan`` == Python-loop oracle;
* parallel-gate reductions $g = 1 \to$ attention branch, $g = 0 \to$ SSM
  branch are *exact* (bitwise), per channel for vector gates — the
  executable content of the granularity-ordering proposition;
* the $r\!:\!1$ interleave schedule has the promised pattern/counts, and the
  residual stack reduces to pure-attention / pure-SSM stacks;
* the decode-state formulas equal the summed sizes of *materialised*
  buffers (audited against real array shapes, not re-derived arithmetic).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from companions.ch14.jax.hybrid_block import (  # noqa: E402
    decode_buffers,
    decode_state_floats,
    full_attention_decode_floats,
    full_causal_attention,
    gated_decay_ssm,
    gated_decay_ssm_naive,
    interleave_hybrid,
    interleave_schedule,
    parallel_gated_hybrid,
    sequential_hybrid,
    sliding_window_attention,
    sliding_window_attention_naive,
)


def _x(length=48, d=8, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((length, d)))


# ---------------------------------------------------------------------------
# Windowed attention.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("window", [1, 3, 16, 48, 64])
def test_band_mask_equals_loop_oracle(seed, window):
    x = _x(seed=seed)
    y_band = sliding_window_attention(x, window)
    y_loop = sliding_window_attention_naive(x, window)
    assert_allclose(np.asarray(y_band), np.asarray(y_loop), rtol=0, atol=1e-12)


@pytest.mark.parametrize("window", [48, 49, 100])
def test_window_geq_length_is_full_attention(window):
    """w >= L: the band mask covers the whole lower triangle — full attention."""
    x = _x(length=48)
    y_win = sliding_window_attention(x, window)
    y_full = full_causal_attention(x)
    assert_allclose(np.asarray(y_win), np.asarray(y_full), rtol=0, atol=1e-12)


def test_first_position_attends_only_to_itself():
    """Position 1 has a one-element window regardless of w: output == input row."""
    x = _x()
    for w in (1, 8):
        y = sliding_window_attention(x, w)
        assert_allclose(np.asarray(y[0]), np.asarray(x[0]), rtol=0, atol=1e-15)


# ---------------------------------------------------------------------------
# Gated-decay EMA.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 3])
def test_ssm_scan_equals_naive_oracle(seed):
    rng = np.random.default_rng(seed)
    x = _x(seed=seed)
    gates = jnp.asarray(rng.uniform(0.0, 1.0, size=x.shape))
    h_scan = gated_decay_ssm(x, gates)
    h_naive = gated_decay_ssm_naive(x, gates)
    assert_allclose(np.asarray(h_scan), np.asarray(h_naive), rtol=0, atol=1e-12)


def test_ema_closed_form_under_constant_input():
    """Constant input + constant gate g: h_t = (1 - g^t) xbar exactly."""
    rng = np.random.default_rng(5)
    d, length = 6, 64
    xbar = rng.standard_normal(d)
    const = jnp.broadcast_to(jnp.asarray(xbar), (length, d))
    for g in (0.5, 0.9, 0.99):
        h = gated_decay_ssm(const, jnp.full(d, g))
        t = np.arange(1, length + 1)[:, None]
        analytic = (1.0 - g**t) * xbar
        assert_allclose(np.asarray(h), analytic, rtol=0, atol=1e-12)


def test_ema_gate_edge_cases():
    """g = 0 passes the input through; g = 1 never writes (state stays 0)."""
    x = _x(length=16, d=4, seed=2)
    h_open = gated_decay_ssm(x, jnp.zeros(4))
    assert_allclose(np.asarray(h_open), np.asarray(x), rtol=0, atol=0)
    h_closed = gated_decay_ssm(x, jnp.ones(4))
    assert_allclose(np.asarray(h_closed), np.zeros(x.shape), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Compositions.
# ---------------------------------------------------------------------------


def test_parallel_gate_scalar_reductions_exact():
    """g = 1 -> attention branch, g = 0 -> SSM branch, bitwise."""
    x = _x()
    gates = jnp.asarray(np.random.default_rng(1).uniform(0.6, 0.95, size=8))
    w = 16
    y1 = parallel_gated_hybrid(x, 1.0, gates, w)
    y0 = parallel_gated_hybrid(x, 0.0, gates, w)
    assert_allclose(np.asarray(y1), np.asarray(sliding_window_attention(x, w)), rtol=0, atol=0)
    assert_allclose(np.asarray(y0), np.asarray(gated_decay_ssm(x, gates)), rtol=0, atol=0)


def test_parallel_vector_gate_selects_per_channel():
    """A 0/1 vector gate routes each channel to its branch exactly."""
    x = _x(d=6)
    gates = jnp.asarray(np.random.default_rng(2).uniform(0.6, 0.95, size=6))
    w = 8
    g_vec = jnp.asarray([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    y = parallel_gated_hybrid(x, g_vec, gates, w)
    attn = sliding_window_attention(x, w)
    ssm = gated_decay_ssm(x, gates)
    expected = np.where(np.asarray(g_vec)[None, :] == 1.0, np.asarray(attn), np.asarray(ssm))
    assert_allclose(np.asarray(y), expected, rtol=0, atol=0)


def test_sequential_orders_do_not_commute():
    """attn(ssm(x)) != ssm(attn(x)) on generic input — composition order matters."""
    x = _x(seed=4)
    gates = jnp.asarray(np.random.default_rng(4).uniform(0.6, 0.95, size=8))
    y_sa = sequential_hybrid(x, gates, 16, order=("ssm", "attn"))
    y_as = sequential_hybrid(x, gates, 16, order=("attn", "ssm"))
    assert float(jnp.max(jnp.abs(y_sa - y_as))) > 1e-3


def test_interleave_schedule_pattern_and_counts():
    assert interleave_schedule(8, 3) == (
        "ssm", "ssm", "ssm", "attn", "ssm", "ssm", "ssm", "attn",
    )
    assert interleave_schedule(9, 2) == (
        "ssm", "ssm", "attn", "ssm", "ssm", "attn", "ssm", "ssm", "attn",
    )
    assert interleave_schedule(4, 0) == ("attn",) * 4
    # ratio >= n_blocks: the first attention block never arrives.
    assert interleave_schedule(3, 5) == ("ssm",) * 3
    for n, r in ((24, 3), (24, 7), (12, 1)):
        sched = interleave_schedule(n, r)
        assert len(sched) == n
        assert sched.count("attn") == n // (r + 1)


def test_interleave_reduces_to_pure_stacks():
    """All-attn / all-ssm schedules equal manually composed residual stacks."""
    x = _x(length=24, d=6, seed=6)
    gates = jnp.asarray(np.random.default_rng(6).uniform(0.6, 0.95, size=6))
    w = 24  # full attention inside each block

    y = interleave_hybrid(x, ("attn",) * 3, gates, w)
    expected = x
    for _ in range(3):
        expected = expected + full_causal_attention(expected)
    assert_allclose(np.asarray(y), np.asarray(expected), rtol=0, atol=1e-12)

    y = interleave_hybrid(x, ("ssm",) * 4, gates, w)
    expected = x
    for _ in range(4):
        expected = expected + gated_decay_ssm(expected, gates)
    assert_allclose(np.asarray(y), np.asarray(expected), rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Decode-cost accounting.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_blocks,ratio,window,d",
    [(24, 3, 1024, 1024), (24, 7, 1024, 1024), (8, 3, 16, 8), (5, 0, 4, 3)],
)
def test_cost_formula_matches_materialised_buffers(n_blocks, ratio, window, d):
    schedule = interleave_schedule(n_blocks, ratio)
    costs = decode_state_floats(schedule, window, d)
    buffers = decode_buffers(schedule, window, d)
    assert sum(int(b.size) for b in buffers) == costs["total"]
    n_attn = schedule.count("attn")
    assert costs["kv_floats"] == n_attn * 2 * window * d
    assert costs["ssm_floats"] == (n_blocks - n_attn) * d
    assert costs["total"] == costs["kv_floats"] + costs["ssm_floats"]


def test_production_reference_costs():
    """The §14.3 reference numbers quoted in prose (n=24, d=1024, w=1024, L=65536)."""
    full = full_attention_decode_floats(24, 65536, 1024)
    assert full == 3_221_225_472
    sched3 = interleave_schedule(24, 3)
    assert decode_state_floats(sched3, 1024, 1024)["total"] == 12_601_344
    sched7 = interleave_schedule(24, 7)
    assert decode_state_floats(sched7, 1024, 1024)["total"] == 6_312_960
    # Ratios quoted in the chapter: 255.6x and 510.3x.
    assert full / 12_601_344 == pytest.approx(255.6, abs=0.05)
    assert full / 6_312_960 == pytest.approx(510.3, abs=0.05)


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------


def test_argument_validation():
    x = _x(length=8, d=4)
    gates = jnp.full(4, 0.5)
    with pytest.raises(ValueError):
        sliding_window_attention(x, 0)
    with pytest.raises(ValueError):
        sliding_window_attention(jnp.zeros(8), 4)  # not (L, d)
    with pytest.raises(ValueError):
        gated_decay_ssm(x, jnp.full(3, 0.5))  # wrong gate width
    with pytest.raises(ValueError):
        gated_decay_ssm(x, jnp.full(4, 1.5))  # gate out of range
    with pytest.raises(ValueError):
        parallel_gated_hybrid(x, 1.2, gates, 4)  # mixing gate out of range
    with pytest.raises(ValueError):
        parallel_gated_hybrid(x, jnp.zeros(3), gates, 4)  # wrong gate shape
    with pytest.raises(ValueError):
        sequential_hybrid(x, gates, 4, order=("ssm", "ssm"))  # not a permutation
    with pytest.raises(ValueError):
        interleave_hybrid(x, (), gates, 4)  # empty schedule
    with pytest.raises(ValueError):
        interleave_hybrid(x, ("ssm", "mlp"), gates, 4)  # unknown layer
    with pytest.raises(ValueError):
        interleave_schedule(0, 3)
    with pytest.raises(ValueError):
        interleave_schedule(8, -1)
    with pytest.raises(ValueError):
        decode_state_floats(("ssm", "mlp"), 4, 4)
    with pytest.raises(ValueError):
        decode_buffers(("ssm",), 0, 4)
    with pytest.raises(ValueError):
        full_attention_decode_floats(0, 8, 4)
