r"""Cross-framework parity: torch hybrid block + two-timescale filters vs JAX.

Two layers (mirroring the ch09-ch12 torch suites):

* **standalone torch assertions** — the window/full-attention identity, the
  EMA closed form, exact parallel-gate reductions, the matched-decay
  optimality identity, and the buffers-vs-Parameters distinction — all
  meaningful without JAX present;
* **cross-framework parity** — recompute the JAX companions in-process on
  the same inputs and pin the torch outputs to them (``< 1e-9``, both
  float64). Tokens for the filter parity are drawn once in NumPy and fed to
  both implementations (PRNG streams never match across frameworks).
  Skipped if JAX is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float64)

from companions.ch14.torch.hybrid_block import (  # noqa: E402
    TinyGatedMixLayer,
    full_causal_attention,
    gated_decay_ssm,
    interleave_hybrid,
    interleave_schedule,
    parallel_gated_hybrid,
    sliding_window_attention,
)
from companions.ch14.torch.two_timescale import (  # noqa: E402
    TwoTimescaleHMM,
    decay_filter_predictions,
    epsilon_to_lambda,
    forward_filter_predictions,
    make_transition,
    mean_cross_entropy,
    mixing_to_uniform_transition,
    window_filter_predictions,
)


def _x(length=40, d=8, seed=0):
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.standard_normal((length, d)))


def _hmm(num_regimes=3, vocab=5, eps=0.1, seed=2):
    """A torch HMM with NumPy-Dirichlet bigram rows (shared with the JAX parity)."""
    rng = np.random.default_rng(seed)
    bigrams = rng.dirichlet(np.full(vocab, 0.5), size=(num_regimes, vocab))
    return TwoTimescaleHMM(
        transition=make_transition(num_regimes, eps),
        bigrams=torch.tensor(bigrams),
    )


# ---------------------------------------------------------------------------
# Standalone torch layer
# ---------------------------------------------------------------------------


def test_window_geq_length_is_full_attention():
    x = _x()
    diff = float(torch.max(torch.abs(sliding_window_attention(x, 40) - full_causal_attention(x))))
    assert diff < 1e-12


def test_ema_closed_form():
    rng = np.random.default_rng(1)
    d, length, g = 6, 32, 0.9
    xbar = rng.standard_normal(d)
    const = torch.tensor(np.tile(xbar, (length, 1)))
    h = gated_decay_ssm(const, torch.full((d,), g))
    t = np.arange(1, length + 1)[:, None]
    analytic = torch.tensor((1.0 - g**t) * xbar)
    assert float(torch.max(torch.abs(h - analytic))) < 1e-12


def test_parallel_gate_reductions_exact():
    x = _x(seed=3)
    gates = torch.tensor(np.random.default_rng(3).uniform(0.6, 0.95, size=8))
    w = 12
    y1 = parallel_gated_hybrid(x, 1.0, gates, w)
    y0 = parallel_gated_hybrid(x, 0.0, gates, w)
    assert torch.equal(y1, sliding_window_attention(x, w))
    assert torch.equal(y0, gated_decay_ssm(x, gates))


def test_matched_decay_equals_full_filter():
    """lambda = lambda*: the fixed-decay filter IS the optimal filter (torch)."""
    hmm = _hmm(eps=0.1)
    rng = np.random.default_rng(4)
    tokens = torch.tensor(rng.integers(0, hmm.vocab, size=96))
    lam_star = epsilon_to_lambda(0.1, hmm.num_regimes)
    full, _ = forward_filter_predictions(hmm, tokens)
    matched, _ = decay_filter_predictions(hmm, tokens, lam_star)
    assert float(torch.max(torch.abs(matched - full))) < 1e-12
    # And the transition-family identity behind it.
    t = make_transition(hmm.num_regimes, 0.1)
    m = mixing_to_uniform_transition(hmm.num_regimes, lam_star)
    assert float(torch.max(torch.abs(t - m))) < 1e-15


def test_window_covering_prefix_equals_full_filter():
    hmm = _hmm()
    rng = np.random.default_rng(5)
    tokens = torch.tensor(rng.integers(0, hmm.vocab, size=64))
    full, _ = forward_filter_predictions(hmm, tokens)
    win = window_filter_predictions(hmm, tokens, 64)
    assert float(torch.max(torch.abs(win - full))) < 1e-12


def test_tiny_layer_buffers_vs_parameters():
    decays = torch.tensor(np.random.default_rng(6).uniform(0.6, 0.95, size=8))
    layer = TinyGatedMixLayer(d=8, window=8, decays=decays)
    param_names = {name for name, _ in layer.named_parameters()}
    buffer_names = {name for name, _ in layer.named_buffers()}
    assert param_names == {"gate_logit"}  # the blend is learned
    assert buffer_names == {"decays"}  # the rates are fixed data
    x = _x(length=16, d=8, seed=7)
    y = layer(x)
    assert y.shape == x.shape
    # Gradients flow to the gate, not the buffer.
    y.square().mean().backward()
    assert layer.gate_logit.grad is not None
    assert not layer.decays.requires_grad


# ---------------------------------------------------------------------------
# Cross-framework parity (skipped without JAX)
# ---------------------------------------------------------------------------


def test_hybrid_block_parity_with_jax():
    pytest.importorskip("jax")
    from companions.ch14.jax import hybrid_block as jx

    import jax.numpy as jnp

    rng = np.random.default_rng(10)
    x_np = rng.standard_normal((48, 8))
    gates_np = rng.uniform(0.6, 0.95, size=8)
    g_vec_np = rng.uniform(0.0, 1.0, size=8)
    x_t, x_j = torch.tensor(x_np), jnp.asarray(x_np)
    gates_t, gates_j = torch.tensor(gates_np), jnp.asarray(gates_np)

    pairs = [
        (sliding_window_attention(x_t, 16), jx.sliding_window_attention(x_j, 16)),
        (full_causal_attention(x_t), jx.full_causal_attention(x_j)),
        (gated_decay_ssm(x_t, gates_t), jx.gated_decay_ssm(x_j, gates_j)),
        (
            parallel_gated_hybrid(x_t, torch.tensor(g_vec_np), gates_t, 16),
            jx.parallel_gated_hybrid(x_j, jnp.asarray(g_vec_np), gates_j, 16),
        ),
        (
            interleave_hybrid(x_t, interleave_schedule(8, 3), gates_t, 16),
            jx.interleave_hybrid(x_j, jx.interleave_schedule(8, 3), gates_j, 16),
        ),
    ]
    for got_t, got_j in pairs:
        np.testing.assert_allclose(
            got_t.detach().numpy(), np.asarray(got_j), rtol=0, atol=1e-9
        )
    assert interleave_schedule(24, 7) == jx.interleave_schedule(24, 7)


def test_filter_parity_with_jax():
    pytest.importorskip("jax")
    from companions.ch14.jax import two_timescale as jx

    import jax.numpy as jnp

    rng = np.random.default_rng(11)
    num_regimes, vocab, eps = 4, 7, 0.05
    bigrams_np = rng.dirichlet(np.full(vocab, 0.4), size=(num_regimes, vocab))
    tokens_np = rng.integers(0, vocab, size=160)

    hmm_t = TwoTimescaleHMM(
        transition=make_transition(num_regimes, eps), bigrams=torch.tensor(bigrams_np)
    )
    hmm_j = jx.TwoTimescaleHMM(
        transition=jx.make_transition(num_regimes, eps), bigrams=jnp.asarray(bigrams_np)
    )
    tokens_t, tokens_j = torch.tensor(tokens_np), jnp.asarray(tokens_np)

    full_t, post_t = forward_filter_predictions(hmm_t, tokens_t)
    full_j, post_j = jx.forward_filter_predictions(hmm_j, tokens_j)
    np.testing.assert_allclose(full_t.numpy(), np.asarray(full_j), rtol=0, atol=1e-9)
    np.testing.assert_allclose(post_t.numpy(), np.asarray(post_j), rtol=0, atol=1e-9)

    decay_t, _ = decay_filter_predictions(hmm_t, tokens_t, 0.3)
    decay_j, _ = jx.decay_filter_predictions(hmm_j, tokens_j, 0.3)
    np.testing.assert_allclose(decay_t.numpy(), np.asarray(decay_j), rtol=0, atol=1e-9)

    for w in (1, 8, 32):
        win_t = window_filter_predictions(hmm_t, tokens_t, w)
        win_j = jx.window_filter_predictions(hmm_j, tokens_j, w)
        np.testing.assert_allclose(win_t.numpy(), np.asarray(win_j), rtol=0, atol=1e-9)

    ce_t = mean_cross_entropy(full_t, tokens_t, burn=16)
    ce_j = jx.mean_cross_entropy(full_j, tokens_j, burn=16)
    assert ce_t == pytest.approx(ce_j, rel=0, abs=1e-9)
