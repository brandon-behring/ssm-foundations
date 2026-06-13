r"""Tests for Chapter 17 §17.3 — the B end-to-end pipeline (b_integration).

Pins the integrated readout (effective state size ↔ probe accuracy ↔ cross-entropy across the
predictor family on the reference HMM) and its reductions to the originating chapters: the probe
accuracies ARE ch16's ``probe_signature``, the paired comparison IS ch16's, the CE is ch14's, the
effective state size is ch15's. The headline is the NEW coherence; the reductions confirm reuse.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch14.jax import two_timescale as tt  # noqa: E402
from companions.ch15.jax import lyapunov_diagnostics as ld  # noqa: E402
from companions.ch16.jax import protocol as pr  # noqa: E402
from companions.ch17.jax import b_integration as b  # noqa: E402


def _readout():
    return b.disentanglement_readout()


def test_probe_monotone_in_effective_state_size() -> None:
    """The integration claim: regime-probe accuracy increases with effective state size."""
    r = _readout()
    assert r["probe_monotone_in_ess"] is True
    assert r["ess_order"] == ["unigram", "decay", "full"]


def test_effective_state_size_values() -> None:
    """Pinned ESS: unigram collapses to 1, decay < full (≈ K=4 regimes)."""
    r = _readout()
    assert abs(r["ess"]["unigram"] - 1.0) < 1e-12          # rank-one uniform -> ESS = 1 exactly
    assert 3.6 < r["ess"]["decay"] < 3.7                   # measured 3.657
    assert 3.99 < r["ess"]["full"] <= 4.0                  # measured 3.998 (K = 4)
    assert r["ess"]["full"] > r["ess"]["decay"] > r["ess"]["unigram"]


def test_unigram_transition_ess_is_one() -> None:
    hmm, _, _ = pr.reference_instance()
    t = b.predictor_transition(hmm, "unigram", 0.2)
    # reduction: our ESS path == ch15's effective_state_size on the eigenvalue magnitudes
    mags = np.abs(np.asarray(jnp.linalg.eigvals(jnp.asarray(t))))
    assert abs(b.effective_state_size_of(t) - ld.effective_state_size(mags)) < 1e-12
    assert abs(b.effective_state_size_of(t) - 1.0) < 1e-12


def test_probe_reduces_to_ch16_probe_signature() -> None:
    """The readout's probe accuracies ARE ch16's probe_signature (exact reuse)."""
    hmm, tokens, regimes = pr.reference_instance()
    sig = pr.probe_signature(hmm, tokens, regimes)
    r = _readout()
    for n in ("full", "decay", "unigram"):
        assert abs(r["probe_acc"][n] - float(sig[n])) < 1e-12
    # caption values (deterministic reference instance)
    assert abs(r["probe_acc"]["full"] - 0.8390) < 1e-3
    assert abs(r["probe_acc"]["decay"] - 0.6912) < 1e-3
    assert abs(r["probe_acc"]["unigram"] - 0.5476) < 1e-3


def test_cross_entropy_reduces_to_ch14() -> None:
    """The readout's CE is ch14's mean_cross_entropy; loss falls as memory grows."""
    hmm, tokens, _ = pr.reference_instance()
    burn = pr._REF_BURN
    ce_full_ch14 = tt.mean_cross_entropy(tt.forward_filter_predictions(hmm, tokens)[0], tokens, burn)
    r = _readout()
    assert abs(r["ce"]["full"] - ce_full_ch14) < 1e-12
    assert abs(r["ce"]["full"] - 1.9289) < 1e-3
    assert r["ce"]["full"] < r["ce"]["decay"] < r["ce"]["unigram"]   # 1.929 < 1.980 < 2.425


def test_paired_comparison_reduces_to_ch16() -> None:
    """The decay-vs-full per-token CE gap: paired SE far below unpaired (shared difficulty removed)."""
    r = _readout()
    p = r["paired_decay_vs_full"]
    assert abs(p["mean_diff"] - 0.0512) < 1e-3
    assert abs(p["se_paired"] - 0.0033) < 1e-3
    assert abs(p["se_unpaired"] - 0.0135) < 1e-3
    assert p["se_paired"] < 0.5 * p["se_unpaired"]      # paired wins (corr ~0.94)
    assert p["correlation"] > 0.9


def test_validation() -> None:
    hmm, _, _ = pr.reference_instance()
    with pytest.raises(ValueError):
        b.predictor_transition(hmm, "bogus", 0.2)
    with pytest.raises(ValueError):
        b.predictor_predictions(hmm, jnp.array([0, 1, 2]), "bogus", 0.2)
