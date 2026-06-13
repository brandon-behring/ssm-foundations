r"""Chapter 17 §17.3 — the B end-to-end pipeline: effective state size ↔ disentanglement.

The B (two-timescale-benchmarks) pilot tests whether an architecture disentangles a fast process
(token bigrams) from a slow one (regime drift). This module runs the full measurement pipeline
end-to-end on the *idealized* reference instance, composing three shipped companions:

* Chapter 14's two-timescale HMM + exact idealized predictors (``two_timescale``);
* Chapter 16's probe-signature protocol + paired comparison (``protocol``);
* Chapter 15's effective state size (``lyapunov_diagnostics``).

**The integrated signature (NEW — links three chapters' instruments on one instance):** for each
idealized predictor, its regime-*propagation* operator's effective state size (ch15), its
regime-recovery probe accuracy (ch16), and its predictive cross-entropy (ch14). The three cohere —
**disentanglement (probe accuracy) and predictive loss both track the predictor's effective state
size** — which is the template the B pilot runs on *trained* checkpoints (probe a real layer's
regime recovery against its measured effective state size). Here every predictor is an exact
idealization on a known HMM, so there is no training and no fitted model — the trained-checkpoint
program is the pilot's, in post_transformers.

Idiomatic-JAX / port credit
---------------------------
Greenfield composition. The HMM + predictors are ch14's, the probe/paired-comparison are ch16's,
the effective state size is ch15's. The ESS↔probe↔CE coherence across the predictor family is the
new object; the component values reduce to the originating chapters' (pinned in the tests).

Usage
-----
::

    PYTHONPATH=. python companions/ch17/jax/b_integration.py
"""

from __future__ import annotations

from pathlib import Path

import jax

# Enable float64 before any jnp array (and before importing ch14/15/16, which import jnp at load).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from companions.ch14.jax import two_timescale as tt  # noqa: E402
from companions.ch15.jax import lyapunov_diagnostics as ld  # noqa: E402
from companions.ch16.jax import protocol as pr  # noqa: E402

__all__ = [
    "predictor_transition",
    "effective_state_size_of",
    "predictor_cross_entropy",
    "predictor_predictions",
    "disentanglement_readout",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch17"

# The three predictors with a clean regime-propagation operator (and thus a well-defined ESS).
_PREDICTORS = ("full", "decay", "unigram")


def predictor_transition(hmm: "tt.TwoTimescaleHMM", name: str, lam: float) -> np.ndarray:
    r"""The $K\times K$ regime-propagation operator an idealized predictor uses.

    * ``full`` — the true sticky transition $T(\varepsilon)$ (the Bayes filter);
    * ``decay`` — the fixed-decay mixing $M(\lambda) = (1-\lambda)I + \lambda\,\mathbf{1}\bar u^\top$;
    * ``unigram`` — the rank-one collapse to the uniform regime prior (no regime memory).
    """
    k = hmm.num_regimes
    if name == "full":
        return np.asarray(hmm.transition)
    if name == "decay":
        return np.asarray(tt.mixing_to_uniform_transition(k, lam))
    if name == "unigram":
        return np.full((k, k), 1.0 / k)
    raise ValueError(f"unknown predictor {name!r}; expected one of {_PREDICTORS}")


def effective_state_size_of(transition: np.ndarray) -> float:
    r"""Effective state size of a regime-propagation operator (ch15, on $|\lambda_i|$).

    The participation ratio of the transition's squared spectral magnitudes — how many regime
    modes the propagation keeps alive step to step. Reuses
    :func:`companions.ch15.jax.lyapunov_diagnostics.effective_state_size`.
    """
    mags = np.abs(np.asarray(jnp.linalg.eigvals(jnp.asarray(transition))))
    return ld.effective_state_size(mags)


def predictor_predictions(hmm: "tt.TwoTimescaleHMM", tokens: jnp.ndarray, name: str, lam: float) -> jnp.ndarray:
    """Predictive distributions of an idealized predictor (ch14)."""
    if name == "full":
        return tt.forward_filter_predictions(hmm, tokens)[0]
    if name == "decay":
        return tt.decay_filter_predictions(hmm, tokens, lam)[0]
    if name == "unigram":
        return tt.unigram_filter_predictions(hmm, tokens)[0]
    raise ValueError(f"unknown predictor {name!r}; expected one of {_PREDICTORS}")


def predictor_cross_entropy(
    hmm: "tt.TwoTimescaleHMM", tokens: jnp.ndarray, name: str, lam: float, burn: int
) -> float:
    """Mean predictive cross-entropy of an idealized predictor (ch14)."""
    return tt.mean_cross_entropy(predictor_predictions(hmm, tokens, name, lam), tokens, burn)


def disentanglement_readout(lam: float | None = None) -> dict[str, object]:
    r"""The integrated readout on the reference instance: (ESS, probe accuracy, CE) per predictor.

    Returns the per-predictor effective state size (ch15), regime-recovery probe accuracy (ch16),
    and cross-entropy (ch14), plus the paired comparison of the full vs decay per-token loss (ch16)
    and the monotonicity flag (probe accuracy increases with effective state size). ``lam`` defaults
    to ch16's mistimed reference decay rate so the ``decay`` entries align with ``probe_signature``.
    """
    hmm, tokens, regimes = pr.reference_instance()
    if lam is None:
        lam = pr._REF_LAM_MIS
    burn = pr._REF_BURN
    probe = pr.probe_signature(hmm, tokens, regimes)  # ch16 defaults (window, lam, burn)

    ess = {n: effective_state_size_of(predictor_transition(hmm, n, lam)) for n in _PREDICTORS}
    ce = {n: predictor_cross_entropy(hmm, tokens, n, lam, burn) for n in _PREDICTORS}
    probe_acc = {n: float(probe[n]) for n in _PREDICTORS}

    # Paired comparison (ch16): the decay-vs-full per-token loss gap, with shared difficulty removed.
    ll_full = pr.per_token_log_losses(predictor_predictions(hmm, tokens, "full", lam), tokens, burn)
    ll_decay = pr.per_token_log_losses(predictor_predictions(hmm, tokens, "decay", lam), tokens, burn)
    paired = pr.paired_comparison_stats(ll_decay, ll_full)

    # The integration claim: probe accuracy is monotone in effective state size across the family.
    order = sorted(_PREDICTORS, key=lambda n: ess[n])
    monotone = all(probe_acc[order[i]] <= probe_acc[order[i + 1]] + 1e-12 for i in range(len(order) - 1))

    return {
        "predictors": _PREDICTORS,
        "ess": ess,
        "probe_acc": probe_acc,
        "ce": ce,
        "probe_full_set": {k: float(v) for k, v in probe.items()},
        "paired_decay_vs_full": paired,
        "ess_order": order,
        "probe_monotone_in_ess": monotone,
        "num_regimes": int(hmm.num_regimes),
        "lam": float(lam),
    }


# ---------------------------------------------------------------------------
# Figure + measured numbers (§17.3).
# ---------------------------------------------------------------------------


def _make_figure() -> None:
    import matplotlib.pyplot as plt

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        save_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    r = disentanglement_readout()
    preds = list(_PREDICTORS)
    ess = [r["ess"][n] for n in preds]
    probe = [r["probe_acc"][n] for n in preds]
    ce = [r["ce"][n] for n in preds]
    paired = r["paired_decay_vs_full"]

    print("Chapter 17 — b_integration.py")
    print("=" * 64)
    print(f"  B disentanglement readout (reference HMM, K={r['num_regimes']} regimes, lam={r['lam']:.4f}):")
    print(f"    {'predictor':<10}{'eff. state size':>16}{'probe acc':>12}{'cross-entropy':>15}")
    for n in preds:
        print(f"    {n:<10}{r['ess'][n]:>16.4f}{r['probe_acc'][n]:>12.4f}{r['ce'][n]:>15.4f}")
    print(f"    probe accuracy monotone in effective state size: {r['probe_monotone_in_ess']} "
          f"(order by ESS: {r['ess_order']})")
    print(f"    full probe set (ch16 probe_signature): "
          f"{ {k: round(v, 4) for k, v in r['probe_full_set'].items()} }")
    print(f"    paired decay-vs-full per-token CE: mean_diff={paired['mean_diff']:.4f}, "
          f"se_paired={paired['se_paired']:.4f} vs se_unpaired={paired['se_unpaired']:.4f} "
          f"(corr={paired['correlation']:.3f})")

    fig, axes = create_tufte_figure(1, 2, figsize=(11.0, 4.3))
    ax_link, ax_ce = axes  # type: ignore[misc]

    ax_link.scatter(ess, probe, s=70, color=SSM_COLORS["accent"], edgecolors="white",
                    linewidths=0.9, zorder=3)
    for x, y, n in zip(ess, probe, preds):
        ax_link.annotate(n, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    set_tufte_title(ax_link, "Disentanglement tracks effective state size")
    set_tufte_labels(ax_link, r"effective state size of the regime propagation", "regime-probe accuracy")

    colors = [SSM_COLORS["highlight"], SSM_COLORS["accent"], SSM_COLORS["alert"]]
    ax_ce.bar(preds, ce, color=colors, width=0.6)
    ax_ce.axhline(ce[0], color=SSM_COLORS["baseline"], lw=0.8, ls=":",
                  label=f"Bayes floor (full) = {ce[0]:.3f}")
    set_tufte_title(ax_ce, "Predictive loss falls as memory grows")
    set_tufte_labels(ax_ce, "predictor", "cross-entropy (nats)")
    ax_ce.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in save_figure(fig, _OUT_DIR / "b-disentanglement", formats=("png",)):
        print(f"  wrote {path.relative_to(_REPO_ROOT)}")
    plt.close(fig)


def main() -> None:
    _make_figure()


if __name__ == "__main__":
    main()
