r"""Chapter 16 §§16.3 & 16.5 — the evaluation-protocol toolkit.

Three protocol components around the ch14 §14.6 two-timescale task (imported,
not copied, from ``companions/ch14/jax/two_timescale.py``), each deterministic
and exactly checkable:

**The composite restriction** (:func:`composite_filter_predictions`) — the
predictor ch14 deferred to the protocol: a window-$w$ filter whose
window-edge prior is not uniform but the *decayed carried prior* — the
$\lambda$-decay filter's posterior at the edge position. It is the hybrid
idealization (cheap carried state outside the window, exact pairwise
computation inside) and it comes with two exact identities:

* ``lam=None`` (uniform edge prior) reproduces ch14's
  ``window_filter_predictions`` exactly — a cross-module check between two
  independently written implementations;
* ``lam = lambda*(eps)`` reproduces the **full Bayes filter** exactly at
  every window size, because the matched decay filter *is* the optimal
  filter (Theorem ``ch14:matched-decay-optimal``) and the within-window
  recursion uses the true transition.

Between the two, a *mistimed* carried prior still beats both pure
restrictions at the reference operating point — measured, not assumed.

**The probe signature** (:func:`probe_signature`) — pilot B's
disentanglement axis demonstrated on idealized states. Each restriction's
internal regime prior $P(z_{t+1} \mid \text{information used})$ is probed
with a closed-form ridge regression (fit on the first half, scored on the
second half — held-out discipline, no iterative training) against the true
regime labels. The accuracy profile across restrictions is the
"disentanglement signature": *where the slow variable lives*. The
companion proposition's demo (:func:`recover_prior_from_predictive`) inverts
the predictive distribution back to the regime prior through the emission
matrix — near-Bayes prediction *forces* the regime information to be present
(Proposition ``ch16:probe-recoverability``); whether a linear probe finds it
at a particular layer of a *trained* network is B's empirical question, not
a theorem.

**Comparison statistics** (:func:`paired_comparison_stats`,
:func:`gaussian_max_inflation`) — the §16.3 honesty toolkit: paired vs
unpaired standard errors on per-token loss differences (same-stream
evaluation makes the pairing correlation large, so the paired SE is the one
that reflects the actual uncertainty), and the max-of-$k$ selection
inflation $\mathbb{E}[\max_k] \le \sigma\sqrt{2\ln k}$ that a
pick-the-best-variant workflow silently adds (Proposition
``ch16:selection-inflation``), measured on Gaussian draws and on a
best-of-$k$ embedding-seed sweep of the MQAR reader.

Port credit
-----------
Greenfield; consumes ``companions.ch14.jax.two_timescale`` (task + filters)
and ``companions.ch16.jax.mqar`` (readers) as libraries. The reference
instance replicates ch14's figure constants and key derivation, so every
cross-entropy printed here is directly comparable with the ch14 §14.6
numbers.

Usage
-----
::

    PYTHONPATH=. python companions/ch16/jax/protocol.py
"""

from __future__ import annotations

from pathlib import Path

import jax

# Enable float64 before any jnp array is created (matches Chapters 4, 7-14).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from companions.ch14.jax.two_timescale import (  # noqa: E402
    TwoTimescaleHMM,
    decay_filter_predictions,
    epsilon_to_lambda,
    forward_filter_predictions,
    make_hmm,
    mean_cross_entropy,
    sample_sequence,
    unigram_filter_predictions,
    window_filter_predictions,
)
from companions.ch16.jax.mqar import (  # noqa: E402
    accuracy as mqar_accuracy,
)
from companions.ch16.jax.mqar import (  # noqa: E402
    make_mqar,
    outer_product_reader,
)

__all__ = [
    "reference_instance",
    "composite_filter_predictions",
    "composite_filter_predictions_naive",
    "filter_regime_priors",
    "per_token_log_losses",
    "paired_comparison_stats",
    "ridge_probe_accuracy",
    "probe_signature",
    "emission_min_singular_value",
    "recover_prior_from_predictive",
    "gaussian_max_inflation",
    "max_inflation_bound",
    "best_of_k_seed_inflation",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch16"

# Reference two-timescale configuration: identical constants and key
# derivation to ch14's figure instance, so cross-entropies match the ch14
# §14.6 printed numbers exactly.
_REF_REGIMES = 4
_REF_VOCAB = 12
_REF_CONCENTRATION = 0.3
_REF_OVERLAP = 0.4
_REF_EPS = 0.02
_REF_LENGTH = 8192
_REF_BURN = 128
_REF_SEED = 0
_REF_WINDOW = 8
_REF_LAM_MIS = epsilon_to_lambda(0.2, _REF_REGIMES)  # mistimed: 10x too fast

# Selection-inflation experiment constants.
_INFLATION_KS = (1, 2, 4, 8, 16, 32, 64)
_INFLATION_TRIALS = 2000
_SEED_SWEEP_K = 16
_SEED_SWEEP_ITEMS = 64
_SEED_SWEEP_PAIRS = 48
_SEED_SWEEP_DIM = 24


def reference_instance() -> tuple[TwoTimescaleHMM, jnp.ndarray, jnp.ndarray]:
    """The ch14 reference task instance, with the regime labels kept.

    Replicates ``companions/ch14/jax/two_timescale.py``'s figure-instance key
    derivation (PRNGKey(0) split into bigram and sequence keys; sequence key
    folded with the eps/overlap fingerprint) so the tokens — and therefore
    every cross-entropy — are identical to ch14's; the regime path, which
    ch14 discarded, is returned for probing.
    """
    key = jax.random.PRNGKey(_REF_SEED)
    key_bigrams, key_seq = jax.random.split(key)
    hmm = make_hmm(
        key_bigrams, _REF_REGIMES, _REF_VOCAB, _REF_EPS, _REF_CONCENTRATION, _REF_OVERLAP
    )
    seq_key = jax.random.fold_in(key_seq, int(_REF_EPS * 1e9) + int(_REF_OVERLAP * 1e3))
    tokens, regimes = sample_sequence(seq_key, hmm, _REF_LENGTH)
    return hmm, tokens, regimes


# ---------------------------------------------------------------------------
# The composite restriction (the predictor ch14 deferred to the protocol).
# ---------------------------------------------------------------------------


def composite_filter_predictions(
    hmm: TwoTimescaleHMM, tokens: jnp.ndarray, window: int, lam: float | None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Window-$w$ filter seeded at its edge with the $\lambda$-decayed carried prior.

    At position $i$ the prediction uses exact Bayes updates (true transition
    $T(\varepsilon)$) over the last $w$ tokens, but the regime prior at the
    window edge is the $\lambda$-decay filter's posterior at the edge
    position — the carried state a fixed-decay recurrence could actually
    supply — instead of the uniform restart. ``lam=None`` keeps the uniform
    restart and must reproduce ch14's ``window_filter_predictions`` exactly;
    ``lam = lambda*(eps)`` makes the carried prior the *true* posterior, so
    the composite reproduces the full filter exactly at every window size.

    Returns
    -------
    preds : jnp.ndarray, shape (L-1, V)
        ``preds[i]`` is the predictive distribution over ``tokens[i+1]``.
    priors : jnp.ndarray, shape (L-1, K)
        ``priors[i]`` is the regime prior $P(z_{i+1} \mid \cdot)$ the
        prediction used — the probe feature.
    """
    if tokens.ndim != 1 or tokens.shape[0] < 2:
        raise ValueError(f"tokens must be a 1-D sequence of length >= 2; got {tokens.shape}")
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}")
    length = int(tokens.shape[0])
    k = hmm.num_regimes
    uniform = jnp.full((k,), 1.0 / k)
    if lam is None:
        edge_priors = jnp.tile(uniform, (length, 1))
    else:
        _, decay_posts = decay_filter_predictions(hmm, tokens, lam)
        # decay_posts[t] estimates P(z_{t+1} | x_{1:t+1}); an edge at position
        # e >= 1 therefore carries decay_posts[e-1]; an edge at e <= 0 is the
        # sequence start, whose true prior is uniform.
        edge_priors = jnp.vstack([uniform[None], decay_posts])
    n_steps = min(window, length) - 1  # pair-updates inside one window

    def one_position(i: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        edge = i - n_steps  # window covers tokens edge .. i
        p0 = jnp.where(edge <= 0, uniform, edge_priors[jnp.clip(edge, 0, length - 1)])
        s_idx = edge + jnp.arange(n_steps)
        valid = s_idx >= 0
        s_clip = jnp.clip(s_idx, 0, length - 2)
        pair_a = tokens[s_clip]
        pair_b = tokens[s_clip + 1]

        def step(p, inp):
            a, b, ok = inp
            prior = p @ hmm.transition
            post = prior * hmm.bigrams[:, a, b]
            post = post / jnp.sum(post)
            return jnp.where(ok, post, p), None

        p_final, _ = jax.lax.scan(step, p0, (pair_a, pair_b, valid))
        prior = p_final @ hmm.transition
        return prior @ hmm.bigrams[:, tokens[i], :], prior

    if n_steps == 0:
        # window == 1: the edge prior is the carried prior at position i.
        p0 = edge_priors[jnp.arange(length - 1)]
        priors = p0 @ hmm.transition
        preds = jnp.einsum("ik,ikv->iv", priors, hmm.bigrams[:, tokens[:-1], :].swapaxes(0, 1))
        return preds, priors
    preds, priors = jax.vmap(one_position)(jnp.arange(length - 1))
    return preds, priors


def composite_filter_predictions_naive(
    hmm: TwoTimescaleHMM, tokens: jnp.ndarray, window: int, lam: float | None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Python-loop oracle for :func:`composite_filter_predictions` (NumPy throughout)."""
    if tokens.ndim != 1 or tokens.shape[0] < 2:
        raise ValueError(f"tokens must be a 1-D sequence of length >= 2; got {tokens.shape}")
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}")
    transition = np.asarray(hmm.transition)
    bigrams = np.asarray(hmm.bigrams)
    toks = np.asarray(tokens)
    length = toks.shape[0]
    k = transition.shape[0]
    uniform = np.full(k, 1.0 / k)
    if lam is None:
        edge_priors = np.tile(uniform, (length, 1))
    else:
        _, decay_posts = decay_filter_predictions(hmm, tokens, lam)
        edge_priors = np.vstack([uniform[None], np.asarray(decay_posts)])
    preds = np.zeros((length - 1, bigrams.shape[-1]))
    priors = np.zeros((length - 1, k))
    for i in range(length - 1):
        edge = i - window + 1
        p = uniform if edge <= 0 else edge_priors[edge]
        for s in range(max(0, edge), i):
            prior = p @ transition
            post = prior * bigrams[:, toks[s], toks[s + 1]]
            p = post / post.sum()
        prior = p @ transition
        preds[i] = prior @ bigrams[:, toks[i], :]
        priors[i] = prior
    return jnp.asarray(preds), jnp.asarray(priors)


def filter_regime_priors(
    hmm: TwoTimescaleHMM, tokens: jnp.ndarray, kind: str, lam: float | None = None
) -> jnp.ndarray:
    r"""The regime prior $P(z_{i+1} \mid \cdot)$ each carried-state filter used, per position.

    Reconstructed from the filter's posterior trajectory: the prior feeding
    prediction $i$ is the previous posterior pushed through the filter's own
    transition (uniform initial posterior at $i = 0$). ``kind`` is ``"full"``,
    ``"decay"`` (requires ``lam``), or ``"unigram"``.

    Returns
    -------
    jnp.ndarray, shape (L-1, K)
    """
    if kind == "full":
        _, posts = forward_filter_predictions(hmm, tokens)
        push = hmm.transition
    elif kind == "decay":
        if lam is None:
            raise ValueError("kind='decay' requires lam")
        from companions.ch14.jax.two_timescale import mixing_to_uniform_transition

        _, posts = decay_filter_predictions(hmm, tokens, lam)
        push = mixing_to_uniform_transition(hmm.num_regimes, lam)
    elif kind == "unigram":
        _, posts = unigram_filter_predictions(hmm, tokens)
        push = hmm.transition
    else:
        raise ValueError(f"unknown kind {kind!r}; expected full | decay | unigram")
    k = hmm.num_regimes
    uniform = jnp.full((1, k), 1.0 / k)
    prev = jnp.vstack([uniform, posts[:-1]])
    return prev @ push


# ---------------------------------------------------------------------------
# Comparison statistics (§16.3).
# ---------------------------------------------------------------------------


def per_token_log_losses(preds: jnp.ndarray, tokens: jnp.ndarray, burn: int = 0) -> jnp.ndarray:
    """Per-position next-token log losses (nats) after burn-in; mean equals ch14's CE."""
    if preds.ndim != 2 or tokens.ndim != 1 or preds.shape[0] != tokens.shape[0] - 1:
        raise ValueError(f"shape mismatch: preds {preds.shape} vs tokens {tokens.shape}")
    if not 0 <= burn < preds.shape[0]:
        raise ValueError(f"burn must be in [0, {preds.shape[0] - 1}]; got {burn}")
    targets = tokens[1:]
    picked = jnp.take_along_axis(preds, targets[:, None], axis=1)[:, 0]
    return -jnp.log(picked[burn:])


def paired_comparison_stats(items_a: jnp.ndarray, items_b: jnp.ndarray) -> dict[str, float]:
    r"""Paired vs unpaired standard errors for the mean difference of per-item scores.

    For per-item scores $a_i, b_i$ on the *same* items,

    .. math::

        \mathrm{SE}_{\text{paired}}^2 = \tfrac{1}{n}\widehat{\mathrm{Var}}(a - b), \qquad
        \mathrm{SE}_{\text{unpaired}}^2
            = \tfrac{1}{n}\bigl(\widehat{\mathrm{Var}}(a) + \widehat{\mathrm{Var}}(b)\bigr),

    so $\mathrm{SE}_{\text{paired}}^2 = \mathrm{SE}_{\text{unpaired}}^2 -
    \tfrac{2}{n}\widehat{\mathrm{Cov}}(a, b)$ — shared item difficulty
    (positive covariance) is subtracted off exactly (Proposition
    ``ch16:paired-comparison``). Sample statistics use the unbiased
    ``ddof=1`` normalisation.
    """
    a = np.asarray(items_a, dtype=float)
    b = np.asarray(items_b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError(f"need matching 1-D score arrays; got {a.shape}, {b.shape}")
    n = a.shape[0]
    if n < 2:
        raise ValueError("need at least 2 items")
    diff = a - b
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    cov = float(np.cov(a, b, ddof=1)[0, 1])
    return {
        "n": float(n),
        "mean_diff": float(np.mean(diff)),
        "se_paired": float(np.sqrt(np.var(diff, ddof=1) / n)),
        "se_unpaired": float(np.sqrt((var_a + var_b) / n)),
        "correlation": float(cov / np.sqrt(var_a * var_b)),
    }


def max_inflation_bound(k: int, sigma: float) -> float:
    r"""The sub-Gaussian maximal bound $\mathbb{E}[\max_{i \le k} X_i] \le \sigma\sqrt{2\ln k}$."""
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0; got {sigma}")
    return float(sigma * np.sqrt(2.0 * np.log(k)))


def gaussian_max_inflation(
    key: jax.Array, ks: tuple[int, ...], n_trials: int, sigma: float = 1.0
) -> np.ndarray:
    r"""Measured $\mathbb{E}[\max_{i \le k} X_i]$ for i.i.d. $\mathcal{N}(0, \sigma^2)$ draws.

    The selection-bias engine in isolation: $k$ equally good variants whose
    measured scores differ only by evaluation noise; reporting the best of
    $k$ inflates the estimate by $\mathbb{E}[\max_k]$, which the
    :func:`max_inflation_bound` envelope $\sigma\sqrt{2\ln k}$ dominates.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1; got {n_trials}")
    if any(k < 1 for k in ks):
        raise ValueError(f"all k must be >= 1; got {ks}")
    draws = sigma * jax.random.normal(key, (n_trials, max(ks)))
    return np.asarray(
        [float(jnp.mean(jnp.max(draws[:, :k], axis=1))) for k in ks]
    )


def best_of_k_seed_inflation(k: int = _SEED_SWEEP_K) -> dict[str, float]:
    """Selection inflation measured on the MQAR reader: pick the best of ``k`` seeds.

    ``k`` embedding seeds of the same outer-product reader (identical true
    accuracy by symmetry) are scored on the same battery of
    ``_SEED_SWEEP_ITEMS`` episodes; the best-of-``k`` score minus the
    across-seed mean is pure selection bias — what a "pick the best variant
    on the eval set" workflow silently reports.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2; got {k}")
    per_seed = []
    for seed in range(k):
        scores = []
        for item in range(_SEED_SWEEP_ITEMS):
            inst = make_mqar(
                jax.random.fold_in(jax.random.PRNGKey(_REF_SEED), 10_000 + item),
                _SEED_SWEEP_PAIRS,
                1024,
                64,
            )
            scores.append(
                mqar_accuracy(outer_product_reader(inst, _SEED_SWEEP_DIM, seed=seed), inst.answers)
            )
        per_seed.append(float(np.mean(scores)))
    per_seed_arr = np.asarray(per_seed)
    return {
        "mean": float(per_seed_arr.mean()),
        "best": float(per_seed_arr.max()),
        "inflation": float(per_seed_arr.max() - per_seed_arr.mean()),
        "spread": float(per_seed_arr.std(ddof=1)),
    }


# ---------------------------------------------------------------------------
# The probe signature (§16.5) — pilot B's disentanglement axis, idealized.
# ---------------------------------------------------------------------------


def ridge_probe_accuracy(
    features: jnp.ndarray, labels: jnp.ndarray, num_classes: int, alpha: float = 1e-6
) -> float:
    r"""Closed-form linear probe: ridge to one-hot labels, argmax, held-out accuracy.

    Fit $W = (X^\top X + \alpha I)^{-1} X^\top Y$ (intercept via an appended
    ones column) on the **first half** of the positions, score argmax
    accuracy on the **second half** — chronological split, no shuffling, no
    iterative training, so the probe itself obeys §16.3's held-out
    discipline and the whole pipeline stays deterministic.
    """
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError(f"need (n, d) features and (n,) labels; got {x.shape}, {y.shape}")
    if x.shape[0] < 4:
        raise ValueError("need at least 4 positions to split")
    if alpha <= 0.0:
        raise ValueError(f"alpha must be > 0; got {alpha}")
    if np.any((y < 0) | (y >= num_classes)):
        raise ValueError(f"labels must lie in [0, {num_classes - 1}]")
    n = x.shape[0]
    half = n // 2
    x1 = np.hstack([x, np.ones((n, 1))])
    onehot = np.eye(num_classes)[y[:half]]
    gram = x1[:half].T @ x1[:half] + alpha * np.eye(x1.shape[1])
    w = np.linalg.solve(gram, x1[:half].T @ onehot)
    pred = np.argmax(x1[half:] @ w, axis=1)
    return float(np.mean(pred == y[half:]))


def probe_signature(
    hmm: TwoTimescaleHMM,
    tokens: jnp.ndarray,
    regimes: jnp.ndarray,
    window: int = _REF_WINDOW,
    lam: float = _REF_LAM_MIS,
    burn: int = _REF_BURN,
) -> dict[str, float]:
    """Regime-probe accuracy from each restriction's internal prior — the signature.

    Probes the regime prior each predictor used (its only carried "state")
    against the true regime labels ``regimes[i+1]`` at the same positions,
    after the shared burn-in. The profile across restrictions is the
    idealized disentanglement signature: the full filter and the
    matched-prior composite carry the slow variable; a short uniform-restart
    window cannot.
    """
    if regimes.shape != tokens.shape:
        raise ValueError(f"regimes and tokens must align; got {regimes.shape}, {tokens.shape}")
    labels = regimes[1:][burn:]
    feats = {
        "full": filter_regime_priors(hmm, tokens, "full"),
        "composite": composite_filter_predictions(hmm, tokens, window, lam)[1],
        "decay": filter_regime_priors(hmm, tokens, "decay", lam=lam),
        "window": composite_filter_predictions(hmm, tokens, window, None)[1],
        "unigram": filter_regime_priors(hmm, tokens, "unigram"),
    }
    return {
        name: ridge_probe_accuracy(f[burn:], labels, hmm.num_regimes)
        for name, f in feats.items()
    }


# ---------------------------------------------------------------------------
# Probe-recoverability demo (the §16.5 proposition, numerically).
# ---------------------------------------------------------------------------


def emission_min_singular_value(hmm: TwoTimescaleHMM) -> float:
    r"""$\sigma = \min_a \sigma_{\min}(B[:, a, :])$ — the inversion constant.

    Positive iff every per-token emission block has full row rank $K$, i.e.
    the predictive distribution determines the regime prior.
    """
    b = np.asarray(hmm.bigrams)
    return float(min(np.linalg.svd(b[:, a, :], compute_uv=False)[-1] for a in range(b.shape[1])))


def recover_prior_from_predictive(
    hmm: TwoTimescaleHMM, preds: jnp.ndarray, tokens: jnp.ndarray
) -> jnp.ndarray:
    r"""Invert $\text{pred}_i = \text{prior}_i\, B[:, x_i, :]$ for the regime prior.

    Least-squares through the per-token emission block — exact (up to
    conditioning) whenever :func:`emission_min_singular_value` is positive.
    Demonstrates the proposition's mechanism: any predictor's regime prior
    is readable off its predictive distribution; nothing about the
    predictor's internals is used.
    """
    if preds.ndim != 2 or preds.shape[0] != tokens.shape[0] - 1:
        raise ValueError(f"shape mismatch: preds {preds.shape} vs tokens {tokens.shape}")
    b = np.asarray(hmm.bigrams)
    p = np.asarray(preds)
    toks = np.asarray(tokens)
    out = np.zeros((p.shape[0], b.shape[0]))
    for i in range(p.shape[0]):
        out[i], *_ = np.linalg.lstsq(b[:, toks[i], :].T, p[i], rcond=None)
    return jnp.asarray(out)


# ---------------------------------------------------------------------------
# Figures + measured numbers (§§16.3, 16.5).
# ---------------------------------------------------------------------------


def _fig_probe_signature(sig: dict[str, float], ceiling: float) -> None:
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
    order = ["full", "composite", "decay", "window", "unigram"]
    labels = {
        "full": "full filter",
        "composite": rf"composite ($w={_REF_WINDOW}$, mistimed $\lambda$)",
        "decay": r"decay (mistimed $\lambda$)",
        "window": rf"window ($w={_REF_WINDOW}$, uniform restart)",
        "unigram": "unigram (slow-only)",
    }
    colors = {
        "full": SSM_COLORS["baseline"],
        "composite": SSM_COLORS["accent"],
        "decay": SSM_COLORS["accent"],
        "window": SSM_COLORS["alert"],
        "unigram": SSM_COLORS["highlight"],
    }
    fig, ax = create_tufte_figure(figsize=(6.4, 3.6))
    y = np.arange(len(order))[::-1]
    vals = [sig[k] for k in order]
    ax.barh(y, vals, height=0.6, color=[colors[k] for k in order], alpha=0.85)
    for yi, v in zip(y, vals):
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(y, [labels[k] for k in order], fontsize=9)
    ax.axvline(1.0 / _REF_REGIMES, color=SSM_COLORS["alert"], lw=0.9, ls=":")
    ax.text(1.0 / _REF_REGIMES - 0.015, y[-1] - 0.62, rf"chance $1/K = {1.0 / _REF_REGIMES:.2f}$",
            rotation=90, va="bottom", ha="right", fontsize=7.5, color=SSM_COLORS["alert"])
    ax.axvline(ceiling, color=SSM_COLORS["baseline"], lw=0.9, ls="--")
    ax.text(ceiling + 0.012, y[-1] - 0.62, f"posterior-argmax ceiling {ceiling:.3f}",
            rotation=90, va="bottom", ha="left", fontsize=7.5, color=SSM_COLORS["baseline"])
    ax.set_xlim(0.0, 1.0)
    set_tufte_title(ax, "Probe signature: where the slow variable lives (measured)")
    set_tufte_labels(ax, "held-out regime-probe accuracy", None)
    fig.tight_layout()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in save_figure(fig, _OUT_DIR / "probe-signature", formats=("png",)):
        print(f"  wrote {p.relative_to(_REPO_ROOT)}")
    plt.close(fig)


def _fig_selection_inflation(measured: np.ndarray) -> None:
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
    ks = np.asarray(_INFLATION_KS)
    bound = np.asarray([max_inflation_bound(int(k), 1.0) for k in ks])
    fig, ax = create_tufte_figure(figsize=(6.4, 3.8))
    ax.plot(ks, measured, "o-", color=SSM_COLORS["accent"], ms=4,
            label=r"measured $\mathbb{E}[\max_k]$ (2000 trials)")
    ax.plot(ks, bound, "s--", color=SSM_COLORS["alert"], ms=4,
            label=r"bound $\sigma\sqrt{2\ln k}$")
    ax.set_xscale("log", base=2)
    set_tufte_title(ax, "Best-of-$k$ selection inflates by $\\sim\\sigma\\sqrt{2\\ln k}$")
    set_tufte_labels(ax, "variants compared $k$ (log)",
                     r"inflation ($\sigma$ units of eval noise)")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in save_figure(fig, _OUT_DIR / "selection-inflation", formats=("png",)):
        print(f"  wrote {p.relative_to(_REPO_ROOT)}")
    plt.close(fig)


def main() -> None:
    print("Chapter 16 — protocol.py")
    print("=" * 64)

    hmm, tokens, regimes = reference_instance()
    lam_star = epsilon_to_lambda(_REF_EPS, _REF_REGIMES)

    # Exact identities of the composite restriction.
    full, _ = forward_filter_predictions(hmm, tokens)
    win = window_filter_predictions(hmm, tokens, _REF_WINDOW)
    comp_uniform, _ = composite_filter_predictions(hmm, tokens, _REF_WINDOW, None)
    print(f"  composite(lam=None) == ch14 window:  max diff = "
          f"{float(jnp.max(jnp.abs(comp_uniform - win))):.2e}")
    comp_star, _ = composite_filter_predictions(hmm, tokens, _REF_WINDOW, lam_star)
    print(f"  composite(lambda*) == full filter:   max diff = "
          f"{float(jnp.max(jnp.abs(comp_star - full))):.2e}   (w={_REF_WINDOW})")

    # The measured ordering at the reference operating point.
    ce_full = mean_cross_entropy(full, tokens, _REF_BURN)
    ce_win = mean_cross_entropy(win, tokens, _REF_BURN)
    decay_preds, _ = decay_filter_predictions(hmm, tokens, _REF_LAM_MIS)
    ce_decay = mean_cross_entropy(decay_preds, tokens, _REF_BURN)
    comp_mis, _ = composite_filter_predictions(hmm, tokens, _REF_WINDOW, _REF_LAM_MIS)
    ce_comp = mean_cross_entropy(comp_mis, tokens, _REF_BURN)
    print(f"  reference config (eps={_REF_EPS}, overlap={_REF_OVERLAP}, L={_REF_LENGTH}, "
          f"burn={_REF_BURN}; mistimed lambda = {_REF_LAM_MIS:.4f} vs lambda* = {lam_star:.4f}):")
    print(f"    CE full filter             = {ce_full:.4f} nats")
    print(f"    CE window w={_REF_WINDOW}             = {ce_win:.4f}  "
          f"(excess {ce_win - ce_full:.4f})")
    print(f"    CE decay mistimed          = {ce_decay:.4f}  (excess {ce_decay - ce_full:.4f})")
    print(f"    CE composite (w + carried) = {ce_comp:.4f}  (excess {ce_comp - ce_full:.4f})")
    print(f"    composite beats window by {ce_win - ce_comp:.4f} nats and "
          f"decay by {ce_decay - ce_comp:.4f} nats")

    # Paired vs unpaired comparison on the same stream (window vs composite).
    losses_win = per_token_log_losses(win, tokens, _REF_BURN)
    losses_comp = per_token_log_losses(comp_mis, tokens, _REF_BURN)
    stats = paired_comparison_stats(losses_win, losses_comp)
    print(f"  paired comparison (window vs composite, n={int(stats['n'])} positions):")
    print(f"    mean diff = {stats['mean_diff']:.4f} nats; correlation = "
          f"{stats['correlation']:.4f}")
    print(f"    SE paired = {stats['se_paired']:.5f}  SE unpaired = {stats['se_unpaired']:.5f}  "
          f"(ratio {stats['se_unpaired'] / stats['se_paired']:.2f}x)")
    z = stats["mean_diff"] / stats["se_paired"]
    z_un = stats["mean_diff"] / stats["se_unpaired"]
    print(f"    z-score: paired {z:.1f} vs unpaired {z_un:.1f}")

    # Probe-recoverability: invert the predictive distribution for the prior.
    sigma_min = emission_min_singular_value(hmm)
    _, win_priors = composite_filter_predictions(hmm, tokens, _REF_WINDOW, None)
    recovered = recover_prior_from_predictive(hmm, comp_uniform, tokens)
    rec_err = float(jnp.max(jnp.abs(recovered - win_priors)))
    print(f"  probe-recoverability (sigma_min = {sigma_min:.4f} > 0):")
    print(f"    prior recovered from predictive, max abs err = {rec_err:.2e}")

    # The probe signature (pilot B's disentanglement axis on idealized states).
    sig = probe_signature(hmm, tokens, regimes)
    _, posts = forward_filter_predictions(hmm, tokens)
    ceiling = float(np.mean(
        np.argmax(np.asarray(posts)[_REF_BURN:], axis=1)
        == np.asarray(regimes[1:][_REF_BURN:])
    ))
    print(f"  probe signature (held-out regime accuracy; chance = {1.0 / _REF_REGIMES:.2f}, "
          f"posterior-argmax ceiling = {ceiling:.4f}):")
    for name in ("full", "composite", "decay", "window", "unigram"):
        print(f"    {name:<10} {sig[name]:.4f}")

    # Selection inflation: Gaussian engine + the MQAR best-of-k seed sweep.
    measured = gaussian_max_inflation(
        jax.random.PRNGKey(_REF_SEED + 1), _INFLATION_KS, _INFLATION_TRIALS
    )
    print("  selection inflation (sigma = 1):")
    print(f"    {'k':>4}  {'measured':>9}  {'bound':>7}")
    for k, m in zip(_INFLATION_KS, measured):
        print(f"    {k:>4}  {m:>9.4f}  {max_inflation_bound(k, 1.0):>7.4f}")
    sweep = best_of_k_seed_inflation()
    print(f"  best-of-{_SEED_SWEEP_K} MQAR seed sweep ({_SEED_SWEEP_ITEMS} shared episodes, "
          f"N={_SEED_SWEEP_PAIRS}, dim={_SEED_SWEEP_DIM}):")
    print(f"    mean accuracy {sweep['mean']:.4f}; best {sweep['best']:.4f}; "
          f"selection inflation +{sweep['inflation']:.4f} "
          f"(seed spread sigma = {sweep['spread']:.4f})")

    print("  figures:")
    _fig_probe_signature(sig, ceiling)
    _fig_selection_inflation(measured)


if __name__ == "__main__":
    main()
