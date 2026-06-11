r"""Chapter 14 §§14.2 & 14.6 — the two-timescale task and its exact predictors.

The seed of pilot B's two-timescale benchmark: a hidden Markov model whose
latent *regime* moves slowly while the visible *tokens* move fast,

* slow process: $z_t \in \{1..K\}$ with sticky transition
  $T(\varepsilon) = (1-\varepsilon) I + \tfrac{\varepsilon}{K-1}(J - I)$ —
  switch probability $\varepsilon \ll 1$ per step;
* fast process: a regime-conditioned bigram,
  $P(x_{t+1} \mid z_{t+1} = j,\ x_t) = B_j[x_t, \cdot]$ on a vocabulary of
  size $V$ (first regime and first token drawn uniform, so the filter
  starts exactly uniform).

The bigrams interpolate between a shared table and regime-specific ones,
$B_j = (1 - \eta)\, C + \eta\, D_j$ (``overlap`` $= \eta$): the dial that
controls how *hard the regimes are to tell apart locally*, i.e. the
**identification timescale** $\tau_{\mathrm{id}}$, which shrinks as the mean per-observation
discrimination $\bar{\imath}$ grows
(:func:`mean_pairwise_discrimination`). The task therefore has **three**
timescales — window $w$, identification $\tau_{\mathrm{id}}$, dwell
$1/\varepsilon$ — and the §14.6 design lesson it makes measurable is that a
two-timescale benchmark only separates architectures when
$w \ll \tau_{\mathrm{id}} \ll 1/\varepsilon$: with $\eta = 1$ (sharply
distinct regimes) a window of a dozen tokens is already near-optimal, and
nothing about carried state can be measured.

Because the model is known, every predictor below is *exact* — no training,
no approximation beyond the stated information restriction:

* :func:`forward_filter_predictions` — the Bayes-optimal predictor
  $P(x_{t+1} \mid x_{1:t})$ (the HMM forward filter);
* :func:`window_filter_predictions` — the **attention idealization**: the
  same Bayes computation restricted to the last $w$ tokens (uniform regime
  prior at the window edge; exact pairwise computation inside it);
* :func:`decay_filter_predictions` — the **fixed-decay state
  idealization**: the filter run with the mixing-to-uniform transition
  $M(\lambda) = (1-\lambda) I + \tfrac{\lambda}{K} J$ in place of
  $T(\varepsilon)$. Since $T(\varepsilon) = M(\lambda^*)$ exactly at
  $\lambda^* = \varepsilon K/(K-1)$, the matched decay filter *is* the
  optimal filter — and any fixed $\lambda \ne \lambda^*$ is a measurably
  mistimed slow manifold;
* :func:`unigram_filter_predictions` — the **slow-manifold-only
  idealization**: tracks the regime but replaces each bigram row with the
  regime's stationary unigram, discarding the fast token-local structure.

The §14.6 figures sweep $\varepsilon$ and $w$ and plot each restriction's
excess cross-entropy over the optimal filter; tests pin the exact
reductions (window $\ge L$ ≡ full filter; $\lambda = \lambda^*$ ≡ full
filter; uniform bigrams ⇒ every predictor at $\log V$) and check the filter
against a brute-force path-enumeration oracle on tiny instances.

Idiomatic-JAX note (NumPy->JAX teaching point)
----------------------------------------------
The windowed predictor is a *masked fixed-length scan* ``vmap``-ed over
positions — ragged early windows are handled by a validity mask that holds
the carry fixed, not by Python-side slicing. The per-position Python loop
(:func:`window_filter_predictions_naive`) is the readable oracle.

Port credit
-----------
Greenfield: authored for this chapter from the pilot-B kickoff task spec
(post_transformers ``notes/research_kickoff_b_two_timescale_benchmarks.md``);
no predecessor implementation exists.

Usage
-----
::

    PYTHONPATH=. python companions/ch14/jax/two_timescale.py
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax

# Enable float64 before any jnp array is created (matches Chapters 4, 7-12).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

__all__ = [
    "TwoTimescaleHMM",
    "make_transition",
    "mixing_to_uniform_transition",
    "epsilon_to_lambda",
    "make_hmm",
    "mean_pairwise_discrimination",
    "sample_sequence",
    "forward_filter_predictions",
    "decay_filter_predictions",
    "unigram_filter_predictions",
    "window_filter_predictions",
    "window_filter_predictions_naive",
    "enumeration_predictions",
    "mean_cross_entropy",
    "regime_stationary_unigrams",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch14"


class TwoTimescaleHMM(NamedTuple):
    """Model parameters: sticky transition (K, K) + per-regime bigrams (K, V, V)."""

    transition: jnp.ndarray
    bigrams: jnp.ndarray

    @property
    def num_regimes(self) -> int:
        return self.transition.shape[0]

    @property
    def vocab(self) -> int:
        return self.bigrams.shape[-1]


# ---------------------------------------------------------------------------
# Model construction.
# ---------------------------------------------------------------------------


def make_transition(num_regimes: int, eps: float) -> jnp.ndarray:
    r"""Sticky transition $T(\varepsilon) = (1-\varepsilon)I + \tfrac{\varepsilon}{K-1}(J-I)$.

    Stay with probability $1 - \varepsilon$, otherwise jump uniformly to one
    of the $K - 1$ other regimes. Valid for $0 \le \varepsilon \le 1$.

    Parameters
    ----------
    num_regimes : int
        $K \ge 2$.
    eps : float
        Per-step switch probability.

    Returns
    -------
    jnp.ndarray, shape (K, K), row-stochastic
    """
    if num_regimes < 2:
        raise ValueError(f"num_regimes must be >= 2; got {num_regimes}")
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"eps must be in [0, 1]; got {eps}")
    k = num_regimes
    eye = jnp.eye(k)
    return (1.0 - eps) * eye + (eps / (k - 1)) * (jnp.ones((k, k)) - eye)


def mixing_to_uniform_transition(num_regimes: int, lam: float) -> jnp.ndarray:
    r"""Fixed-decay transition $M(\lambda) = (1-\lambda)I + \tfrac{\lambda}{K}J$.

    The "forget toward uniform at rate $\lambda$" family — the transition a
    fixed-decay state model implicitly assumes. Coincides with the true
    sticky transition exactly at $\lambda^* = \varepsilon K/(K-1)$
    (:func:`epsilon_to_lambda`).

    Parameters
    ----------
    num_regimes : int
        $K \ge 2$.
    lam : float
        Mixing rate in $[0, 1]$.

    Returns
    -------
    jnp.ndarray, shape (K, K), row-stochastic
    """
    if num_regimes < 2:
        raise ValueError(f"num_regimes must be >= 2; got {num_regimes}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1]; got {lam}")
    k = num_regimes
    return (1.0 - lam) * jnp.eye(k) + (lam / k) * jnp.ones((k, k))


def epsilon_to_lambda(eps: float, num_regimes: int) -> float:
    r"""The matching rate $\lambda^* = \varepsilon K / (K - 1)$.

    Solves $M(\lambda^*) = T(\varepsilon)$ entrywise; requires
    $\varepsilon \le (K-1)/K$ so that $\lambda^* \le 1$.
    """
    if num_regimes < 2:
        raise ValueError(f"num_regimes must be >= 2; got {num_regimes}")
    lam = eps * num_regimes / (num_regimes - 1)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(
            f"eps={eps} gives lambda*={lam:.4f} outside [0, 1] for K={num_regimes}"
        )
    return lam


def make_hmm(
    key: jax.Array,
    num_regimes: int,
    vocab: int,
    eps: float,
    concentration: float = 0.3,
    overlap: float = 1.0,
) -> TwoTimescaleHMM:
    r"""Draw a two-timescale HMM: sticky transition + interpolated Dirichlet bigrams.

    A shared table $C$ and regime tables $D_j$ are drawn with i.i.d.
    $\mathrm{Dirichlet}(\alpha \mathbf{1}_V)$ rows, then interpolated:

    .. math:: B_j = (1 - \eta)\, C + \eta\, D_j, \qquad \eta = \text{overlap}.

    ``overlap=1`` gives fully distinct regimes (identifiable from a handful
    of tokens); small ``overlap`` makes regimes nearly indistinguishable
    locally, stretching the identification timescale
    $\tau_{\mathrm{id}} \sim \log K / \bar{\imath}$
    (:func:`mean_pairwise_discrimination`) — the dial a two-timescale
    benchmark must control. Every entry is strictly positive almost surely
    (validated, since the filters take products of these entries).

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    num_regimes : int
        $K \ge 2$.
    vocab : int
        $V \ge 2$.
    eps : float
        Switch probability.
    concentration : float
        Dirichlet concentration $\alpha > 0$ per symbol.
    overlap : float
        Interpolation weight $\eta \in (0, 1]$ toward the regime-specific
        tables. ($\eta = 0$ would make all regimes identical — degenerate,
        so it is rejected.)

    Returns
    -------
    TwoTimescaleHMM
    """
    if vocab < 2:
        raise ValueError(f"vocab must be >= 2; got {vocab}")
    if concentration <= 0.0:
        raise ValueError(f"concentration must be > 0; got {concentration}")
    if not 0.0 < overlap <= 1.0:
        raise ValueError(f"overlap must be in (0, 1]; got {overlap}")
    transition = make_transition(num_regimes, eps)
    key_common, key_distinct = jax.random.split(key)
    alpha = jnp.full((vocab,), concentration)
    common = jax.random.dirichlet(key_common, alpha, shape=(vocab,))
    distinct = jax.random.dirichlet(key_distinct, alpha, shape=(num_regimes, vocab))
    bigrams = (1.0 - overlap) * common[None] + overlap * distinct
    if not bool(jnp.all(bigrams > 0.0)):
        raise ValueError("bigram rows must be strictly positive; re-draw with a new key")
    return TwoTimescaleHMM(transition=transition, bigrams=bigrams)


def mean_pairwise_discrimination(hmm: TwoTimescaleHMM) -> float:
    r"""Mean per-observation discrimination $\bar{\imath}$ between regimes, in nats.

    The average over ordered regime pairs $z \ne z'$, with uniform weight on
    the source symbol $a$, of $\mathrm{KL}(B_z[a, \cdot] \,\|\, B_{z'}[a,
    \cdot])$ — the expected one-step log-likelihood-ratio drift a filter
    accumulates *per token* when separating $z$ from $z'$. Smaller
    $\bar{\imath}$ means regimes take longer to tell apart; it is a
    *hardness diagnostic*, not a calibrated timescale — the §14.6 crossover
    figure measures the actual window scale, which on this task sits an
    order of magnitude above $\log K / \bar{\imath}$. Uniform source
    weighting is a deliberate simplification — the exact drift would weight
    $a$ by its within-regime stationary frequency.

    Returns
    -------
    float
        $\bar{\imath} \ge 0$; equals 0 iff all regime tables coincide.
    """
    b = hmm.bigrams
    k = hmm.num_regimes
    # kl[z, z', a] = KL(B_z[a] || B_z'[a]) with uniform weight over a.
    log_ratio = jnp.log(b)[:, None, :, :] - jnp.log(b)[None, :, :, :]
    kl = jnp.sum(b[:, None, :, :] * log_ratio, axis=-1)
    off_diagonal = 1.0 - jnp.eye(k)
    per_pair = jnp.mean(kl, axis=-1)  # uniform over source symbol a
    return float(jnp.sum(per_pair * off_diagonal) / (k * (k - 1)))


def sample_sequence(
    key: jax.Array, hmm: TwoTimescaleHMM, length: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Sample ``(tokens, regimes)`` of the given length.

    $z_1 \sim \mathrm{Unif}(K)$, $x_1 \sim \mathrm{Unif}(V)$ (independent of
    $z_1$, so the filter's uniform initial posterior is exact); thereafter
    $z_{t+1} \sim T[z_t, \cdot]$ and $x_{t+1} \sim B_{z_{t+1}}[x_t, \cdot]$.

    Returns
    -------
    tokens : jnp.ndarray, shape (length,), int
    regimes : jnp.ndarray, shape (length,), int
    """
    if length < 2:
        raise ValueError(f"length must be >= 2; got {length}")
    key_z0, key_x0, key_scan = jax.random.split(key, 3)
    z0 = jax.random.randint(key_z0, (), 0, hmm.num_regimes)
    x0 = jax.random.randint(key_x0, (), 0, hmm.vocab)
    log_t = jnp.log(hmm.transition)
    log_b = jnp.log(hmm.bigrams)

    def step(carry, step_key):
        z, x = carry
        key_z, key_x = jax.random.split(step_key)
        z_new = jax.random.categorical(key_z, log_t[z])
        x_new = jax.random.categorical(key_x, log_b[z_new, x])
        return (z_new, x_new), (z_new, x_new)

    keys = jax.random.split(key_scan, length - 1)
    _, (zs, xs) = jax.lax.scan(step, (z0, x0), keys)
    tokens = jnp.concatenate([x0[None], xs])
    regimes = jnp.concatenate([z0[None], zs])
    return tokens, regimes


# ---------------------------------------------------------------------------
# Exact predictors. All return ``preds`` of shape (L-1, V):
# preds[i] = P(x_{i+1} | information available at position i)  (0-indexed).
# ---------------------------------------------------------------------------


def _validate_tokens(hmm: TwoTimescaleHMM, tokens: jnp.ndarray) -> None:
    if tokens.ndim != 1 or tokens.shape[0] < 2:
        raise ValueError(f"tokens must be a 1-D sequence of length >= 2; got {tokens.shape}")
    if bool(jnp.any((tokens < 0) | (tokens >= hmm.vocab))):
        raise ValueError(f"tokens must lie in [0, {hmm.vocab - 1}]")


def _filter_with_transition(
    hmm: TwoTimescaleHMM, tokens: jnp.ndarray, transition: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Forward filter under an arbitrary regime transition; returns (preds, posteriors)."""

    def step(p, pair):
        a, b = pair
        prior = p @ transition
        pred = prior @ hmm.bigrams[:, a, :]
        lik = hmm.bigrams[:, a, b]
        post = prior * lik
        post = post / jnp.sum(post)
        return post, (pred, post)

    p0 = jnp.full((hmm.num_regimes,), 1.0 / hmm.num_regimes)
    pairs = (tokens[:-1], tokens[1:])
    _, (preds, posts) = jax.lax.scan(step, p0, pairs)
    return preds, posts


def forward_filter_predictions(
    hmm: TwoTimescaleHMM, tokens: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""The Bayes-optimal predictor: $P(x_{t+1} \mid x_{1:t})$ via the forward filter.

    Returns
    -------
    preds : jnp.ndarray, shape (L-1, V)
        ``preds[i]`` is the predictive distribution over ``tokens[i+1]``.
    posteriors : jnp.ndarray, shape (L-1, K)
        ``posteriors[i]`` is $P(z_{i+1} \mid x_{1:i+1})$ (post-update).
    """
    _validate_tokens(hmm, tokens)
    return _filter_with_transition(hmm, tokens, hmm.transition)


def decay_filter_predictions(
    hmm: TwoTimescaleHMM, tokens: jnp.ndarray, lam: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""The fixed-decay state idealization: the filter under $M(\lambda)$.

    Identical recursion, but the regime prior is mixed toward uniform at the
    fixed rate $\lambda$ instead of propagated by the true sticky
    transition. At $\lambda = \lambda^*(\varepsilon)$ this *is* the optimal
    filter; elsewhere the slow manifold is mistimed.
    """
    _validate_tokens(hmm, tokens)
    mix = mixing_to_uniform_transition(hmm.num_regimes, lam)
    return _filter_with_transition(hmm, tokens, mix)


def regime_stationary_unigrams(hmm: TwoTimescaleHMM) -> jnp.ndarray:
    r"""Per-regime stationary unigram $\pi_j$ solving $\pi_j B_j = \pi_j$.

    Solved as a least-squares system with the normalisation constraint
    appended; residuals are validated below $10^{-10}$ (fail loud).

    Returns
    -------
    jnp.ndarray, shape (K, V)
    """
    bigrams = np.asarray(hmm.bigrams)
    k, v, _ = bigrams.shape
    out = np.zeros((k, v))
    for j in range(k):
        a = np.vstack([bigrams[j].T - np.eye(v), np.ones((1, v))])
        rhs = np.concatenate([np.zeros(v), np.ones(1)])
        pi, *_ = np.linalg.lstsq(a, rhs, rcond=None)
        residual = float(np.max(np.abs(pi @ bigrams[j] - pi)))
        if residual > 1e-10 or float(np.min(pi)) < -1e-12:
            raise ValueError(
                f"stationary solve failed for regime {j}: residual={residual:.2e}, "
                f"min={float(np.min(pi)):.2e}"
            )
        pi = np.clip(pi, 0.0, None)
        out[j] = pi / pi.sum()
    return jnp.asarray(out)


def unigram_filter_predictions(
    hmm: TwoTimescaleHMM, tokens: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""The slow-manifold-only idealization: regime tracking, unigram emissions.

    A Bayes filter for the *misspecified* model in which regime $j$ emits
    i.i.d. tokens from its stationary unigram $\pi_j$: all token-local
    (bigram) structure is compressed away, leaving only the slow signal.
    """
    _validate_tokens(hmm, tokens)
    unigrams = regime_stationary_unigrams(hmm)

    def step(p, pair):
        _, b = pair
        prior = p @ hmm.transition
        pred = prior @ unigrams
        post = prior * unigrams[:, b]
        post = post / jnp.sum(post)
        return post, (pred, post)

    p0 = jnp.full((hmm.num_regimes,), 1.0 / hmm.num_regimes)
    pairs = (tokens[:-1], tokens[1:])
    _, (preds, posts) = jax.lax.scan(step, p0, pairs)
    return preds, posts


def window_filter_predictions(
    hmm: TwoTimescaleHMM, tokens: jnp.ndarray, window: int
) -> jnp.ndarray:
    r"""The attention idealization: Bayes-exact over the last ``window`` tokens only.

    The prediction at position $i$ uses tokens $x_{i-w+1:i}$ with a
    *uniform* regime prior at the window edge — exact pairwise computation
    inside the window, zero carried state outside it. ``window >= L``
    reproduces the full filter exactly (the filter's initial posterior is
    uniform by construction).

    Implementation: a fixed-length masked scan over the $w - 1$ within-window
    bigram pairs, ``vmap``-ed over positions; invalid (pre-sequence) steps
    hold the carry fixed.

    Returns
    -------
    preds : jnp.ndarray, shape (L-1, V)
    """
    _validate_tokens(hmm, tokens)
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}")
    length = tokens.shape[0]
    k = hmm.num_regimes
    n_steps = min(window, length) - 1  # pair-updates inside one window

    def one_position(i: jnp.ndarray) -> jnp.ndarray:
        # Pairs (s, s+1) for s = i - n_steps .. i - 1; valid iff s >= 0.
        s_idx = i - n_steps + jnp.arange(n_steps)
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

        p0 = jnp.full((k,), 1.0 / k)
        p_final, _ = jax.lax.scan(step, p0, (pair_a, pair_b, valid))
        prior = p_final @ hmm.transition
        return prior @ hmm.bigrams[:, tokens[i], :]

    if n_steps == 0:
        # window == 1: every prediction conditions only on the current token.
        prior = jnp.full((k,), 1.0 / k) @ hmm.transition
        return prior @ hmm.bigrams[:, tokens[:-1], :].swapaxes(0, 1)
    return jax.vmap(one_position)(jnp.arange(length - 1))


def window_filter_predictions_naive(
    hmm: TwoTimescaleHMM, tokens: jnp.ndarray, window: int
) -> jnp.ndarray:
    r"""Python-loop oracle for :func:`window_filter_predictions` (NumPy throughout)."""
    _validate_tokens(hmm, tokens)
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}")
    transition = np.asarray(hmm.transition)
    bigrams = np.asarray(hmm.bigrams)
    toks = np.asarray(tokens)
    length = toks.shape[0]
    k = transition.shape[0]
    preds = np.zeros((length - 1, bigrams.shape[-1]))
    for i in range(length - 1):
        lo = max(0, i - window + 1)
        p = np.full(k, 1.0 / k)
        for s in range(lo, i):
            prior = p @ transition
            post = prior * bigrams[:, toks[s], toks[s + 1]]
            p = post / post.sum()
        prior = p @ transition
        preds[i] = prior @ bigrams[:, toks[i], :]
    return jnp.asarray(preds)


def enumeration_predictions(hmm: TwoTimescaleHMM, tokens: jnp.ndarray) -> jnp.ndarray:
    r"""Brute-force oracle: marginalise over **all** regime paths explicitly.

    For each position $i$, sums the joint over the $K^{i+2}$ regime paths
    $z_{0:i+1}$ — a completely independent code path (NumPy products over
    :func:`itertools.product`) for validating the forward filter on tiny
    instances. Cost is exponential; refuse sequences longer than 8.
    """
    import itertools

    _validate_tokens(hmm, tokens)
    toks = np.asarray(tokens)
    length = toks.shape[0]
    if length > 8:
        raise ValueError(f"enumeration oracle is exponential; length must be <= 8, got {length}")
    transition = np.asarray(hmm.transition)
    bigrams = np.asarray(hmm.bigrams)
    k = transition.shape[0]
    v = bigrams.shape[-1]
    preds = np.zeros((length - 1, v))
    for i in range(length - 1):
        scores = np.zeros(v)
        # Paths over z_0 .. z_{i+1}; x_0 uniform and z_0 uniform contribute
        # constant factors that cancel in the normalisation.
        for path in itertools.product(range(k), repeat=i + 2):
            weight = 1.0
            for t in range(1, i + 2):
                weight *= transition[path[t - 1], path[t]]
                if t <= i:
                    weight *= bigrams[path[t], toks[t - 1], toks[t]]
            scores += weight * bigrams[path[i + 1], toks[i], :]
        preds[i] = scores / scores.sum()
    return jnp.asarray(preds)


def mean_cross_entropy(preds: jnp.ndarray, tokens: jnp.ndarray, burn: int = 0) -> float:
    r"""Mean next-token cross-entropy $-\overline{\log P(x_{t+1})}$, nats, after burn-in.

    Parameters
    ----------
    preds : jnp.ndarray, shape (L-1, V)
    tokens : jnp.ndarray, shape (L,)
    burn : int
        Number of initial predictions to discard (all predictors are
        compared on the same positions ``burn .. L-2``).

    Returns
    -------
    float
    """
    if preds.ndim != 2 or tokens.ndim != 1 or preds.shape[0] != tokens.shape[0] - 1:
        raise ValueError(f"shape mismatch: preds {preds.shape} vs tokens {tokens.shape}")
    if not 0 <= burn < preds.shape[0]:
        raise ValueError(f"burn must be in [0, {preds.shape[0] - 1}]; got {burn}")
    targets = tokens[1:]
    picked = jnp.take_along_axis(preds, targets[:, None], axis=1)[:, 0]
    return float(jnp.mean(-jnp.log(picked[burn:])))


# ---------------------------------------------------------------------------
# Figures + measured numbers (§14.6).
# ---------------------------------------------------------------------------

_FIG_REGIMES = 4
_FIG_VOCAB = 12
_FIG_CONCENTRATION = 0.3
_FIG_OVERLAP = 0.4  # regimes hard to identify locally (tau_id >> a short window)
_FIG_LENGTH = 8192
_FIG_BURN = 128
_FIG_SEED = 0


def _figure_instance(
    eps: float, overlap: float = _FIG_OVERLAP
) -> tuple[TwoTimescaleHMM, jnp.ndarray]:
    """One seeded (hmm, tokens) pair; bigram draw shared across (eps, overlap)."""
    key = jax.random.PRNGKey(_FIG_SEED)
    key_bigrams, key_seq = jax.random.split(key)
    hmm = make_hmm(key_bigrams, _FIG_REGIMES, _FIG_VOCAB, eps, _FIG_CONCENTRATION, overlap)
    seq_key = jax.random.fold_in(key_seq, int(eps * 1e9) + int(overlap * 1e3))
    tokens, _ = sample_sequence(seq_key, hmm, _FIG_LENGTH)
    return hmm, tokens


def _excess_table(eps: float, windows: tuple[int, ...], lam_fixed: float) -> dict[str, float]:
    hmm, tokens = _figure_instance(eps)
    full, _ = forward_filter_predictions(hmm, tokens)
    ce_full = mean_cross_entropy(full, tokens, _FIG_BURN)
    out = {"full": ce_full}
    for w in windows:
        preds = window_filter_predictions(hmm, tokens, w)
        out[f"window{w}"] = mean_cross_entropy(preds, tokens, _FIG_BURN) - ce_full
    preds, _ = decay_filter_predictions(hmm, tokens, lam_fixed)
    out["decay_fixed"] = mean_cross_entropy(preds, tokens, _FIG_BURN) - ce_full
    preds, _ = unigram_filter_predictions(hmm, tokens)
    out["unigram"] = mean_cross_entropy(preds, tokens, _FIG_BURN) - ce_full
    return out


def _fig_two_timescale_error() -> None:
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
    eps_grid = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    lam_fixed = epsilon_to_lambda(0.05, _FIG_REGIMES)
    rows = [_excess_table(e, windows=(8, 64), lam_fixed=lam_fixed) for e in eps_grid]

    fig, axes = create_tufte_figure(2, 1, figsize=(6.4, 5.6), sharex=True)
    ax_state, ax_uni = axes[0], axes[1]
    series = [
        ("window8", "window $w=8$", SSM_COLORS["alert"], "-"),
        ("window64", "window $w=64$", SSM_COLORS["alert"], "--"),
        ("decay_fixed", r"decay, $\lambda$ matched at $\varepsilon=0.05$",
         SSM_COLORS["accent"], "-"),
        ("unigram", "unigram (slow-only)", SSM_COLORS["baseline"], "-"),
    ]
    print("  two-timescale-error table (excess CE in nats over the exact filter, "
          f"overlap={_FIG_OVERLAP}):")
    print(f"    {'eps':>6}  " + "  ".join(f"{k:>12}" for k, *_ in series) + "  ce_full")
    for e, row in zip(eps_grid, rows):
        print(f"    {e:>6.3f}  " + "  ".join(f"{row[k]:>12.4f}" for k, *_ in series)
              + f"  {row['full']:.4f}")
    noise_floor = max(0.0, -min(row[k] for row in rows for k, *_ in series))
    print(f"    (Monte-Carlo floor: differences below ~{max(noise_floor, 1e-4):.0e} nats "
          "are indistinguishable from 0 at this L)")
    for key_name, label, color, ls in series[:3]:
        ax_state.plot(eps_grid, [r[key_name] for r in rows], color=color, ls=ls, lw=1.8,
                      marker="o", ms=3.5, label=label)
    key_name, label, color, ls = series[3]
    ax_uni.plot(eps_grid, [r[key_name] for r in rows], color=color, ls=ls, lw=1.8,
                marker="o", ms=3.5, label=label)
    ax_state.set_xscale("log")
    set_tufte_title(ax_state, "Restricted state: window / mistimed decay (measured)")
    set_tufte_labels(ax_state, None, "excess CE (nats)")
    ax_state.legend(frameon=False, fontsize=8.5)
    set_tufte_title(ax_uni, "No fast structure: unigram emissions (note the scale)")
    set_tufte_labels(ax_uni, r"switch probability $\varepsilon$ (log)", "excess CE (nats)")
    ax_uni.set_ylim(0.0, None)
    ax_uni.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in save_figure(fig, _OUT_DIR / "two-timescale-error", formats=("png",)):
        print(f"  wrote {p.relative_to(_REPO_ROOT)}")
    plt.close(fig)


def _fig_window_crossover() -> None:
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
    windows = (2, 4, 8, 16, 32, 64, 128, 256)
    eps = 0.005
    overlaps = ((1.0, SSM_COLORS["baseline"]), (0.4, SSM_COLORS["accent"]),
                (0.2, SSM_COLORS["alert"]))

    fig, ax = create_tufte_figure(figsize=(6.4, 4.0))
    print(f"  window-crossover table (excess CE in nats, eps={eps}; "
          "overlap controls identification difficulty):")
    header = "  ".join(f"eta={o:<4}" for o, _ in overlaps)
    print(f"    {'w':>5}  {header}")
    table = {}
    for eta, color in overlaps:
        hmm, tokens = _figure_instance(eps, overlap=eta)
        disc = mean_pairwise_discrimination(hmm)
        full, _ = forward_filter_predictions(hmm, tokens)
        ce_full = mean_cross_entropy(full, tokens, _FIG_BURN)
        excesses = []
        for w in windows:
            preds = window_filter_predictions(hmm, tokens, w)
            excesses.append(mean_cross_entropy(preds, tokens, _FIG_BURN) - ce_full)
        table[eta] = excesses
        ax.plot(windows, excesses, color=color, lw=1.8, marker="o", ms=3.5,
                label=rf"$\eta = {eta}$ ($\bar\imath = {disc:.3f}$ nats/token)")
        print(f"    eta={eta}: mean pairwise discrimination = {disc:.4f} nats/token")
    for i, w in enumerate(windows):
        print(f"    {w:>5}  " + "  ".join(f"{table[o][i]:>8.4f}" for o, _ in overlaps))
    ax.set_xscale("log", base=2)
    set_tufte_title(ax, "Window needed tracks identification difficulty (measured)")
    set_tufte_labels(ax, "window $w$ (log)", "excess cross-entropy (nats)")
    ax.legend(frameon=False, fontsize=9)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in save_figure(fig, _OUT_DIR / "window-crossover", formats=("png",)):
        print(f"  wrote {p.relative_to(_REPO_ROOT)}")
    plt.close(fig)


def main() -> None:
    print("Chapter 14 — two_timescale.py")
    print("=" * 64)

    # Reference configuration for the prose numbers.
    eps = 0.02
    hmm, tokens = _figure_instance(eps)
    lam_star = epsilon_to_lambda(eps, hmm.num_regimes)

    # §14.2 the transition-family identity T(eps) == M(lambda*).
    t_true = make_transition(hmm.num_regimes, eps)
    m_star = mixing_to_uniform_transition(hmm.num_regimes, lam_star)
    print(f"  T(eps) == M(lambda*) entrywise:      max diff = "
          f"{float(jnp.max(jnp.abs(t_true - m_star))):.2e}   (lambda* = {lam_star:.6f})")

    # §14.6 filter == brute-force path enumeration (tiny instance).
    key = jax.random.PRNGKey(7)
    tiny_hmm = make_hmm(key, 2, 3, 0.15, 0.5)
    tiny_tokens, _ = sample_sequence(jax.random.fold_in(key, 1), tiny_hmm, 7)
    filt_preds, _ = forward_filter_predictions(tiny_hmm, tiny_tokens)
    enum_preds = enumeration_predictions(tiny_hmm, tiny_tokens)
    print(f"  filter == enumeration oracle (L=7):  max diff = "
          f"{float(jnp.max(jnp.abs(filt_preds - enum_preds))):.2e}")

    # §14.6 exact reductions.
    full, _ = forward_filter_predictions(hmm, tokens)
    win_all = window_filter_predictions(hmm, tokens, int(tokens.shape[0]))
    print(f"  window w=L == full filter:           max diff = "
          f"{float(jnp.max(jnp.abs(win_all - full))):.2e}")
    matched, _ = decay_filter_predictions(hmm, tokens, lam_star)
    print(f"  decay lambda=lambda* == full filter: max diff = "
          f"{float(jnp.max(jnp.abs(matched - full))):.2e}")

    # §14.6 the reference excess-CE numbers.
    disc = mean_pairwise_discrimination(hmm)
    ce_full = mean_cross_entropy(full, tokens, _FIG_BURN)
    ce_win8 = mean_cross_entropy(window_filter_predictions(hmm, tokens, 8), tokens, _FIG_BURN)
    mismatched, _ = decay_filter_predictions(hmm, tokens, epsilon_to_lambda(0.2, 4))
    ce_decay = mean_cross_entropy(mismatched, tokens, _FIG_BURN)
    uni, _ = unigram_filter_predictions(hmm, tokens)
    ce_uni = mean_cross_entropy(uni, tokens, _FIG_BURN)
    print(f"  reference config (eps={eps}, overlap={_FIG_OVERLAP}, L={_FIG_LENGTH}, "
          f"burn={_FIG_BURN}):")
    print(f"    mean pairwise discrimination = {disc:.4f} nats/token")
    print(f"    CE optimal filter        = {ce_full:.4f} nats")
    print(f"    CE window w=8            = {ce_win8:.4f}  (excess {ce_win8 - ce_full:.4f})")
    print(f"    CE decay lambda*(0.2)    = {ce_decay:.4f}  (excess {ce_decay - ce_full:.4f})")
    print(f"    CE unigram (slow-only)   = {ce_uni:.4f}  (excess {ce_uni - ce_full:.4f})")

    print("  figures:")
    _fig_two_timescale_error()
    _fig_window_crossover()


if __name__ == "__main__":
    main()
