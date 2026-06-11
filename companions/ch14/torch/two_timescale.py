r"""Chapter 14 torch companion — the two-timescale filters.

Mirrors the predictor family of ``companions/ch14/jax/two_timescale.py`` in
eager PyTorch: the exact forward filter, the fixed-decay (mixing-to-uniform)
filter, and the windowed (uniform-restart) filter. float64 throughout for
``< 1e-9`` parity against the JAX companion.

Sequence *generation* is deliberately not mirrored: PRNG streams do not
match across frameworks, so parity tests draw tokens once (NumPy) and feed
the same integers to both implementations. The brute-force enumeration
oracle and the stationary-unigram predictor also stay JAX-side.

Port credit
-----------
Mirrors the JAX module (greenfield for this chapter, from the pilot-B
kickoff task spec).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

__all__ = [
    "TwoTimescaleHMM",
    "make_transition",
    "mixing_to_uniform_transition",
    "epsilon_to_lambda",
    "forward_filter_predictions",
    "decay_filter_predictions",
    "window_filter_predictions",
    "mean_cross_entropy",
]

torch.set_default_dtype(torch.float64)


class TwoTimescaleHMM(NamedTuple):
    """Model parameters: sticky transition (K, K) + per-regime bigrams (K, V, V)."""

    transition: Tensor
    bigrams: Tensor

    @property
    def num_regimes(self) -> int:
        return self.transition.shape[0]

    @property
    def vocab(self) -> int:
        return self.bigrams.shape[-1]


def make_transition(num_regimes: int, eps: float) -> Tensor:
    r"""Sticky transition $T(\varepsilon) = (1-\varepsilon)I + \tfrac{\varepsilon}{K-1}(J-I)$."""
    if num_regimes < 2:
        raise ValueError(f"num_regimes must be >= 2; got {num_regimes}")
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"eps must be in [0, 1]; got {eps}")
    k = num_regimes
    eye = torch.eye(k)
    return (1.0 - eps) * eye + (eps / (k - 1)) * (torch.ones(k, k) - eye)


def mixing_to_uniform_transition(num_regimes: int, lam: float) -> Tensor:
    r"""Fixed-decay transition $M(\lambda) = (1-\lambda)I + \tfrac{\lambda}{K}J$."""
    if num_regimes < 2:
        raise ValueError(f"num_regimes must be >= 2; got {num_regimes}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1]; got {lam}")
    k = num_regimes
    return (1.0 - lam) * torch.eye(k) + (lam / k) * torch.ones(k, k)


def epsilon_to_lambda(eps: float, num_regimes: int) -> float:
    r"""The matching rate $\lambda^* = \varepsilon K / (K - 1)$."""
    if num_regimes < 2:
        raise ValueError(f"num_regimes must be >= 2; got {num_regimes}")
    lam = eps * num_regimes / (num_regimes - 1)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(
            f"eps={eps} gives lambda*={lam:.4f} outside [0, 1] for K={num_regimes}"
        )
    return lam


def _validate_tokens(hmm: TwoTimescaleHMM, tokens: Tensor) -> None:
    if tokens.ndim != 1 or tokens.shape[0] < 2:
        raise ValueError(f"tokens must be a 1-D sequence of length >= 2; got {tuple(tokens.shape)}")
    if bool(torch.any((tokens < 0) | (tokens >= hmm.vocab))):
        raise ValueError(f"tokens must lie in [0, {hmm.vocab - 1}]")


def _filter_with_transition(
    hmm: TwoTimescaleHMM, tokens: Tensor, transition: Tensor
) -> tuple[Tensor, Tensor]:
    length = tokens.shape[0]
    k = hmm.num_regimes
    p = torch.full((k,), 1.0 / k)
    preds, posts = [], []
    for t in range(length - 1):
        a, b = int(tokens[t]), int(tokens[t + 1])
        prior = p @ transition
        preds.append(prior @ hmm.bigrams[:, a, :])
        post = prior * hmm.bigrams[:, a, b]
        p = post / post.sum()
        posts.append(p)
    return torch.stack(preds), torch.stack(posts)


def forward_filter_predictions(hmm: TwoTimescaleHMM, tokens: Tensor) -> tuple[Tensor, Tensor]:
    r"""The Bayes-optimal predictor $P(x_{t+1} \mid x_{1:t})$ (eager forward filter)."""
    _validate_tokens(hmm, tokens)
    return _filter_with_transition(hmm, tokens, hmm.transition)


def decay_filter_predictions(
    hmm: TwoTimescaleHMM, tokens: Tensor, lam: float
) -> tuple[Tensor, Tensor]:
    r"""The fixed-decay state idealization: the filter under $M(\lambda)$."""
    _validate_tokens(hmm, tokens)
    mix = mixing_to_uniform_transition(hmm.num_regimes, lam)
    return _filter_with_transition(hmm, tokens, mix)


def window_filter_predictions(hmm: TwoTimescaleHMM, tokens: Tensor, window: int) -> Tensor:
    r"""The attention idealization: Bayes-exact over the last ``window`` tokens only."""
    _validate_tokens(hmm, tokens)
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}")
    length = tokens.shape[0]
    k = hmm.num_regimes
    preds = []
    for i in range(length - 1):
        lo = max(0, i - window + 1)
        p = torch.full((k,), 1.0 / k)
        for s in range(lo, i):
            prior = p @ hmm.transition
            post = prior * hmm.bigrams[:, int(tokens[s]), int(tokens[s + 1])]
            p = post / post.sum()
        prior = p @ hmm.transition
        preds.append(prior @ hmm.bigrams[:, int(tokens[i]), :])
    return torch.stack(preds)


def mean_cross_entropy(preds: Tensor, tokens: Tensor, burn: int = 0) -> float:
    r"""Mean next-token cross-entropy in nats after burn-in (same contract as JAX)."""
    if preds.ndim != 2 or tokens.ndim != 1 or preds.shape[0] != tokens.shape[0] - 1:
        raise ValueError(f"shape mismatch: preds {tuple(preds.shape)} vs tokens {tuple(tokens.shape)}")
    if not 0 <= burn < preds.shape[0]:
        raise ValueError(f"burn must be in [0, {preds.shape[0] - 1}]; got {burn}")
    targets = tokens[1:]
    picked = preds.gather(1, targets[:, None].long())[:, 0]
    return float(torch.mean(-torch.log(picked[burn:])))
