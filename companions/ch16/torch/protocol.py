r"""Chapter 16 torch companion — the composite filter and the ridge probe.

Mirrors the protocol-specific compute paths of
``companions/ch16/jax/protocol.py`` in eager PyTorch:

* :func:`composite_filter_predictions` — the window-$w$ filter seeded at its
  edge with the $\lambda$-decayed carried prior (or the uniform restart when
  ``lam=None``), written as a per-position Python loop over explicit Bayes
  updates — the readable eager mirror of the JAX masked-scan version;
* :func:`ridge_probe_accuracy` — the closed-form held-out linear probe
  (first-half fit / second-half score), via ``torch.linalg.solve``.

The HMM filters it composes with live in
``companions/ch14/torch/two_timescale.py`` (imported, mirroring the JAX-side
dependency on ch14); the comparison statistics are framework-free NumPy in
the JAX module and are not duplicated here.

Port credit
-----------
Mirrors the JAX module (greenfield for this chapter).
"""

from __future__ import annotations

import torch
from torch import Tensor

from companions.ch14.torch.two_timescale import (
    TwoTimescaleHMM,
    decay_filter_predictions,
)

__all__ = [
    "composite_filter_predictions",
    "ridge_probe_accuracy",
]

torch.set_default_dtype(torch.float64)


def composite_filter_predictions(
    hmm: TwoTimescaleHMM, tokens: Tensor, window: int, lam: float | None
) -> tuple[Tensor, Tensor]:
    r"""Window-$w$ filter with a $\lambda$-decayed carried prior at the window edge.

    Same contract as the JAX version: returns ``(preds, priors)`` with
    ``preds[i]`` the predictive distribution over ``tokens[i+1]`` and
    ``priors[i]`` the regime prior the prediction used. ``lam=None`` is the
    uniform restart (ch14's window filter); ``lam = lambda*(eps)`` makes the
    composite the full Bayes filter exactly.
    """
    if tokens.ndim != 1 or tokens.shape[0] < 2:
        raise ValueError(f"tokens must be a 1-D sequence of length >= 2; got {tuple(tokens.shape)}")
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}")
    length = int(tokens.shape[0])
    k = hmm.num_regimes
    uniform = torch.full((k,), 1.0 / k)
    if lam is None:
        edge_priors = uniform.expand(length, k)
    else:
        _, decay_posts = decay_filter_predictions(hmm, tokens, lam)
        edge_priors = torch.vstack([uniform[None], decay_posts])
    toks = tokens.tolist()
    preds = torch.zeros(length - 1, hmm.vocab)
    priors = torch.zeros(length - 1, k)
    for i in range(length - 1):
        edge = i - window + 1
        p = uniform if edge <= 0 else edge_priors[edge]
        for s in range(max(0, edge), i):
            prior = p @ hmm.transition
            post = prior * hmm.bigrams[:, toks[s], toks[s + 1]]
            p = post / post.sum()
        prior = p @ hmm.transition
        preds[i] = prior @ hmm.bigrams[:, toks[i], :]
        priors[i] = prior
    return preds, priors


def ridge_probe_accuracy(
    features: Tensor, labels: Tensor, num_classes: int, alpha: float = 1e-6
) -> float:
    """Closed-form ridge to one-hot labels; argmax accuracy on the second half."""
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"need (n, d) features and (n,) labels; got {tuple(features.shape)}, "
            f"{tuple(labels.shape)}"
        )
    if features.shape[0] < 4:
        raise ValueError("need at least 4 positions to split")
    if alpha <= 0.0:
        raise ValueError(f"alpha must be > 0; got {alpha}")
    if bool(((labels < 0) | (labels >= num_classes)).any()):
        raise ValueError(f"labels must lie in [0, {num_classes - 1}]")
    n = features.shape[0]
    half = n // 2
    x1 = torch.hstack([features.double(), torch.ones(n, 1)])
    onehot = torch.eye(num_classes)[labels[:half]]
    gram = x1[:half].T @ x1[:half] + alpha * torch.eye(x1.shape[1])
    w = torch.linalg.solve(gram, x1[:half].T @ onehot)
    pred = torch.argmax(x1[half:] @ w, dim=1)
    return float((pred == labels[half:]).double().mean())
