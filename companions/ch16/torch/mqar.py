r"""Chapter 16 torch companion — the MQAR readers (scoring path).

Mirrors the reader family of ``companions/ch16/jax/mqar.py`` in eager
PyTorch: the exact-match induction reader, the additive outer-product
reader, and the fading-memory decay reader, plus the accuracy metric.
float64 throughout for exact integer-decode parity against the JAX
companion.

The task *generator*, the NumPy scan oracle, the slot reader (a pure-Python
ring buffer with no framework content), and the L90/AUC metrics stay
JAX-side; parity tests build an episode once via the JAX generator and feed
the same integer tokens to both implementations. Key embeddings are drawn
from the same NumPy ``default_rng`` stream as the JAX module, so the two
frameworks read the *identical* state.

Port credit
-----------
Mirrors the JAX module (greenfield for this chapter; task semantics follow
``zoology``'s MQAR, reference only).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "MQARInstance",
    "induction_reader",
    "outer_product_reader",
    "decay_reader",
    "accuracy",
]

torch.set_default_dtype(torch.float64)


class MQARInstance(NamedTuple):
    """One tokenized MQAR episode (see the JAX module for the layout contract)."""

    tokens: Tensor
    query_positions: Tensor
    answers: Tensor
    n_keys: int
    n_values: int
    n_distractors: int = 0

    @property
    def n_pairs(self) -> int:
        return int(self.query_positions.shape[0])

    @property
    def n_stored(self) -> int:
        return self.n_pairs + self.n_distractors

    @property
    def filler_id(self) -> int:
        return self.n_keys + self.n_values


def _key_embeddings(n_keys: int, dim: int, seed: int) -> Tensor:
    """Unit-norm key embeddings from the same NumPy stream as the JAX module."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n_keys, dim))
    return torch.from_numpy(emb / np.linalg.norm(emb, axis=1, keepdims=True))


def induction_reader(instance: MQARInstance, beta: float = 30.0) -> Tensor:
    """Exact-match attention over earlier positions; reads the successor token."""
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0; got {beta}")
    toks = instance.tokens
    n_vals = instance.n_values
    length = toks.shape[0]
    successor = torch.cat([toks[1:], torch.tensor([instance.filler_id], dtype=toks.dtype)])
    is_value = (successor >= instance.n_keys) & (successor < instance.n_keys + n_vals)
    succ_value = torch.zeros(length, n_vals)
    idx = torch.clamp(successor - instance.n_keys, 0, n_vals - 1)
    succ_value[torch.arange(length), idx] = 1.0
    succ_value = torch.where(is_value[:, None], succ_value, torch.zeros(()))
    out = torch.zeros(instance.n_pairs, dtype=toks.dtype)
    positions = torch.arange(length)
    for i, p in enumerate(instance.query_positions.tolist()):
        scores = beta * (toks == toks[p]).double()
        scores = torch.where(positions < p, scores, torch.tensor(-torch.inf))
        weights = torch.softmax(scores, dim=0)
        value_mass = weights @ succ_value
        out[i] = instance.n_keys + int(torch.argmax(value_mass))
    return out


def outer_product_reader(instance: MQARInstance, dim: int, seed: int = 0) -> Tensor:
    """The additive-state reader on token embeddings (generic unit keys)."""
    if dim < 1:
        raise ValueError(f"dim must be >= 1; got {dim}")
    emb = _key_embeddings(instance.n_keys, dim, seed)
    toks = instance.tokens
    n = instance.n_stored
    stored_keys = toks[0 : 2 * n : 2]
    stored_vals = toks[1 : 2 * n : 2] - instance.n_keys
    val_onehot = torch.zeros(n, instance.n_values)
    val_onehot[torch.arange(n), stored_vals] = 1.0
    state = emb[stored_keys].T @ val_onehot
    read = emb[toks[instance.query_positions]] @ state
    return instance.n_keys + torch.argmax(read, dim=1).to(toks.dtype)


def decay_reader(instance: MQARInstance, dim: int, rho: float, seed: int = 0) -> Tensor:
    """The fading-memory reader: per-pair closed-form decay weights, argmax decode."""
    if dim < 1:
        raise ValueError(f"dim must be >= 1; got {dim}")
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must be in (0, 1]; got {rho}")
    emb = _key_embeddings(instance.n_keys, dim, seed)
    toks = instance.tokens
    n = instance.n_stored
    stored_keys = toks[0 : 2 * n : 2]
    stored_vals = toks[1 : 2 * n : 2] - instance.n_keys
    write_pos = 1 + 2 * torch.arange(n)
    val_onehot = torch.zeros(n, instance.n_values)
    val_onehot[torch.arange(n), stored_vals] = 1.0
    out = torch.zeros(instance.n_pairs, dtype=toks.dtype)
    for i, p in enumerate(instance.query_positions.tolist()):
        weights = torch.pow(torch.tensor(rho), (p - 1 - write_pos).double())
        state = (emb[stored_keys] * weights[:, None]).T @ val_onehot
        read = emb[toks[p]] @ state
        out[i] = instance.n_keys + int(torch.argmax(read))
    return out


def accuracy(predictions: Tensor, answers: Tensor) -> float:
    """Fraction of queries answered exactly."""
    if predictions.shape != answers.shape:
        raise ValueError(f"shape mismatch: {predictions.shape} vs {answers.shape}")
    return float((predictions == answers).double().mean())
