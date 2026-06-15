r"""Chapter 16 §§16.2 & 16.4 — tokenized MQAR as a protocol object.

Chapter 11 §11.6 showed the *mechanism* behind the associative-recall capacity
wall: a fixed additive state $S = \sum_i \phi(k_i) v_i^\top$ retrieves with
interference error growing like $\sqrt{K/d_k}$
(``companions/ch11/jax/mqar_recall.py``). This module builds the *benchmark*
view of the same task — multi-query associative recall (MQAR; Arora et al.
2024, arXiv:2312.04927) as **tokenized sequences with an accuracy metric** —
because Chapter 16's subject is the protocol, not the mechanism:

* a generator emitting ``[k_1, v_1, ..., k_N, v_N, <gap fillers>, q_1..q_N]``
  with disjoint key/value/filler alphabets, distinct keys, and each stored key
  queried exactly once (the layout that makes the capacity identity *exact*);
* an **oracle** (independent code path: a NumPy scan for the unique earlier
  occurrence of the queried key) that defines ground truth;
* idealized readers, each an information restriction rather than a trained
  model (the ch14 §14.6 discipline):

  - :func:`induction_reader` — exact-match attention that copies the token
    after the matching key: the induction-head circuit (Olsson et al. 2022),
    correct at every load;
  - :func:`outer_product_reader` — the additive-state reader on token
    embeddings (the ch11 mechanism, tokenized): exact below capacity with
    orthonormal key embeddings, interference-limited above it;
  - :func:`slot_reader` — the **exact-capacity idealization**: a ring buffer
    holding the last $d$ pairs, answering correctly iff the queried key is
    still stored, abstaining otherwise. Its accuracy is *exactly*
    $\min(1, d/N)$, which is what makes the discriminative-regime
    proposition checkable at ``rtol=0``;
  - :func:`decay_reader` — the fading-memory reader: the additive state
    decayed by $\rho$ per token. Padding the key-to-query separation with
    *neutral fillers* rescales every stored weight uniformly and the argmax
    read-out is provably unchanged (the negative control, pinned exactly);
    padding with *distractor pairs* — fresher writes the decayed target must
    compete against — degrades retrieval, the structural miniature of
    RULER-style multi-key length stress (Hsieh et al. 2024).

* the §16.4 **length-robustness metrics**: :func:`l90` (longest separation
  retaining 90% of short-range accuracy) and :func:`auc_log2` (mean accuracy
  over log-spaced separations), computed on the decay reader's measured curve.

The §16.2 design payload is the discriminative-regime proposition: two state
sizes $d_1 < d_2$ produce an accuracy gap of exactly $0$ for $N \le d_1$,
exactly $1 - d_1/d_2$ at $N = d_2$ (the maximum), and $(d_2 - d_1)/N \to 0$
beyond — a benchmark sized outside $(d_1, \infty)$'s knee region measures
nothing about the comparison. This generalizes the ch14 §14.6 lesson
($w \ll \tau_{\mathrm{id}} \ll 1/\varepsilon$) from timescales to capacity.

No training anywhere: every reader is an analytic restriction, every number
is exact or seeded-deterministic.

Idiomatic-JAX / port credit
---------------------------
Greenfield. Task semantics mirror ``zoology``'s MQAR generator (reference
only; no code ported). The additive/decayed readers reuse the ch11
``mqar_recall.py`` mechanism shape on token embeddings; the decayed state is
computed in closed form (per-pair weights $\rho^{\Delta_i}$) rather than by a
per-step loop — the loop version is the oracle.

Usage
-----
::

    PYTHONPATH=. python companions/ch16/jax/mqar.py
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax

# Enable float64 before any jnp array is created (matches Chapters 4, 7-14).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

__all__ = [
    "MQARInstance",
    "make_mqar",
    "oracle_recall",
    "induction_reader",
    "outer_product_reader",
    "slot_reader",
    "slot_accuracy_exact",
    "decay_reader",
    "decay_reader_naive",
    "accuracy",
    "l90",
    "auc_log2",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch16"


class MQARInstance(NamedTuple):
    """One tokenized MQAR episode.

    ``tokens`` is ``[k_1, v_1, ..., k_N, v_N, <distractor pairs>, <gap
    fillers>, q_1, ..., q_N]``: the *target* pairs, then ``n_distractors``
    extra pairs whose keys are never queried (the RULER-style multi-key
    stressor), then ``gap`` neutral fillers, then one query per target key.
    Token ids partition the vocabulary into keys ``0..n_keys-1``, values
    ``n_keys..n_keys+n_values-1``, and a single filler id
    ``n_keys + n_values`` (used for the gap and as the abstain symbol).
    """

    tokens: jnp.ndarray
    query_positions: jnp.ndarray
    answers: jnp.ndarray
    n_keys: int
    n_values: int
    n_distractors: int = 0

    @property
    def n_pairs(self) -> int:
        return int(self.query_positions.shape[0])

    @property
    def n_stored(self) -> int:
        """Total stored pairs: targets + distractors."""
        return self.n_pairs + self.n_distractors

    @property
    def filler_id(self) -> int:
        return self.n_keys + self.n_values

    @property
    def vocab(self) -> int:
        return self.n_keys + self.n_values + 1


# ---------------------------------------------------------------------------
# Generator + oracle.
# ---------------------------------------------------------------------------


def make_mqar(
    key: jax.Array,
    n_pairs: int,
    n_keys: int,
    n_values: int,
    gap: int = 0,
    n_distractors: int = 0,
) -> MQARInstance:
    """Draw one MQAR episode: distinct keys, each *target* key queried exactly once.

    All stored keys (targets + distractors) are sampled without replacement
    from the key alphabet, values with replacement; queries are a seeded
    permutation of the target keys. Two separation dials, deliberately
    distinct: ``n_distractors`` inserts *later-written pairs* between targets
    and queries (the RULER-style stressor — fresher writes that a fading
    memory must compete against), while ``gap`` inserts *neutral fillers*
    (the negative control: a uniform rescaling of every stored weight, which
    an argmax read-out provably ignores).

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    n_pairs : int
        Number of target pairs ``N >= 1``.
    n_keys, n_values : int
        Alphabet sizes (both ``>= 2``); requires
        ``n_pairs + n_distractors <= n_keys`` for distinct keys.
    gap : int
        Number of neutral filler tokens before the query region.
    n_distractors : int
        Number of never-queried pairs written after the targets.

    Returns
    -------
    MQARInstance
    """
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1; got {n_pairs}")
    if n_keys < 2 or n_values < 2:
        raise ValueError(f"alphabets need n_keys, n_values >= 2; got {n_keys}, {n_values}")
    if n_distractors < 0:
        raise ValueError(f"n_distractors must be >= 0; got {n_distractors}")
    if n_pairs + n_distractors > n_keys:
        raise ValueError(
            f"distinct keys need n_pairs + n_distractors <= n_keys; "
            f"got {n_pairs} + {n_distractors} > {n_keys}"
        )
    if gap < 0:
        raise ValueError(f"gap must be >= 0; got {gap}")
    key_k, key_v, key_q = jax.random.split(key, 3)
    all_keys = jax.random.choice(key_k, n_keys, shape=(n_pairs + n_distractors,), replace=False)
    all_values = n_keys + jax.random.randint(key_v, (n_pairs + n_distractors,), 0, n_values)
    target_keys, target_values = all_keys[:n_pairs], all_values[:n_pairs]
    perm = jax.random.permutation(key_q, n_pairs)
    filler = jnp.full((gap,), n_keys + n_values, dtype=all_keys.dtype)
    pairs = jnp.stack([all_keys, all_values], axis=1).reshape(-1)
    tokens = jnp.concatenate([pairs, filler, target_keys[perm]])
    query_positions = 2 * (n_pairs + n_distractors) + gap + jnp.arange(n_pairs)
    return MQARInstance(
        tokens=tokens,
        query_positions=query_positions,
        answers=target_values[perm],
        n_keys=n_keys,
        n_values=n_values,
        n_distractors=n_distractors,
    )


def oracle_recall(instance: MQARInstance) -> jnp.ndarray:
    """Ground-truth reader: scan for the unique earlier occurrence of the queried key.

    Independent code path (pure NumPy scan over raw tokens; never touches
    ``instance.answers``): for query token ``q`` at position ``p``, find the
    unique ``j < p`` with ``tokens[j] == q`` and return ``tokens[j + 1]``.
    Uniqueness holds by construction (distinct keys, disjoint alphabets,
    each key queried once); the scan *validates* it and fails loud otherwise.
    """
    toks = np.asarray(instance.tokens)
    out = np.zeros(instance.n_pairs, dtype=toks.dtype)
    for i, p in enumerate(np.asarray(instance.query_positions)):
        matches = np.flatnonzero(toks[:p] == toks[p])
        if matches.shape[0] != 1:
            raise ValueError(
                f"query at position {p} has {matches.shape[0]} earlier matches; expected 1"
            )
        out[i] = toks[matches[0] + 1]
    return jnp.asarray(out)


# ---------------------------------------------------------------------------
# Idealized readers (information restrictions, not trained models).
# ---------------------------------------------------------------------------


def induction_reader(instance: MQARInstance, beta: float = 30.0) -> jnp.ndarray:
    r"""Exact-match attention: attend to earlier copies of the query, read the next token.

    The induction-head circuit (Olsson et al. 2022, arXiv:2209.11895) as an
    analytic reader: from query position $p$, attention weights over $j < p$
    are $\mathrm{softmax}_j(\beta\,[x_j = x_p])$ — one-hot token embeddings
    make the score an exact match indicator — and the read-out aggregates the
    *successor* tokens $x_{j+1}$, decoded by argmax over value ids. With the
    unique-match layout the matched weight is $e^\beta / (e^\beta + p - 1)$,
    so any $\beta \gtrsim 20$ decodes exactly; correctness is independent of
    the load $N$ — the capacity-unbounded contrast to :func:`slot_reader`.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0; got {beta}")
    toks = instance.tokens
    n_vals = instance.n_values
    successor = jnp.concatenate([toks[1:], jnp.array([instance.filler_id], dtype=toks.dtype)])
    # successor one-hot mass restricted to value ids; filler/key successors map to zero rows.
    succ_value = jax.nn.one_hot(successor - instance.n_keys, n_vals, dtype=jnp.float64)
    succ_value = jnp.where(
        ((successor >= instance.n_keys) & (successor < instance.n_keys + n_vals))[:, None],
        succ_value,
        0.0,
    )

    def one_query(p: jnp.ndarray) -> jnp.ndarray:
        scores = beta * (toks == toks[p]).astype(jnp.float64)
        scores = jnp.where(jnp.arange(toks.shape[0]) < p, scores, -jnp.inf)
        weights = jax.nn.softmax(scores)
        value_mass = weights @ succ_value  # (n_values,)
        return instance.n_keys + jnp.argmax(value_mass)

    return jax.vmap(one_query)(instance.query_positions)


def _key_embeddings(n_keys: int, dim: int, seed: int, orthonormal: bool) -> jnp.ndarray:
    """Unit-norm key embeddings (n_keys, dim); orthonormal rows need n_keys <= dim."""
    rng = np.random.default_rng(seed)
    if orthonormal:
        if n_keys > dim:
            raise ValueError(f"orthonormal embeddings need n_keys <= dim; got {n_keys} > {dim}")
        q, _ = np.linalg.qr(rng.standard_normal((dim, n_keys)))
        return jnp.asarray(q.T)
    emb = rng.standard_normal((n_keys, dim))
    return jnp.asarray(emb / np.linalg.norm(emb, axis=1, keepdims=True))


def outer_product_reader(
    instance: MQARInstance, dim: int, seed: int = 0, orthonormal: bool = False
) -> jnp.ndarray:
    r"""The additive-state reader: $S = \sum_i \phi(k_i) e_{v_i}^\top$, read $S^\top \phi(q)$.

    The ch11 §11.6 mechanism on tokens: keys embed as unit vectors
    $\phi(k) \in \mathbb{R}^{\text{dim}}$, values as one-hot indicators, and
    the reader decodes $\operatorname{argmax}$ over the value coordinates of
    $S^\top \phi(q) = e_{v_q} + \sum_{i \ne q} \langle \phi(k_q), \phi(k_i)
    \rangle e_{v_i}$. With orthonormal key embeddings the interference term
    vanishes (exact recall — possible only while the *alphabet* fits,
    ``n_keys <= dim``); with generic unit embeddings the crosstalk grows with
    the load and decoding degrades past $N \approx \text{dim}$
    (Proposition ``ch11:linattn-capacity``).
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1; got {dim}")
    emb = _key_embeddings(instance.n_keys, dim, seed, orthonormal)
    toks = instance.tokens
    n = instance.n_stored
    stored_keys = toks[0 : 2 * n : 2]
    stored_vals = toks[1 : 2 * n : 2] - instance.n_keys
    state = emb[stored_keys].T @ jax.nn.one_hot(stored_vals, instance.n_values, dtype=jnp.float64)
    read = emb[toks[instance.query_positions]] @ state  # (n_queries, n_values)
    return instance.n_keys + jnp.argmax(read, axis=1)


def slot_reader(instance: MQARInstance, n_slots: int) -> jnp.ndarray:
    """The exact-capacity idealization: keep the last ``n_slots`` pairs, else abstain.

    A ring buffer over arrival order — the fading-state caricature with a
    hard edge. A query answers correctly iff its key is among the last
    ``min(N, n_slots)`` stored pairs; an evicted key returns the filler id
    (the abstain symbol, never a valid value, so accidental correctness is
    impossible). Plain NumPy loop on purpose: this is the proposition's
    object, kept readable.
    """
    if n_slots < 1:
        raise ValueError(f"n_slots must be >= 1; got {n_slots}")
    toks = np.asarray(instance.tokens)
    n = instance.n_stored
    buffer: dict[int, int] = {}
    order: list[int] = []
    for i in range(n):
        k, v = int(toks[2 * i]), int(toks[2 * i + 1])
        buffer[k] = v
        order.append(k)
        if len(order) > n_slots:
            del buffer[order.pop(0)]
    out = np.array(
        [buffer.get(int(toks[p]), instance.filler_id) for p in np.asarray(instance.query_positions)]
    )
    return jnp.asarray(out)


def slot_accuracy_exact(n_pairs: int, n_slots: int) -> float:
    r"""The closed form behind :func:`slot_reader`: accuracy $= \min(1, d/N)$ exactly.

    With distinct keys and each stored key queried exactly once, exactly
    $\min(N, d)$ of the $N$ queried keys remain in a last-$d$ buffer.
    """
    if n_pairs < 1 or n_slots < 1:
        raise ValueError(f"n_pairs and n_slots must be >= 1; got {n_pairs}, {n_slots}")
    return min(1.0, n_slots / n_pairs)


def decay_reader(instance: MQARInstance, dim: int, rho: float, seed: int = 0) -> jnp.ndarray:
    r"""The fading-memory reader: the additive state decayed by $\rho$ per token.

    The state a fixed-decay recurrence carries: writing pair $i$ at its value
    position $t_i$ and decaying every step, the state read at query position
    $p$ is $S_p = \sum_i \rho^{\,p - 1 - t_i} \phi(k_i) e_{v_i}^\top$ —
    computed here in closed form via the per-pair exponents (no loop), with
    :func:`decay_reader_naive` as the per-step oracle.

    Two separations behave *differently*, and the difference is the §16.4
    design lesson. A neutral-filler gap multiplies every stored weight by the
    same $\rho^g$, and the argmax read-out is scale-invariant — predictions
    are *exactly* unchanged (the negative control, pinned in the tests). What
    stresses a fading memory is **fresher competing writes**: $s$ distractor
    pairs between target and query leave the target's signal at
    $\rho^{2s + O(1)}$ against interference from later pairs decayed far
    less, so retrieval degrades as $s$ grows — the structural miniature of
    RULER-style multi-key length stress that the L90/AUC metrics quantify.
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1; got {dim}")
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must be in (0, 1]; got {rho}")
    emb = _key_embeddings(instance.n_keys, dim, seed, orthonormal=False)
    toks = instance.tokens
    n = instance.n_stored
    stored_keys = toks[0 : 2 * n : 2]
    stored_vals = toks[1 : 2 * n : 2] - instance.n_keys
    write_pos = 1 + 2 * jnp.arange(n)  # pair i is complete at its value position
    val_onehot = jax.nn.one_hot(stored_vals, instance.n_values, dtype=jnp.float64)

    def one_query(p: jnp.ndarray) -> jnp.ndarray:
        weights = rho ** (p - 1 - write_pos).astype(jnp.float64)
        state = (emb[stored_keys] * weights[:, None]).T @ val_onehot
        read = emb[toks[p]] @ state
        return instance.n_keys + jnp.argmax(read)

    return jax.vmap(one_query)(instance.query_positions)


def decay_reader_naive(
    instance: MQARInstance, dim: int, rho: float, seed: int = 0
) -> jnp.ndarray:
    """Per-step loop oracle for :func:`decay_reader` (NumPy throughout)."""
    if dim < 1:
        raise ValueError(f"dim must be >= 1; got {dim}")
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must be in (0, 1]; got {rho}")
    emb = np.asarray(_key_embeddings(instance.n_keys, dim, seed, orthonormal=False))
    toks = np.asarray(instance.tokens)
    n = instance.n_stored
    query_set = {int(p) for p in np.asarray(instance.query_positions)}
    state = np.zeros((dim, instance.n_values))
    out = []
    for t in range(toks.shape[0]):
        if t > 0:
            state = rho * state
        if t in query_set:
            read = emb[toks[t]] @ state
            out.append(instance.n_keys + int(np.argmax(read)))
        if t % 2 == 1 and t < 2 * n:  # a pair completes at its value position
            onehot = np.zeros(instance.n_values)
            onehot[toks[t] - instance.n_keys] = 1.0
            state = state + np.outer(emb[toks[t - 1]], onehot)
    return jnp.asarray(np.array(out))


def accuracy(predictions: jnp.ndarray, answers: jnp.ndarray) -> float:
    """Fraction of queries answered exactly (float64 mean, so small ratios stay exact)."""
    if predictions.shape != answers.shape:
        raise ValueError(f"shape mismatch: {predictions.shape} vs {answers.shape}")
    return float(jnp.mean((predictions == answers).astype(jnp.float64)))


# ---------------------------------------------------------------------------
# Length-robustness metrics (§16.4; survey-style L90 + AUC over log separation).
# ---------------------------------------------------------------------------


def _validate_curve(separations: np.ndarray, accuracies: np.ndarray) -> None:
    if separations.ndim != 1 or separations.shape != accuracies.shape:
        raise ValueError(
            f"separations and accuracies must be matching 1-D arrays; "
            f"got {separations.shape}, {accuracies.shape}"
        )
    if separations.shape[0] < 2:
        raise ValueError("need at least 2 points")
    if np.any(separations <= 0):
        raise ValueError("separations must be positive (use separation = gap + 1)")
    if np.any(np.diff(separations) <= 0):
        raise ValueError("separations must be strictly increasing")
    if np.any((accuracies < 0) | (accuracies > 1)):
        raise ValueError("accuracies must lie in [0, 1]")


def l90(separations: np.ndarray, accuracies: np.ndarray, threshold: float = 0.9) -> float:
    r"""Longest separation retaining ``threshold`` of the shortest-separation accuracy.

    $L_{90} = \max\{s_i : \mathrm{acc}(s_i) \ge \theta \cdot
    \mathrm{acc}(s_{\min})\}$ on the measured grid (well-defined: the
    shortest separation always qualifies). The survey-style single-number
    summary of *where* a length-stress curve breaks.
    """
    separations = np.asarray(separations, dtype=float)
    accuracies = np.asarray(accuracies, dtype=float)
    _validate_curve(separations, accuracies)
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1]; got {threshold}")
    qualifying = separations[accuracies >= threshold * accuracies[0]]
    return float(qualifying.max())


def auc_log2(separations: np.ndarray, accuracies: np.ndarray) -> float:
    r"""Mean accuracy over $\log_2$ separation (trapezoid), normalized to $[0, 1]$.

    $\mathrm{AUC} = \int \mathrm{acc}\, d(\log_2 s) \,/\, (\log_2 s_{\max} -
    \log_2 s_{\min})$ — the survey-style aggregate that rewards holding
    accuracy across *scales* of separation rather than across raw tokens.
    """
    separations = np.asarray(separations, dtype=float)
    accuracies = np.asarray(accuracies, dtype=float)
    _validate_curve(separations, accuracies)
    x = np.log2(separations)
    return float(np.trapezoid(accuracies, x) / (x[-1] - x[0]))


# ---------------------------------------------------------------------------
# Figures + measured numbers (§§16.2, 16.4).
# ---------------------------------------------------------------------------

_FIG_N_KEYS = 2048
_FIG_N_VALUES = 64
_FIG_SEED = 0
_FIG_D1 = 16
_FIG_D2 = 64
_FIG_LOADS = (4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512)
_FIG_SEPARATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)  # distractor pairs
_FIG_RHOS = (0.97, 0.99, 0.999, 1.0)
_FIG_DECAY_PAIRS = 8
_FIG_DECAY_DIM = 64
_FIG_N_SEEDS = 8


def _instance(
    n_pairs: int, gap: int = 0, n_distractors: int = 0, seed_offset: int = 0
) -> MQARInstance:
    key = jax.random.fold_in(
        jax.random.PRNGKey(_FIG_SEED), 1000 * n_pairs + gap + 31 * n_distractors + seed_offset
    )
    return make_mqar(key, n_pairs, _FIG_N_KEYS, _FIG_N_VALUES, gap=gap, n_distractors=n_distractors)


def _mean_reader_accuracy(reader, loads: tuple[int, ...], n_seeds: int = _FIG_N_SEEDS) -> list[float]:
    out = []
    for n in loads:
        accs = []
        for s in range(n_seeds):
            inst = _instance(n, seed_offset=7919 * s)
            accs.append(accuracy(reader(inst), inst.answers))
        out.append(float(np.mean(accs)))
    return out


def _fig_discriminative_regime() -> None:
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
    loads = np.asarray(_FIG_LOADS)
    slot_small = np.asarray([slot_accuracy_exact(int(n), _FIG_D1) for n in loads])
    slot_big = np.asarray([slot_accuracy_exact(int(n), _FIG_D2) for n in loads])
    outer_small = np.asarray(
        _mean_reader_accuracy(lambda i: outer_product_reader(i, _FIG_D1), tuple(_FIG_LOADS))
    )
    outer_big = np.asarray(
        _mean_reader_accuracy(lambda i: outer_product_reader(i, _FIG_D2), tuple(_FIG_LOADS))
    )
    gap_hard = slot_big - slot_small
    gap_soft = outer_big - outer_small

    print("  discriminative-regime table (slot accuracies exact; outer-product mean of "
          f"{_FIG_N_SEEDS} seeds):")
    print(f"    {'N':>4}  {'slot16':>7}  {'slot64':>7}  {'gap':>7}  {'outer16':>8}  "
          f"{'outer64':>8}  {'gap':>7}")
    for i, n in enumerate(loads):
        print(f"    {n:>4}  {slot_small[i]:>7.4f}  {slot_big[i]:>7.4f}  {gap_hard[i]:>7.4f}  "
              f"{outer_small[i]:>8.4f}  {outer_big[i]:>8.4f}  {gap_soft[i]:>7.4f}")
    print(f"    hard gap peak: {float(gap_hard.max()):.4f} at N={int(loads[gap_hard.argmax()])} "
          f"(exact 1 - d1/d2 = {1 - _FIG_D1 / _FIG_D2:.4f}); "
          f"gap at N={int(loads[-1])}: {float(gap_hard[-1]):.4f} "
          f"(exact (d2-d1)/N = {(_FIG_D2 - _FIG_D1) / int(loads[-1]):.4f})")

    fig, axes = create_tufte_figure(2, 1, figsize=(6.4, 5.6), sharex=True)
    ax_acc, ax_gap = axes[0], axes[1]
    ax_acc.plot(loads, slot_small, "o-", color=SSM_COLORS["accent"], ms=3.5,
                label=rf"slot reader, $d_1={_FIG_D1}$ (exact)")
    ax_acc.plot(loads, slot_big, "s-", color=SSM_COLORS["highlight"], ms=3.5,
                label=rf"slot reader, $d_2={_FIG_D2}$ (exact)")
    ax_acc.plot(loads, outer_small, "o--", color=SSM_COLORS["accent"], ms=3.5, alpha=0.55,
                label=rf"outer-product, $d={_FIG_D1}$")
    ax_acc.plot(loads, outer_big, "s--", color=SSM_COLORS["highlight"], ms=3.5, alpha=0.55,
                label=rf"outer-product, $d={_FIG_D2}$")
    ax_acc.axhline(1.0, color=SSM_COLORS["baseline"], lw=1.2, ls=":",
                   label="induction reader (exact match)")
    ax_acc.set_xscale("log", base=2)
    set_tufte_title(ax_acc, "Recall accuracy vs load: the knee sits at the state size")
    set_tufte_labels(ax_acc, None, "recall accuracy")
    ax_acc.legend(frameon=False, fontsize=8)
    ax_gap.plot(loads, gap_hard, "o-", color=SSM_COLORS["alert"], ms=3.5,
                label="slot gap (exact)")
    ax_gap.plot(loads, gap_soft, "o--", color=SSM_COLORS["alert"], ms=3.5, alpha=0.55,
                label="outer-product gap")
    ax_gap.axvspan(_FIG_D1, _FIG_D2, color=SSM_COLORS["alert"], alpha=0.10,
                   label=rf"discriminative regime $({_FIG_D1}, {_FIG_D2}]$")
    ax_gap.set_xscale("log", base=2)
    set_tufte_labels(ax_gap, r"stored pairs $N$ (log)", r"accuracy gap $d_2$ vs $d_1$")
    set_tufte_title(ax_gap, "The benchmark separates the two states only near the knee")
    ax_gap.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in save_figure(fig, _OUT_DIR / "discriminative-regime", formats=("png",)):
        print(f"  wrote {p.relative_to(_REPO_ROOT)}")
    plt.close(fig)


def _decay_curve(rho: float) -> tuple[np.ndarray, np.ndarray]:
    seps = np.asarray(_FIG_SEPARATIONS, dtype=float)
    accs = []
    for s_pairs in _FIG_SEPARATIONS:
        per_seed = []
        for s in range(_FIG_N_SEEDS):
            inst = _instance(_FIG_DECAY_PAIRS, n_distractors=s_pairs, seed_offset=7919 * s)
            per_seed.append(accuracy(decay_reader(inst, _FIG_DECAY_DIM, rho), inst.answers))
        accs.append(float(np.mean(per_seed)))
    return seps, np.asarray(accs)


def _fig_length_robustness() -> None:
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
    colors = (SSM_COLORS["alert"], SSM_COLORS["accent"], SSM_COLORS["highlight"],
              SSM_COLORS["baseline"])
    fig, ax = create_tufte_figure(figsize=(6.4, 4.2))
    print(f"  length-robustness table (targets N={_FIG_DECAY_PAIRS}, dim={_FIG_DECAY_DIM}, "
          f"distractor-pair separation, mean of {_FIG_N_SEEDS} seeds):")
    for rho, color in zip(_FIG_RHOS, colors):
        seps, accs = _decay_curve(rho)
        metric_l90 = l90(seps, accs)
        metric_auc = auc_log2(seps, accs)
        rho_label = rf"$\rho={rho}$" if rho < 1.0 else r"$\rho=1$ (no decay)"
        print(f"    rho={rho}: L90={metric_l90:.0f}  AUC={metric_auc:.4f}  "
              f"accs={np.array2string(accs, precision=3)}")
        ax.plot(seps, accs, "o-", color=color, ms=3.5,
                label=rho_label + rf": $L_{{90}}={metric_l90:.0f}$, AUC$={metric_auc:.2f}$")
        ax.axvline(metric_l90, color=color, lw=0.8, ls=":")
    oracle_acc = []
    for s in range(_FIG_N_SEEDS):
        inst = _instance(_FIG_DECAY_PAIRS, n_distractors=_FIG_SEPARATIONS[-1],
                         seed_offset=7919 * s)
        oracle_acc.append(accuracy(induction_reader(inst), inst.answers))
    print(f"    induction reader at s={_FIG_SEPARATIONS[-1]} distractors: "
          f"accuracy={float(np.mean(oracle_acc)):.4f}")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.0, 1.05)
    set_tufte_title(ax, "Fading memory fails at a separation set by its decay (measured)")
    set_tufte_labels(ax, r"distractor pairs $s$ between targets and queries (log)",
                     "recall accuracy")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in save_figure(fig, _OUT_DIR / "length-robustness", formats=("png",)):
        print(f"  wrote {p.relative_to(_REPO_ROOT)}")
    plt.close(fig)


def main() -> None:
    print("Chapter 16 — mqar.py")
    print("=" * 64)

    # Generator + oracle + exact readers on a reference instance.
    inst = _instance(32)
    oracle = oracle_recall(inst)
    print(f"  oracle == stored answers:            "
          f"{bool(jnp.all(oracle == inst.answers))}  (N=32, independent scan)")
    ind = induction_reader(inst)
    print(f"  induction reader == oracle:          {bool(jnp.all(ind == oracle))}")
    inst_small = make_mqar(jax.random.PRNGKey(_FIG_SEED), 32, 64, _FIG_N_VALUES)
    ortho = outer_product_reader(inst_small, dim=64, orthonormal=True)
    print(f"  orthonormal outer product == oracle: "
          f"{bool(jnp.all(ortho == oracle_recall(inst_small)))} "
          f"(n_keys=64 <= dim=64, zero interference)")

    # The exact-capacity identity behind the discriminative-regime proposition.
    print("  slot reader == min(1, d/N) exactly:")
    for n in (8, 16, 64, 256):
        inst_n = _instance(n)
        measured = accuracy(slot_reader(inst_n, _FIG_D1), inst_n.answers)
        exact = slot_accuracy_exact(n, _FIG_D1)
        print(f"    N={n:>3}, d={_FIG_D1}: measured={measured:.6f}  exact={exact:.6f}  "
              f"diff={abs(measured - exact):.1e}")

    # Decay reader: closed form == per-step loop (distractors + neutral gap mixed).
    inst_g = _instance(_FIG_DECAY_PAIRS, gap=17, n_distractors=24)
    fast = decay_reader(inst_g, _FIG_DECAY_DIM, 0.99)
    slow = decay_reader_naive(inst_g, _FIG_DECAY_DIM, 0.99)
    print(f"  decay reader closed form == loop:    {bool(jnp.all(fast == slow))} "
          f"(s=24 distractors + 17 fillers)")

    # The negative control: a neutral-filler gap rescales every stored weight by the
    # same rho^g, and the argmax read-out is scale-invariant -> predictions identical.
    key_ctl = jax.random.PRNGKey(_FIG_SEED + 99)
    base = make_mqar(key_ctl, _FIG_DECAY_PAIRS, _FIG_N_KEYS, _FIG_N_VALUES, gap=0)
    padded = MQARInstance(
        tokens=jnp.concatenate(
            [base.tokens[: 2 * _FIG_DECAY_PAIRS],
             jnp.full((512,), base.filler_id, dtype=base.tokens.dtype),
             base.tokens[2 * _FIG_DECAY_PAIRS :]]
        ),
        query_positions=base.query_positions + 512,
        answers=base.answers,
        n_keys=base.n_keys,
        n_values=base.n_values,
    )
    same = bool(jnp.all(decay_reader(base, _FIG_DECAY_DIM, 0.97)
                        == decay_reader(padded, _FIG_DECAY_DIM, 0.97)))
    print(f"  neutral-gap negative control:        {same} "
          "(gap 0 vs 512: rho=0.97 predictions exactly equal)")

    print("  figures:")
    _fig_discriminative_regime()
    _fig_length_robustness()


if __name__ == "__main__":
    main()
