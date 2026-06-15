r"""Chapter 15 §15.3 — the information-counting bound on lossless recall (P1′).

The chapter's one self-contained impossibility result, stated at a deliberately
*architecture-agnostic* altitude: a deterministic recurrence whose state lives in
$d$ coordinates at $b$ bits each can represent at most $2^{db}$ distinct
post-prefix states, so by pigeonhole it cannot losslessly distinguish more than
$2^{db}$ prefixes. Verbatim copying of length-$n$ sequences over an alphabet
$\Sigma$ needs $|\Sigma|^n$ distinct states, hence

$$ d\,b \;\ge\; n \log_2 |\Sigma| \qquad\Longleftrightarrow\qquad n \;\le\; \frac{d\,b}{\log_2|\Sigma|}. $$

This is the honest, *weaker* shadow of the deep results Chapter 15 cites but does
not re-prove — the $\mathsf{TC}^0$ ceiling (Merrill & Sabharwal) and the copying
separation (Jelassi et al.). It generalizes two shipped propositions, which it
backward-references rather than re-deriving:

* ``ch11:linattn-capacity`` — the linear-attention rank wall ($\operatorname{rank}
  S \le \min(K, d_k, d_v)$) is the *instance* where the state is an outer-product sum;
* ``ch16:discriminative-regime`` — the exact-capacity slot model
  (``companions.ch16.jax.mqar``) with accuracy $\min(1, d/N)$ is the *instance*
  where the state is a $d$-slot ring buffer.

This module verifies the bound numerically by **importing ch16's slot model** (no
re-implementation): with $b = \log_2(\text{vocab})$ bits per slot the counting
threshold is exactly $n^\* = d$, and the measured recall cliff of ``slot_reader``
sits there at ``rtol=0``. The figure overlays the *measured* cliff on the *derived*
threshold.

No training anywhere: the bound is a counting argument, the slot model an analytic
information restriction.

Idiomatic-JAX / port credit
---------------------------
Greenfield. The bound is elementary information counting; the empirical shadow
reuses ``companions/ch16/jax/mqar.py`` (itself the tokenized form of the ch11
mechanism). All arithmetic is in *bits* (log-domain) so nothing overflows — the
naive $2^{db}$ vs $|\Sigma|^n$ state counts would overflow long before any
realistic $n$.

Usage
-----
::

    PYTHONPATH=. python companions/ch15/jax/copying_bound.py
"""

from __future__ import annotations

import math
from pathlib import Path

import jax

# Enable float64 before any jnp array exists (matches every companion since Ch 4)
# and before importing ch16's mqar, which imports jnp at module load.
jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from companions.ch16.jax import mqar  # noqa: E402

__all__ = [
    "state_capacity_bits",
    "min_lossless_state_bits",
    "max_recallable_length",
    "recall_cliff_load",
    "slot_bits_per_pair",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch15"


def state_capacity_bits(d: int, b: float) -> float:
    r"""Information a $d$-coordinate, $b$-bit-per-coordinate state can hold: $d\,b$ bits.

    Parameters
    ----------
    d : int
        Number of state coordinates (``>= 1``).
    b : float
        Bits per coordinate (``> 0``); e.g. ``log2(vocab)`` for one stored token.

    Returns
    -------
    float
        ``d * b``.
    """
    if d < 1:
        raise ValueError(f"d must be >= 1; got {d}")
    if b <= 0.0:
        raise ValueError(f"b must be > 0; got {b}")
    return float(d) * float(b)


def min_lossless_state_bits(n: int, vocab: int) -> float:
    r"""Bits any state must carry to losslessly recall a length-$n$ sequence over $\Sigma$.

    The $|\Sigma|^n$ distinct length-$n$ strings must map to distinct states, so the
    state must distinguish at least $\log_2 |\Sigma|^n = n \log_2 |\Sigma|$ bits.

    Parameters
    ----------
    n : int
        Sequence length (``>= 1``).
    vocab : int
        Alphabet size $|\Sigma|$ (``>= 2``).

    Returns
    -------
    float
        ``n * log2(vocab)``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    if vocab < 2:
        raise ValueError(f"vocab must be >= 2; got {vocab}")
    return float(n) * math.log2(vocab)


def max_recallable_length(d: int, b: float, vocab: int) -> int:
    r"""Largest $n$ the bound permits: $\lfloor d\,b / \log_2|\Sigma| \rfloor$.

    Beyond this length the pigeonhole collision is forced and lossless recall is
    impossible for *any* such recurrence, regardless of training.
    """
    if vocab < 2:
        raise ValueError(f"vocab must be >= 2; got {vocab}")
    return int(math.floor(state_capacity_bits(d, b) / math.log2(vocab)))


def slot_bits_per_pair(vocab: int) -> float:
    r"""Bits one slot of the ch16 slot model carries: $\log_2(\text{vocab})$.

    Choosing $b = \log_2(\text{vocab})$ makes the abstract capacity $d\,b$ coincide
    with the slot model's $d$-pair budget, so the counting threshold and the
    measured slot cliff land at the same load.
    """
    if vocab < 2:
        raise ValueError(f"vocab must be >= 2; got {vocab}")
    return math.log2(vocab)


def recall_cliff_load(d: int) -> int:
    r"""The load $N$ at which the bound bites for a $d$-pair store: $n^\* = d$.

    With $b = \log_2(\text{vocab})$ bits per slot, $d\,b = d\log_2(\text{vocab})$,
    and the lossless requirement $n\log_2(\text{vocab})$ meets it exactly at
    $n = d$. The ch16 slot model realizes this: accuracy is $1$ for $N \le d$ and
    $\min(1, d/N) < 1$ beyond.
    """
    if d < 1:
        raise ValueError(f"d must be >= 1; got {d}")
    return d


# ---------------------------------------------------------------------------
# Figure + measured numbers (§15.3): the derived threshold meets the measured cliff.
# ---------------------------------------------------------------------------

_FIG_D = 16
_FIG_VOCAB = 64  # b = log2(64) = 6 bits/pair; threshold n* = d = 16
_FIG_N_KEYS = 2048
_FIG_N_VALUES = 64
_FIG_SEED = 0
_FIG_LOADS = (4, 8, 16, 24, 32, 48, 64, 96, 128, 256)
_FIG_N_SEEDS = 8


def _slot_instance(n_pairs: int, seed_offset: int = 0) -> mqar.MQARInstance:
    key = jax.random.fold_in(jax.random.PRNGKey(_FIG_SEED), 1000 * n_pairs + seed_offset)
    return mqar.make_mqar(key, n_pairs, _FIG_N_KEYS, _FIG_N_VALUES)


def _measured_slot_accuracy(loads: tuple[int, ...]) -> np.ndarray:
    out = []
    for n in loads:
        accs = []
        for s in range(_FIG_N_SEEDS):
            inst = _slot_instance(n, 7919 * s)
            accs.append(mqar.accuracy(mqar.slot_reader(inst, _FIG_D), inst.answers))
        out.append(float(np.mean(accs)))
    return np.asarray(out)


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
    b = slot_bits_per_pair(_FIG_VOCAB)
    cap = state_capacity_bits(_FIG_D, b)
    nstar = max_recallable_length(_FIG_D, b, _FIG_VOCAB)

    loads = np.asarray(_FIG_LOADS)
    exact = np.asarray([mqar.slot_accuracy_exact(int(n), _FIG_D) for n in loads])
    measured = _measured_slot_accuracy(_FIG_LOADS)

    print("  copying-bound numbers:")
    print(f"    d={_FIG_D}, vocab={_FIG_VOCAB}, b=log2(vocab)={b:.4f} bits/pair")
    print(f"    state capacity d*b = {cap:.4f} bits; counting threshold n* = {nstar}")
    print(f"    recall_cliff_load(d) = {recall_cliff_load(_FIG_D)}  (== n* == d)")
    print(f"    {'N':>4}  {'exact min(1,d/N)':>16}  {'measured slot':>14}  {'|diff|':>8}")
    for i, n in enumerate(loads):
        print(f"    {n:>4}  {exact[i]:>16.6f}  {measured[i]:>14.6f}  {abs(exact[i]-measured[i]):>8.1e}")

    fig, axes = create_tufte_figure(1, 2, figsize=(11.0, 4.3))
    ax_cliff, ax_bits = axes  # type: ignore[misc]

    # Panel A: the measured cliff sits at the capacity threshold N = d.
    fine = np.unique(np.concatenate([loads, np.arange(1, int(loads.max()) + 1)]))
    ax_cliff.plot(fine, [mqar.slot_accuracy_exact(int(n), _FIG_D) for n in fine],
                  color=SSM_COLORS["baseline"], lw=1.2, label=r"exact $\min(1, d/N)$")
    ax_cliff.scatter(loads, measured, s=42, color=SSM_COLORS["accent"], edgecolors="white",
                     linewidths=0.8, zorder=3, label=f"measured slot reader ({_FIG_N_SEEDS} seeds)")
    ax_cliff.axvline(recall_cliff_load(_FIG_D), color=SSM_COLORS["alert"], lw=1.0, ls="--",
                     label=rf"capacity threshold $n^*=d={_FIG_D}$")
    ax_cliff.set_xscale("log", base=2)
    set_tufte_title(ax_cliff, "The recall cliff sits at the state capacity")
    set_tufte_labels(ax_cliff, r"stored pairs $N$ (log)", "recall accuracy")
    ax_cliff.legend(frameon=False, fontsize=8, loc="lower left")

    # Panel B: the bits inequality — fixed budget d*b vs the n*log2|Σ| requirement.
    ns = np.arange(1, 33)
    need = np.asarray([min_lossless_state_bits(int(n), _FIG_VOCAB) for n in ns])
    ax_bits.plot(ns, need, color=SSM_COLORS["accent"], lw=1.6,
                 label=r"required $n\log_2|\Sigma|$")
    ax_bits.axhline(cap, color=SSM_COLORS["alert"], lw=1.4, ls="-",
                    label=rf"state budget $d\,b={cap:.0f}$ bits")
    ax_bits.axvline(nstar, color=SSM_COLORS["baseline"], lw=1.0, ls=":")
    ax_bits.scatter([nstar], [cap], s=55, color=SSM_COLORS["highlight"], zorder=4,
                    label=rf"crossing $n^*={nstar}$")
    set_tufte_title(ax_bits, "Lossless recall needs more bits than the state holds")
    set_tufte_labels(ax_bits, r"sequence length $n$", "bits")
    ax_bits.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in save_figure(fig, _OUT_DIR / "copying-bound", formats=("png",)):
        print(f"  wrote {p.relative_to(_REPO_ROOT)}")
    plt.close(fig)


def main() -> None:
    print("Chapter 15 — copying_bound.py")
    print("=" * 64)
    # The bound's content as a one-line sanity check: the crossing is at n = d.
    b = slot_bits_per_pair(_FIG_VOCAB)
    assert max_recallable_length(_FIG_D, b, _FIG_VOCAB) == recall_cliff_load(_FIG_D)
    print(f"  threshold check: max_recallable_length(d={_FIG_D}, b=log2 vocab, vocab={_FIG_VOCAB}) "
          f"= {max_recallable_length(_FIG_D, b, _FIG_VOCAB)} == d  ✓")
    _make_figure()


if __name__ == "__main__":
    main()
