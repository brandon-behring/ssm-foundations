r"""Chapter 12 §12.1-12.2 — the delta rule as one explicit gradient step.

The associative-recall objective at time $t$ is the squared retrieval error

.. math::

    \mathcal{L}_t(S) = \tfrac{1}{2}\,\|S k_t - v_t\|^2,

whose gradient flow $\dot S = -\nabla\mathcal{L}_t(S) = (v_t - S k_t)\,k_t^\top$
is the chapter's organizing ODE. One **explicit (forward-Euler) step** of size
$\beta_t$ is the delta rule, and the delta rule *is* DeltaNet's state update
(Yang et al., arXiv:2406.06484):

.. math::

    S_t = S_{t-1} + \beta_t\,(v_t - S_{t-1} k_t)\,k_t^\top
        = S_{t-1}\,(I - \beta_t k_t k_t^\top) + \beta_t\,v_t k_t^\top,
    \qquad o_t = S_t q_t.

Convention (held fixed for all of Chapter 12): $S_t \in \mathbb{R}^{d_v \times d_k}$
with retrieval $S k$ — the *right-projector* form used by ch03's low-rank-update
discussion and the DeltaNet paper. Chapter 11 §11.7 wrote the transposed
(left-projector) form $S \in \mathbb{R}^{d_k \times d_v}$; the two are the same
operator under $S \mapsto S^\top$.

The $(I - \beta_t k_t k_t^\top)$ factor *selectively erases* along $k_t$ before
the rank-one write $\beta_t v_t k_t^\top$ — this is the "smarter write rule"
Chapter 11 §11.6 promised as the answer to the additive capacity wall. Two
load-bearing facts pinned here:

* **fixed point** — $S^\star = v k^\top / \|k\|^2$ satisfies
  $\mathrm{step}(S^\star) = S^\star$ exactly (the zero of the gradient flow);
* **recall payoff** — storing $K$ pairs with sequential delta-rule writes
  ($\beta = 1$, unit keys) retrieves with much lower interference error than the
  additive state $S = \sum_i v_i k_i^\top$ of Chapter 11; with *orthonormal*
  keys both are exact to machine precision (no interference to erase).

Computationally the rank-one form needs one mat-vec plus one outer product per
step ($O(d_v d_k)$) — the $(I - \beta k k^\top)$ matrix is never materialised
(the naive oracle below materialises it on purpose, as an independent witness).

Idiomatic-JAX note (NumPy->JAX teaching point)
----------------------------------------------
The sequential carry uses ``lax.scan`` (the Chapters 8-11 primitive), not the
Python ``for`` loop a NumPy reference would write; the naive oracle *is* that
Python loop, kept deliberately different (materialised projector, explicit
loop) so the ``< 1e-12`` agreement is a correctness certificate, not a tautology.

Port credit
-----------
Ported from ``post_transformers/experiments/jax/week12/delta_rule.py``
(arXiv:2406.06484 §3 has the recurrent form), simplified from batched
``(B, L, d)`` to this book's unbatched ``(L, d)`` companion contract. The
recall-comparison helpers are new here (they redeem the ch11 §11.6 promise).

Usage
-----
::

    PYTHONPATH=. python companions/ch12/jax/delta_rule.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax

# Enable float64 before any jnp array is created (matches Chapters 4, 7-11).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "delta_rule_step",
    "delta_rule_recurrent",
    "delta_rule_naive",
    "delta_rule_fixed_point",
    "additive_state",
    "recall_errors",
    "overwrite_retrieval",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch12"


# ---------------------------------------------------------------------------
# §12.1 — the delta rule: one explicit gradient step on the recall objective
# ---------------------------------------------------------------------------


def _check_stream(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, rates: jnp.ndarray) -> None:
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError(f"q, k, v must each be 2D (L, d); got {q.shape}, {k.shape}, {v.shape}")
    if q.shape != k.shape:
        raise ValueError(f"q and k must share shape (L, d_k); got {q.shape} and {k.shape}")
    if v.shape[0] != q.shape[0]:
        raise ValueError(f"v must have the same length L as q/k; got {v.shape[0]} vs {q.shape[0]}")
    if rates.shape != (q.shape[0],):
        raise ValueError(f"rates must have shape (L,) = ({q.shape[0]},); got {rates.shape}")


def delta_rule_step(
    state: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    beta: jnp.ndarray | float,
) -> jnp.ndarray:
    r"""One delta-rule update $S \leftarrow S + \beta\,(v - S k)\,k^\top$.

    The rank-one form: one mat-vec and one outer product, $O(d_v d_k)$; the
    $(I - \beta k k^\top)$ projector is never materialised.

    Parameters
    ----------
    state : jnp.ndarray, shape (d_v, d_k)
        Current state $S$.
    key : jnp.ndarray, shape (d_k,)
    value : jnp.ndarray, shape (d_v,)
    beta : scalar
        Learning rate (DeltaNet's write strength); stability requires
        $\beta\|k\|^2 \in (0, 2)$ — see ``stability.py``.

    Returns
    -------
    jnp.ndarray, shape (d_v, d_k)
        Updated state.
    """
    if state.ndim != 2 or key.ndim != 1 or value.ndim != 1:
        raise ValueError(
            f"expected state (d_v, d_k), key (d_k,), value (d_v,); "
            f"got {state.shape}, {key.shape}, {value.shape}"
        )
    if state.shape != (value.shape[0], key.shape[0]):
        raise ValueError(
            f"state must have shape (d_v, d_k) = ({value.shape[0]}, {key.shape[0]}); "
            f"got {state.shape}"
        )
    error = value - state @ key  # (d_v,)
    return state + beta * jnp.outer(error, key)


def delta_rule_fixed_point(key: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    r"""The unique fixed point $S^\star = v k^\top / \|k\|^2$ of the single-pair update.

    $S^\star$ retrieves exactly ($S^\star k = v$), so the error term vanishes and
    every $\beta$ leaves it invariant — it is the zero of the gradient flow that
    both DeltaNet (explicit step) and Longhorn (implicit step) discretise.

    Parameters
    ----------
    key : jnp.ndarray, shape (d_k,)
        Must be nonzero.
    value : jnp.ndarray, shape (d_v,)

    Returns
    -------
    jnp.ndarray, shape (d_v, d_k)
    """
    k_sq = float(key @ key)
    if k_sq == 0.0:
        raise ValueError("key must be nonzero: the fixed point v k^T / ||k||^2 is undefined")
    return jnp.outer(value, key) / k_sq


# ---------------------------------------------------------------------------
# §12.2 — DeltaNet over a sequence: lax.scan vs the materialised-projector oracle
# ---------------------------------------------------------------------------


def delta_rule_recurrent(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    betas: jnp.ndarray,
    initial_state: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Sequential DeltaNet via ``lax.scan``: update with $(k_t, v_t, \beta_t)$, read $o_t = S_t q_t$.

    The read uses the **post-update** state (canonical DeltaNet convention): the
    association written at $t$ is visible to the query at $t$.

    Parameters
    ----------
    q, k : jnp.ndarray, shape (L, d_k)
    v : jnp.ndarray, shape (L, d_v)
    betas : jnp.ndarray, shape (L,)
    initial_state : jnp.ndarray, shape (d_v, d_k), optional
        Chunk-entry state for streaming / chunkwise use (``chunkwise.py``);
        defaults to zeros.

    Returns
    -------
    outputs : jnp.ndarray, shape (L, d_v)
    final_state : jnp.ndarray, shape (d_v, d_k)
    """
    _check_stream(q, k, v, betas)
    d_k, d_v = q.shape[1], v.shape[1]
    if initial_state is None:
        initial_state = jnp.zeros((d_v, d_k), dtype=v.dtype)
    elif initial_state.shape != (d_v, d_k):
        raise ValueError(
            f"initial_state must have shape (d_v, d_k) = ({d_v}, {d_k}); got {initial_state.shape}"
        )

    def step(state: jnp.ndarray, inp) -> tuple[jnp.ndarray, jnp.ndarray]:
        q_t, k_t, v_t, beta_t = inp
        new_state = state + beta_t * jnp.outer(v_t - state @ k_t, k_t)
        return new_state, new_state @ q_t

    final_state, outputs = jax.lax.scan(step, initial_state, (q, k, v, betas))
    return outputs, final_state


def delta_rule_naive(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    betas: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Python-loop oracle: materialises $S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$.

    Deliberately the *projector* form (an explicit $(d_k, d_k)$ matrix per step),
    not the rank-one rearrangement — so its ``< 1e-12`` agreement with
    :func:`delta_rule_recurrent` certifies the algebraic identity between the
    two forms, not just the scan plumbing. Same post-update read convention.

    Parameters and returns as :func:`delta_rule_recurrent` (zero initial state).
    """
    _check_stream(q, k, v, betas)
    length, d_k = q.shape
    d_v = v.shape[1]
    state = jnp.zeros((d_v, d_k), dtype=v.dtype)
    identity = jnp.eye(d_k, dtype=v.dtype)
    outputs = []
    for t in range(length):
        erase = identity - betas[t] * jnp.outer(k[t], k[t])  # (d_k, d_k), materialised
        state = state @ erase + betas[t] * jnp.outer(v[t], k[t])
        outputs.append(state @ q[t])
    return jnp.stack(outputs), state


# ---------------------------------------------------------------------------
# §12.2 — the recall payoff: delta-rule overwrite vs ch11's additive accumulation
# ---------------------------------------------------------------------------


def additive_state(keys: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    r"""Chapter 11's additive state $S = \sum_i v_i k_i^\top$ in this chapter's convention.

    The $A = I$ accumulation (no erasure): every stored pair interferes with
    every retrieval through the key overlaps $k_i^\top k_j$.

    Parameters
    ----------
    keys : jnp.ndarray, shape (K, d_k)
    values : jnp.ndarray, shape (K, d_v)

    Returns
    -------
    jnp.ndarray, shape (d_v, d_k)
    """
    if keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != values.shape[0]:
        raise ValueError(
            f"keys (K, d_k) and values (K, d_v) must share K; got {keys.shape} and {values.shape}"
        )
    return values.T @ keys


def recall_errors(
    n_pairs: int,
    d_k: int = 32,
    d_v: int = 16,
    seed: int = 0,
    orthonormal: bool = False,
) -> tuple[float, float]:
    r"""Mean retrieval error after storing ``n_pairs`` associations: additive vs delta rule.

    Stores $(k_i, v_i)$ with unit-norm keys, then retrieves every stored key and
    reports the mean $\ell_\infty$ error $\max_j |(S k_i - v_i)_j|$ averaged over
    pairs. The delta-rule writer runs one sequential pass with $\beta = 1$.

    With ``orthonormal=True`` the keys are exactly orthonormal (QR of a random
    matrix; requires ``n_pairs <= d_k``) and *both* states retrieve exactly —
    interference, not capacity, is what the delta rule erases.

    Parameters
    ----------
    n_pairs : int
        Number of stored associations $K$.
    d_k, d_v : int
        Key and value dimensions.
    seed : int
    orthonormal : bool
        Use exactly orthonormal keys instead of random unit-norm keys.

    Returns
    -------
    additive_error, delta_error : float
    """
    import numpy as np

    if orthonormal and n_pairs > d_k:
        raise ValueError(f"orthonormal keys need n_pairs <= d_k; got {n_pairs} > {d_k}")
    rng = np.random.default_rng(seed)
    if orthonormal:
        mat = rng.standard_normal((d_k, n_pairs))
        keys = jnp.asarray(np.linalg.qr(mat)[0].T)  # (K, d_k), orthonormal rows
    else:
        raw = rng.standard_normal((n_pairs, d_k))
        keys = jnp.asarray(raw / np.linalg.norm(raw, axis=1, keepdims=True))
    values = jnp.asarray(rng.standard_normal((n_pairs, d_v)))

    s_add = additive_state(keys, values)
    _, s_delta = delta_rule_recurrent(keys, keys, values, jnp.ones(n_pairs))

    def mean_err(s: jnp.ndarray) -> float:
        return float(jnp.mean(jnp.max(jnp.abs(keys @ s.T - values), axis=1)))

    return mean_err(s_add), mean_err(s_delta)


def overwrite_retrieval(d_k: int = 8, d_v: int = 4, seed: int = 0) -> tuple[float, float]:
    r"""The defining delta-rule semantics: re-storing a key *replaces* its value.

    Stores $(k, v_1)$, then re-stores $(k, v_2)$ on the same unit key with
    $\beta = 1$. The delta rule retrieves $v_2$ exactly (the erase term removes
    $v_1$ before the write); the additive state retrieves $v_1 + v_2$ — the old
    binding lingers forever under $A = I$ accumulation.

    Parameters
    ----------
    d_k, d_v : int
    seed : int

    Returns
    -------
    delta_residual, additive_residual : float
        $\max_j |(S k - v_2)_j|$ for the delta-rule state (machine zero) and
        the additive state ($= \max_j |v_{1,j}|$, the stale residue).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    key = rng.standard_normal(d_k)
    key = jnp.asarray(key / np.linalg.norm(key))
    v1 = jnp.asarray(rng.standard_normal(d_v))
    v2 = jnp.asarray(rng.standard_normal(d_v))

    s_delta = delta_rule_step(jnp.zeros((d_v, d_k)), key, v1, 1.0)
    s_delta = delta_rule_step(s_delta, key, v2, 1.0)
    s_add = additive_state(jnp.stack([key, key]), jnp.stack([v1, v2]))

    delta_residual = float(jnp.max(jnp.abs(s_delta @ key - v2)))
    additive_residual = float(jnp.max(jnp.abs(s_add @ key - v2)))
    return delta_residual, additive_residual


# ---------------------------------------------------------------------------
# Figure: recall error vs number of stored pairs (the §11.6 wall, revisited)
# ---------------------------------------------------------------------------

_RECALL_KS = (2, 4, 8, 16, 24, 32, 48, 64)


def make_recall_figure() -> "Figure":
    """Left: random unit keys — delta-rule overwrite beats additive accumulation.
    Right: orthonormal keys — both exact (it is interference the delta rule erases)."""
    import numpy as np

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    ks = np.asarray(_RECALL_KS)
    add_err = np.empty(len(ks))
    delta_err = np.empty(len(ks))
    for i, kk in enumerate(ks):
        add_err[i], delta_err[i] = recall_errors(int(kk))

    ks_orth = np.asarray([kk for kk in _RECALL_KS if kk <= 32])
    add_orth = np.empty(len(ks_orth))
    delta_orth = np.empty(len(ks_orth))
    for i, kk in enumerate(ks_orth):
        add_orth[i], delta_orth[i] = recall_errors(int(kk), orthonormal=True)

    fig, (ax1, ax2) = create_tufte_figure(ncols=2, figsize=(11.0, 4.2))

    ax1.semilogy(ks, add_err, "s-", color=SSM_COLORS["highlight"], label="additive (ch11, $A=I$)")
    ax1.semilogy(ks, delta_err, "o-", color=SSM_COLORS["accent"], label=r"delta rule ($\beta=1$)")
    ax1.axvline(32, color=SSM_COLORS["baseline"], lw=0.8, ls=":")
    ax1.annotate(r"$K = d_k$", xy=(32, ax1.get_ylim()[0]), xytext=(33, 1.5e-2),
                 fontsize=9, color=SSM_COLORS["baseline"])
    set_tufte_title(ax1, "Random unit keys: overwrite beats accumulate")
    set_tufte_labels(ax1, xlabel=r"stored pairs $K$ ($d_k = 32$)",
                     ylabel=r"mean $\ell_\infty$ retrieval error")
    ax1.legend(loc="lower right", fontsize=8, frameon=False)

    ax2.semilogy(ks_orth, np.maximum(add_orth, 1e-18), "s-", color=SSM_COLORS["highlight"],
                 label="additive")
    ax2.semilogy(ks_orth, np.maximum(delta_orth, 1e-18), "o-", color=SSM_COLORS["accent"],
                 label="delta rule")
    ax2.axhline(1e-12, color=SSM_COLORS["alert"], lw=0.8, ls="--", label=r"$10^{-12}$ pin")
    set_tufte_title(ax2, "Orthonormal keys: both exact")
    set_tufte_labels(ax2, xlabel=r"stored pairs $K \leq d_k$",
                     ylabel=r"mean $\ell_\infty$ retrieval error")
    ax2.legend(loc="center right", fontsize=8, frameon=False)

    fig.tight_layout()
    return fig


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from companions._shared.plot_utils import save_figure

    print("Chapter 12 — delta_rule.py")
    print("=" * 64)

    rng = np.random.default_rng(0)
    length, d_k, d_v = 48, 8, 6
    q = jnp.asarray(rng.standard_normal((length, d_k)))
    # Unit-norm keys + beta < 1 keep beta * ||k||^2 inside the (0, 2) stability
    # interval (§12.4) — the regime the chapter recommends running DeltaNet in.
    k_raw = rng.standard_normal((length, d_k))
    k = jnp.asarray(k_raw / np.linalg.norm(k_raw, axis=1, keepdims=True))
    v = jnp.asarray(rng.standard_normal((length, d_v)))
    betas = jnp.asarray(rng.uniform(0.1, 0.9, size=length))

    # §12.2 scan == materialised-projector oracle.
    y_scan, s_scan = delta_rule_recurrent(q, k, v, betas)
    y_naive, s_naive = delta_rule_naive(q, k, v, betas)
    print(f"  scan == naive oracle:      max diff = "
          f"{float(jnp.max(jnp.abs(y_scan - y_naive))):.2e}  (outputs)")
    print(f"                             max diff = "
          f"{float(jnp.max(jnp.abs(s_scan - s_naive))):.2e}  (final state)")

    # §12.1 fixed point: step(S*) == S* for any beta.
    key1 = jnp.asarray(rng.standard_normal(d_k))
    val1 = jnp.asarray(rng.standard_normal(d_v))
    s_star = delta_rule_fixed_point(key1, val1)
    drift = max(
        float(jnp.max(jnp.abs(delta_rule_step(s_star, key1, val1, b) - s_star)))
        for b in (0.1, 1.0, 2.5)
    )
    print(f"  fixed point S* = v k^T/|k|^2: max |step(S*) - S*| = {drift:.2e} over beta grid")

    # §12.2 beta = 1, unit key: the just-written pair is retrieved exactly.
    k_unit = key1 / jnp.linalg.norm(key1)
    s_any = jnp.asarray(rng.standard_normal((d_v, d_k)))
    s_after = delta_rule_step(s_any, k_unit, val1, 1.0)
    print(f"  beta=1, unit key: |S k - v| after write = "
          f"{float(jnp.max(jnp.abs(s_after @ k_unit - val1))):.2e}")

    # §12.2 the defining semantics: re-storing a key replaces its value.
    d_res, a_res = overwrite_retrieval()
    print(f"  overwrite (k, v1)->(k, v2): delta retrieves v2 to {d_res:.2e}; "
          f"additive is off by {a_res:.3f} (the stale |v1| residue)")

    # §12.2 recall payoff (numbers quoted in the figure caption).
    print("  recall error (mean l_inf, d_k=32): additive vs delta")
    for kk in (16, 32, 64):
        a, d = recall_errors(kk)
        print(f"    K={kk:2d} random keys:      additive = {a:.3f}   delta = {d:.3f}")
    a, d = recall_errors(32, orthonormal=True)
    print(f"    K=32 orthonormal keys:  additive = {a:.2e}   delta = {d:.2e}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_recall_figure()
    for p in save_figure(fig, _OUT_DIR / "delta-vs-additive-recall", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
