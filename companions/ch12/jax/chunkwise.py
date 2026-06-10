r"""Chapter 12 §12.5 — chunkwise DeltaNet and the WY representation.

The recurrent DeltaNet of ``delta_rule.py`` is $O(L)$ *sequential*. The DeltaNet
paper (Yang et al., arXiv:2406.06484 §3.2/§4) recovers hardware parallelism by
splitting the sequence into chunks of size $C$: unrolling the recurrence across
one chunk gives

.. math::

    S_{\mathrm{end}} = S_{\mathrm{entry}}\,P + R,
    \qquad
    P = \prod_{t \in \mathrm{chunk}} (I - \beta_t k_t k_t^\top),

so the cross-chunk recurrence is one affine update per chunk ($L/C$ steps)
while everything inside a chunk is chunk-local. The erase product $P$ is a
product of rank-one perturbations of $I$, and exactly as in the blocked
Householder QR of numerical linear algebra it admits a compact **WY
representation**

.. math::

    P = I - W^\top Y, \qquad
    w_t = \beta_t\Bigl(k_t - \textstyle\sum_{j<t} (k_j^\top k_t)\,w_j\Bigr),
    \quad y_t = k_t,

(rows of $W, Y \in \mathbb{R}^{C \times d_k}$ hold the factor columns), so
applying $P$ to a state is two GEMMs, never a $(d_k, d_k)$ matrix product
chain. Same move as Chapter 11's recurrent↔parallel duality
(`ch11:recurrent-parallel-equivalence`): one operator, two computation orders,
equality to machine precision as the correctness certificate. Pinned here:

* ``chunkwise == recurrent`` outputs and final state, ``< 1e-12``, for every
  chunk size $C$ dividing $L$ (including $C = 1$ and $C = L$);
* ``I - W^T Y == explicit product`` $\prod(I - \beta_t k_t k_t^\top)$,
  ``< 1e-12`` in the stable regime (unit-norm keys, $\beta < 1$; with
  $\beta\|k\|^2 \gg 2$ the product's entries grow exponentially and the
  *absolute* agreement degrades with them);
* :func:`apply_wy_to_state` equals right-multiplication by the explicit
  product, ``< 1e-12``.

Idiomatic-JAX note (NumPy->JAX teaching point)
----------------------------------------------
The WY recursion is loop-carried in the *rows written so far*, so it stays a
Python loop over the (small) chunk — unrolled under ``jit`` — while the
within-chunk sweep reuses the ``lax.scan`` driver via its ``initial_state``
hook. Knowing *which* loop to scan and which to unroll is the JAX skill this
module teaches.

Port credit
-----------
Ported from ``post_transformers/experiments/jax/week12/chunkwise.py``
(arXiv:2406.06484), with two simplifications: the dead scan-body code in the
WY builder is dropped, and the chunk driver seeds
``delta_rule_recurrent(initial_state=...)`` instead of duplicating the scan.

Usage
-----
::

    PYTHONPATH=. python companions/ch12/jax/chunkwise.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax

# Enable float64 before any jnp array is created (matches Chapters 4, 7-11).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from companions.ch12.jax.delta_rule import delta_rule_recurrent  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "chunk_wy_representation",
    "apply_wy_to_state",
    "explicit_erase_product",
    "delta_rule_chunkwise",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch12"


# ---------------------------------------------------------------------------
# §12.5 — the WY representation of a chunk's erase product
# ---------------------------------------------------------------------------


def chunk_wy_representation(
    keys: jnp.ndarray, betas: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Build $(W, Y)$ with $\prod_{t=1}^{C}(I - \beta_t k_t k_t^\top) = I - W^\top Y$.

    Householder/WY recursion: with $P_{1:t-1} = I - W_{1:t-1}^\top Y_{1:t-1}$,

    .. math::

        P_{1:t} = P_{1:t-1}(I - \beta_t k_t k_t^\top)
                = P_{1:t-1} - \beta_t (P_{1:t-1} k_t) k_t^\top,

    so the new row is $w_t = \beta_t P_{1:t-1} k_t
    = \beta_t (k_t - \sum_{j<t} (k_j^\top k_t)\, w_j)$ and $y_t = k_t$.
    Each $w_t$ depends on all earlier rows — a genuinely sequential (but
    chunk-sized) recursion.

    Parameters
    ----------
    keys : jnp.ndarray, shape (C, d_k)
        Chunk keys.
    betas : jnp.ndarray, shape (C,)
        Chunk learning rates.

    Returns
    -------
    W : jnp.ndarray, shape (C, d_k)
    Y : jnp.ndarray, shape (C, d_k)
        ``Y is keys`` (returned for call-site symmetry).
    """
    if keys.ndim != 2:
        raise ValueError(f"keys must be 2D (C, d_k); got shape {keys.shape}")
    if betas.shape != (keys.shape[0],):
        raise ValueError(
            f"betas must have shape (C,) = ({keys.shape[0]},); got {betas.shape}"
        )
    rows = []
    for t in range(keys.shape[0]):
        correction = sum(((keys[j] @ keys[t]) * rows[j] for j in range(t)), start=0.0)
        rows.append(betas[t] * (keys[t] - correction))
    return jnp.stack(rows), keys


def apply_wy_to_state(state: jnp.ndarray, W: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    r"""Apply the chunk erase product $P = I - W^\top Y$ to a state: $SP = S - (S W^\top) Y$.

    Two GEMMs — $(d_v, C)$ then $(d_v, d_k)$ — instead of materialising the
    $(d_k, d_k)$ product.

    Parameters
    ----------
    state : jnp.ndarray, shape (d_v, d_k)
    W, Y : jnp.ndarray, shape (C, d_k)

    Returns
    -------
    jnp.ndarray, shape (d_v, d_k)
    """
    if state.shape[-1] != W.shape[-1] or W.shape != Y.shape:
        raise ValueError(
            f"state (d_v, d_k), W and Y (C, d_k) must share d_k; got "
            f"state {state.shape}, W {W.shape}, Y {Y.shape}"
        )
    return state - (state @ W.T) @ Y


def explicit_erase_product(keys: jnp.ndarray, betas: jnp.ndarray) -> jnp.ndarray:
    r"""The materialised product $\prod_{t=1}^{C}(I - \beta_t k_t k_t^\top)$ (test oracle).

    Multiplies the $(d_k, d_k)$ factors left-to-right (the order the recurrence
    applies them to a *row* state: $S P_1 P_2 \cdots$). Deliberately the
    expensive form — the independent witness for the WY identity.

    Parameters
    ----------
    keys : jnp.ndarray, shape (C, d_k)
    betas : jnp.ndarray, shape (C,)

    Returns
    -------
    jnp.ndarray, shape (d_k, d_k)
    """
    if keys.ndim != 2 or betas.shape != (keys.shape[0],):
        raise ValueError(f"keys (C, d_k) and betas (C,) required; got {keys.shape}, {betas.shape}")
    d_k = keys.shape[1]
    product = jnp.eye(d_k, dtype=keys.dtype)
    for t in range(keys.shape[0]):
        product = product @ (jnp.eye(d_k, dtype=keys.dtype) - betas[t] * jnp.outer(keys[t], keys[t]))
    return product


# ---------------------------------------------------------------------------
# §12.5 — the chunkwise driver
# ---------------------------------------------------------------------------


def delta_rule_chunkwise(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    betas: jnp.ndarray,
    chunk_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Chunkwise DeltaNet: cross-chunk state passing, chunk-local sweeps.

    Equals :func:`delta_rule.delta_rule_recurrent` exactly (modulo float
    accumulation order) — pinned ``< 1e-12`` in ``tests/test_chunkwise.py``.
    The within-chunk sweep reuses the recurrent driver seeded with the chunk's
    entry state; the WY primitives above are the algebra a fused kernel would
    exploit, kept tested next to it.

    Parameters
    ----------
    q, k : jnp.ndarray, shape (L, d_k)
    v : jnp.ndarray, shape (L, d_v)
    betas : jnp.ndarray, shape (L,)
    chunk_size : int
        Must divide $L$ exactly; pad upstream if needed.

    Returns
    -------
    outputs : jnp.ndarray, shape (L, d_v)
    final_state : jnp.ndarray, shape (d_v, d_k)
    """
    length = q.shape[0]
    if chunk_size <= 0 or length % chunk_size != 0:
        raise ValueError(
            f"chunk_size = {chunk_size} must be positive and divide L = {length} exactly"
        )
    d_k, d_v = q.shape[1], v.shape[1]
    state = jnp.zeros((d_v, d_k), dtype=v.dtype)
    out_chunks = []
    for c in range(length // chunk_size):
        s, e = c * chunk_size, (c + 1) * chunk_size
        outs, state = delta_rule_recurrent(
            q[s:e], k[s:e], v[s:e], betas[s:e], initial_state=state
        )
        out_chunks.append(outs)
    return jnp.concatenate(out_chunks), state


# ---------------------------------------------------------------------------
# Figure: both equivalences are exact across chunk sizes
# ---------------------------------------------------------------------------

_CHUNK_SIZES = (1, 2, 4, 8, 16, 32, 64)


def make_chunkwise_figure() -> "Figure":
    """Left: max |chunkwise - recurrent| across chunk sizes (machine zero).
    Right: max |(I - W^T Y) - explicit product| across chunk sizes (machine zero)."""
    import numpy as np

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    rng = np.random.default_rng(0)
    length, d_k, d_v = 64, 8, 6
    q = jnp.asarray(rng.standard_normal((length, d_k)))
    # Unit-norm keys + beta < 1: the stable regime (beta * ||k||^2 < 2, §12.4),
    # so the erase product stays bounded and the WY agreement is eps-scale at
    # every chunk size.
    k_raw = rng.standard_normal((length, d_k))
    k = jnp.asarray(k_raw / np.linalg.norm(k_raw, axis=1, keepdims=True))
    v = jnp.asarray(rng.standard_normal((length, d_v)))
    betas = jnp.asarray(rng.uniform(0.1, 0.9, size=length))
    y_ref, s_ref = delta_rule_recurrent(q, k, v, betas)

    sizes = np.asarray(_CHUNK_SIZES)
    resid_chunk = np.empty(len(sizes))
    for i, c in enumerate(sizes):
        y_c, s_c = delta_rule_chunkwise(q, k, v, betas, int(c))
        resid_chunk[i] = max(
            float(jnp.max(jnp.abs(y_c - y_ref))), float(jnp.max(jnp.abs(s_c - s_ref)))
        )

    wy_sizes = np.asarray([c for c in _CHUNK_SIZES if c >= 2])
    resid_wy = np.empty(len(wy_sizes))
    for i, c in enumerate(wy_sizes):
        keys_c = k[: int(c)]
        betas_c = betas[: int(c)]
        W, Y = chunk_wy_representation(keys_c, betas_c)
        p_wy = jnp.eye(d_k) - W.T @ Y
        p_explicit = explicit_erase_product(keys_c, betas_c)
        resid_wy[i] = float(jnp.max(jnp.abs(p_wy - p_explicit)))

    fig, (ax1, ax2) = create_tufte_figure(ncols=2, figsize=(11.0, 4.2))

    ax1.semilogy(sizes, np.maximum(resid_chunk, 1e-18), "o-", color=SSM_COLORS["accent"])
    ax1.axhline(1e-12, color=SSM_COLORS["alert"], lw=0.8, ls="--", label=r"$10^{-12}$ pin")
    ax1.set_xscale("log", base=2)
    set_tufte_title(ax1, "Chunkwise $\\equiv$ recurrent (float64)")
    set_tufte_labels(ax1, xlabel=r"chunk size $C$ ($L = 64$)",
                     ylabel=r"max $|$chunkwise $-$ recurrent$|$")
    ax1.legend(loc="upper left", fontsize=8, frameon=False)

    ax2.semilogy(wy_sizes, np.maximum(resid_wy, 1e-18), "o-", color=SSM_COLORS["accent"])
    ax2.axhline(1e-12, color=SSM_COLORS["alert"], lw=0.8, ls="--", label=r"$10^{-12}$ pin")
    ax2.set_xscale("log", base=2)
    set_tufte_title(ax2, "WY factors $\\equiv$ explicit product")
    set_tufte_labels(ax2, xlabel=r"chunk size $C$",
                     ylabel=r"max $|(I - W^\top Y) - \prod(I - \beta kk^\top)|$")
    ax2.legend(loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    return fig


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from companions._shared.plot_utils import save_figure

    print("Chapter 12 — chunkwise.py")
    print("=" * 64)

    rng = np.random.default_rng(0)
    length, d_k, d_v = 64, 8, 6
    q = jnp.asarray(rng.standard_normal((length, d_k)))
    # Stable regime, matching the figure and the test stream.
    k_raw = rng.standard_normal((length, d_k))
    k = jnp.asarray(k_raw / np.linalg.norm(k_raw, axis=1, keepdims=True))
    v = jnp.asarray(rng.standard_normal((length, d_v)))
    betas = jnp.asarray(rng.uniform(0.1, 0.9, size=length))
    y_ref, s_ref = delta_rule_recurrent(q, k, v, betas)

    # §12.5 chunkwise == recurrent across chunk sizes.
    for c in (1, 8, 64):
        y_c, s_c = delta_rule_chunkwise(q, k, v, betas, c)
        resid = max(float(jnp.max(jnp.abs(y_c - y_ref))), float(jnp.max(jnp.abs(s_c - s_ref))))
        print(f"  C={c:2d}: max |chunkwise - recurrent| = {resid:.2e}")

    # §12.5 WY == explicit erase product, and the two-GEMM application.
    keys_c, betas_c = k[:16], betas[:16]
    W, Y = chunk_wy_representation(keys_c, betas_c)
    p_wy = jnp.eye(d_k) - W.T @ Y
    p_explicit = explicit_erase_product(keys_c, betas_c)
    print(f"  C=16: max |(I - W^T Y) - explicit product| = "
          f"{float(jnp.max(jnp.abs(p_wy - p_explicit))):.2e}")
    state = jnp.asarray(rng.standard_normal((d_v, d_k)))
    resid_apply = float(jnp.max(jnp.abs(apply_wy_to_state(state, W, Y) - state @ p_explicit)))
    print(f"  C=16: max |apply_wy(S) - S @ product|      = {resid_apply:.2e}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_chunkwise_figure()
    for p in save_figure(fig, _OUT_DIR / "chunkwise-equivalence", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
