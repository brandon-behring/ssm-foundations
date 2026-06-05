r"""Chapter 11 §11.6 — associative recall and the capacity of a finite state.

The honest landing of the chapter. Linear attention stores $K$ key-value pairs
in a *fixed-size* matrix state $S = \sum_i \phi(k_i)v_i^\top$ of rank
$\le \min(K, d_k)$ (Proposition ``ch11:linattn-capacity``). Reading key $k_j$
back with the (unnormalized) linear-attention rule gives

.. math::

    \hat v_j = S^\top\phi(k_j)
             = \sum_i \langle\phi(k_j),\phi(k_i)\rangle\,v_i
             = v_j + \underbrace{\sum_{i\ne j}\langle\phi(k_j),\phi(k_i)\rangle\,v_i}_{\text{interference}},

so exact recall ($\hat v_j = v_j$ for all $j$) needs the feature Gram
$A = \Phi\Phi^\top$ to equal the identity — i.e. orthonormal features, possible
only when $K \le d_k$. Past capacity the interference cannot vanish, and the
per-binding retrieval error grows like $\sqrt{K/d_k}$. **Multi-query associative
recall (MQAR)** is exactly this read-many-bindings test (Arora et al. 2024,
arXiv:2312.04927); it is where linear attention falls behind selective SSMs and
softmax attention, whose exponential sharpening (or larger structured state)
sidesteps the bound.

This is a **fixed-weight mechanism demonstration, not a trained-model
reproduction.** No training: pairs are stored at fixed feature maps and read back
analytically. The metric is the per-binding retrieval error
$\frac1K\sum_j\lVert\hat v_j - v_j\rVert$ (the simplest feature map $\phi=\mathrm
{id}$ — raw-dot-product linear attention — so the rank story is exact). Trained
MQAR accuracy curves are cited to Zoology; this module shows *why* they bend:
capacity, not optimization.

Idiomatic-JAX / port credit
---------------------------
Greenfield. Synthetic-recall semantics mirror ``zoology``'s MQAR generator
(reference only; the training harness is **not** ported). Pure ``jnp`` linear
algebra (a Gram matmul and a softmax); no scan.

Usage
-----
::

    PYTHONPATH=. python companions/ch11/jax/mqar_recall.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "random_unit_keys",
    "orthonormal_keys",
    "orthonormal_values",
    "linear_retrieval_error",
    "softmax_retrieval_error",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch11"


def random_unit_keys(n_pairs: int, dim: int, seed: int = 0) -> jnp.ndarray:
    """``n_pairs`` random unit-norm keys in R^dim (generic, non-orthogonal)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    keys = rng.standard_normal((n_pairs, dim))
    return jnp.asarray(keys / np.linalg.norm(keys, axis=1, keepdims=True))


def orthonormal_keys(n_pairs: int, dim: int, seed: int = 0) -> jnp.ndarray:
    """``n_pairs`` orthonormal keys in R^dim (requires n_pairs <= dim) via QR."""
    import numpy as np

    if n_pairs > dim:
        raise ValueError(f"orthonormal keys need n_pairs <= dim; got {n_pairs} > {dim}")
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((dim, n_pairs)))
    return jnp.asarray(q.T)  # (n_pairs, dim), rows orthonormal


def orthonormal_values(n_pairs: int, dim_v: int, seed: int = 1) -> jnp.ndarray:
    """``n_pairs`` orthonormal value vectors in R^dim_v (requires n_pairs <= dim_v)."""
    import numpy as np

    if n_pairs > dim_v:
        raise ValueError(f"orthonormal values need n_pairs <= dim_v; got {n_pairs} > {dim_v}")
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((dim_v, n_pairs)))
    return jnp.asarray(q.T)


def linear_retrieval_error(keys: jnp.ndarray, values: jnp.ndarray) -> float:
    r"""Mean per-binding error $\frac1K\sum_j\lVert\hat v_j - v_j\rVert$, $\phi=\mathrm{id}$.

    $\hat v_j = \sum_i\langle k_j,k_i\rangle v_i$ is the unnormalized linear-
    attention readout (RetNet/GLA use no $z$-normalizer). Zero iff the keys are
    orthonormal (perfect recall below capacity); grows like $\sqrt{K/d_k}$ past it.
    """
    if keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != values.shape[0]:
        raise ValueError(f"keys (K, d) and values (K, d_v) must share K; got {keys.shape}, {values.shape}")
    gram = keys @ keys.T  # (K, K) = feature Gram with phi = id
    y_hat = gram @ values  # (K, d_v)
    return float(jnp.mean(jnp.linalg.norm(y_hat - values, axis=1)))


def softmax_retrieval_error(keys: jnp.ndarray, values: jnp.ndarray, beta: float = 16.0) -> float:
    r"""Same retrieval error for *softmax* attention $\hat v_j = \sum_i w_{ji} v_i$.

    Weights $w_{ji} = \mathrm{softmax}(\beta\,\langle k_j,k_i\rangle)_i$. Exponential
    sharpening concentrates the weight on the self-match (the strict-max self
    dot-product of distinct unit keys), so the error stays near zero *independent
    of $K$* — the capacity-unbounded oracle standing in for softmax / a large
    selective SSM.
    """
    if keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != values.shape[0]:
        raise ValueError(f"keys (K, d) and values (K, d_v) must share K; got {keys.shape}, {values.shape}")
    weights = jax.nn.softmax(beta * (keys @ keys.T), axis=1)  # (K, K)
    y_hat = weights @ values
    return float(jnp.mean(jnp.linalg.norm(y_hat - values, axis=1)))


# ---------------------------------------------------------------------------
# Figure: retrieval error vs number of KV pairs, two capacities + the oracle
# ---------------------------------------------------------------------------

_KS = (2, 4, 8, 16, 24, 32, 48, 64, 96, 128)


def _mean_linear_error(n_pairs: int, dim: int, dim_v: int = 160, n_seeds: int = 4) -> float:
    import numpy as np

    errs = [linear_retrieval_error(random_unit_keys(n_pairs, dim, s), orthonormal_values(n_pairs, dim_v, 100 + s))
            for s in range(n_seeds)]
    return float(np.mean(errs))


def _mean_softmax_error(n_pairs: int, dim: int, dim_v: int = 160, n_seeds: int = 4) -> float:
    import numpy as np

    errs = [softmax_retrieval_error(random_unit_keys(n_pairs, dim, s), orthonormal_values(n_pairs, dim_v, 100 + s))
            for s in range(n_seeds)]
    return float(np.mean(errs))


def make_mqar_figure() -> Figure:
    """Retrieval error vs K: linear attention at two state sizes d_k + the softmax oracle."""
    import numpy as np

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    ks = np.asarray(_KS)
    lin_small = np.asarray([_mean_linear_error(int(k), 16) for k in ks])
    lin_big = np.asarray([_mean_linear_error(int(k), 64) for k in ks])
    oracle = np.asarray([_mean_softmax_error(int(k), 64) for k in ks])

    fig, ax = create_tufte_figure(figsize=(6.6, 4.3))
    ax.plot(ks, lin_small, "o-", color=SSM_COLORS["accent"], label=r"linear attn, $d_k=16$")
    ax.plot(ks, lin_big, "s-", color=SSM_COLORS["highlight"], label=r"linear attn, $d_k=64$")
    ax.plot(ks, oracle, "^-", color=SSM_COLORS["baseline"], label="softmax (exact)")
    ax.axvline(16, color=SSM_COLORS["alert"], lw=0.8, ls=":", label=r"$d_k=16$ capacity")
    set_tufte_title(ax, "Retrieval error grows once $K$ outruns the state")
    set_tufte_labels(ax, xlabel=r"number of key-value pairs $K$", ylabel=r"mean $\|\hat v_j - v_j\|$")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def main() -> None:
    import matplotlib.pyplot as plt

    from companions._shared.plot_utils import save_figure

    print("Chapter 11 — mqar_recall.py")
    print("=" * 64)

    # Below capacity with ORTHONORMAL keys (K <= d_k): exact recall, zero error.
    print("  orthonormal keys, K <= d_k=32: exact recall")
    for n_pairs in (4, 16, 32):
        err = linear_retrieval_error(orthonormal_keys(n_pairs, 32, 0), orthonormal_values(n_pairs, 64, 1))
        print(f"    K={n_pairs:3d}: linear retrieval error = {err:.2e}  (orthonormal -> ~0)")

    # Generic keys: error grows with K, shrinks with d_k; softmax stays ~0.
    print("  random keys: error ~ sqrt(K/d_k); softmax oracle ~ 0")
    for n_pairs in (16, 64, 128):
        e16 = _mean_linear_error(n_pairs, 16)
        e64 = _mean_linear_error(n_pairs, 64)
        es = _mean_softmax_error(n_pairs, 64)
        print(f"    K={n_pairs:3d}: linear d_k=16 -> {e16:.3f}   d_k=64 -> {e64:.3f}   softmax -> {es:.3f}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_mqar_figure()
    for p in save_figure(fig, _OUT_DIR / "mqar-capacity", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
