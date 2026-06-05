r"""Chapter 11 §11.1-11.2 — linear attention as a matrix-state linear recurrence.

Replacing the softmax score $\exp(q^\top k)$ with a *separable* score
$\phi(q)^\top\phi(k)$ (Katharopoulos et al. 2020, arXiv:2006.16236) turns
attention into a linear recurrence on a **matrix state**. With queries/keys
$q_t, k_t \in \mathbb{R}^{d}$, values $v_t \in \mathbb{R}^{d_v}$, and a feature
map $\phi:\mathbb{R}^{d}\to\mathbb{R}^{d_k}$,

.. math::

    S_t = S_{t-1} + \phi(k_t)\,v_t^\top \in \mathbb{R}^{d_k\times d_v},
    \qquad z_t = z_{t-1} + \phi(k_t),
    \qquad y_t = \frac{S_t^\top\phi(q_t)}{z_t^\top\phi(q_t)}.

This is a genuine state-space recurrence with transition $A = I$ (pure
accumulation, no forgetting — the LTI face; Chapter 8). It is the dual of
Chapter 9's scan with a *matrix* carry, and it is exactly the §9.6 duality
(`ch09:ssd-duality`) read from the attention side: the recurrent form equals the
masked-parallel form $Y = (L \circ (Q_\phi K_\phi^\top))\,V$ with $L$ the
*all-ones* causal mask (vs Chapter 9's decay mask; gating in §11.3 puts the decay
back). This module pins three load-bearing facts:

* **§11.2 recurrent == parallel** to ``< 1e-12`` (float64) — Theorem
  ``ch11:recurrent-parallel-equivalence``;
* **capacity** — the state $S = \sum_i \phi(k_i)v_i^\top$ is a sum of $K$
  rank-one terms, so $\operatorname{rank} S \le \min(K, d_k)$ (Proposition
  ``ch11:linattn-capacity``, the mechanism behind the §11.6 MQAR limit);
* **§11.2 normalizer precision** — the normalized output is a large/large
  quotient whose recurrent (cumsum) and parallel (matmul) summation orders agree
  to ``< 1e-12`` in float64 but cap out near ``1e-7`` (one part in $10^7$, ~1 ulp)
  in float32. That ``~1e9`` precision gap is why the companions are float64: the
  ``1e-12`` identity pin is simply unreachable in single precision.

Idiomatic-JAX note (NumPy->JAX teaching point)
----------------------------------------------
The recurrent form is a genuine sequential carry, so it uses ``lax.scan`` (the
same primitive as Chapters 8-10), not a Python ``for`` loop a NumPy reference
would write; the parallel form is one batched ``einsum``. The two share no code,
which is the point — they are independent computations of the same operator, so
their agreement is a real correctness certificate, not a tautology.

Port credit
-----------
Greenfield: no predecessor JAX implementation exists for the linear-attention
recurrence (``post_transformers/experiments/jax/week11`` ships only the Hyena
FFTConv primitive; see ``fftconv.py``). The matrix-state form is authored from
the paper math (Katharopoulos et al., arXiv:2006.16236) and is the direct dual
of ``companions/ch09/jax/ssd_semiseparable.py``.

Usage
-----
::

    PYTHONPATH=. python companions/ch11/jax/linear_attention.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import jax

# Enable float64 before any jnp array is created (matches Chapters 4, 7-10).
# The normalizer is a large/large quotient; float32 masks the recurrent/parallel
# agreement (see test_float64_vs_float32_normalizer).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "feature_map_elu",
    "feature_map_relu",
    "resolve_phi",
    "linear_attention_recurrent",
    "linear_attention_parallel",
    "linear_attention_state",
    "recurrent_parallel_residual",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT_DIR = _REPO_ROOT / "public" / "figures" / "ch11"


# ---------------------------------------------------------------------------
# §11.1 — feature maps (the separable score phi(q).phi(k))
# ---------------------------------------------------------------------------


def feature_map_elu(x: jnp.ndarray) -> jnp.ndarray:
    r"""Katharopoulos feature map $\phi(x) = \mathrm{elu}(x) + 1$ (strictly positive).

    Strict positivity ($\phi(x) > 0$ elementwise) keeps the normalizer
    $z_t^\top\phi(q_t)$ bounded away from zero, so the softmax-free attention is
    still a convex combination of values. This is the default $\phi$ of
    Katharopoulos et al. (2020).
    """
    return jax.nn.elu(x) + 1.0


def feature_map_relu(x: jnp.ndarray) -> jnp.ndarray:
    r"""ReLU feature map $\phi(x) = \max(x, 0)$ (non-negative).

    Cheaper than :func:`feature_map_elu` but only *non-negative*, so the
    normalizer can vanish if every key feature is zero in some coordinate; used
    here to show the recurrent/parallel equivalence holds for *any* $\phi$.
    """
    return jax.nn.relu(x)


_PHI: dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "elu": feature_map_elu,
    "relu": feature_map_relu,
}


def resolve_phi(feature_map: str | Callable[[jnp.ndarray], jnp.ndarray]):
    if callable(feature_map):
        return feature_map
    if feature_map not in _PHI:
        raise ValueError(f"unknown feature_map {feature_map!r}; expected one of {sorted(_PHI)}")
    return _PHI[feature_map]


def _check_qkv(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray) -> None:
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError(f"q, k, v must each be 2D (L, d); got {q.shape}, {k.shape}, {v.shape}")
    if q.shape != k.shape:
        raise ValueError(f"q and k must share shape (L, d); got {q.shape} and {k.shape}")
    if v.shape[0] != q.shape[0]:
        raise ValueError(f"v must have the same length L as q/k; got {v.shape[0]} vs {q.shape[0]}")


def _require_positive_features(feature_map: str | Callable, normalize: bool) -> None:
    """Guard the normalized path: a non-strictly-positive map (relu) can zero the
    normalizer $z_t^\\top\\phi(q_t)$ and return a silent NaN. Fail loudly instead."""
    if normalize and feature_map == "relu":
        raise ValueError(
            "normalize=True requires a strictly positive feature map; feature_map='relu' "
            "can zero the normalizer z·φ(q) and return NaN. Use normalize=False or feature_map='elu'."
        )


# ---------------------------------------------------------------------------
# §11.2 — the two faces: recurrent (matrix-state scan) and parallel (masked matmul)
# ---------------------------------------------------------------------------


def linear_attention_recurrent(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    feature_map: str | Callable[[jnp.ndarray], jnp.ndarray] = "elu",
    normalize: bool = True,
) -> jnp.ndarray:
    r"""Recurrent linear attention via ``lax.scan`` on the matrix state $S_t$.

    Carries $(S_t, z_t)$ with $S_t = S_{t-1} + \phi(k_t)v_t^\top$ and
    $z_t = z_{t-1} + \phi(k_t)$, emitting $y_t = S_t^\top\phi(q_t)$ (unnormalized)
    or $y_t = S_t^\top\phi(q_t) / (z_t^\top\phi(q_t))$ (normalized). $O(L\,d_k\,d_v)$
    time, $O(d_k\,d_v)$ memory — the streaming/inference mode.

    Parameters
    ----------
    q, k : jnp.ndarray, shape (L, d)
        Queries and keys (pre-feature-map).
    v : jnp.ndarray, shape (L, d_v)
        Values.
    feature_map : {"elu", "relu"} or callable, default "elu"
        The kernel feature map $\phi$.
    normalize : bool, default True
        Divide by $z_t^\top\phi(q_t)$ (true linear attention) or return the raw
        matrix-state readout (used by §11.6 capacity demos).

    Returns
    -------
    y : jnp.ndarray, shape (L, d_v)
    """
    _check_qkv(q, k, v)
    _require_positive_features(feature_map, normalize)
    phi = resolve_phi(feature_map)
    qf, kf = phi(q), phi(k)  # (L, d_k)
    d_k, d_v = qf.shape[1], v.shape[1]

    def step(carry, inp):
        s, z = carry  # s: (d_k, d_v), z: (d_k,)
        kf_t, v_t, qf_t = inp
        s = s + jnp.outer(kf_t, v_t)
        z = z + kf_t
        num = s.T @ qf_t  # (d_v,)
        if normalize:
            return (s, z), num / (z @ qf_t)
        return (s, z), num

    # Carry dtypes must match the (possibly promoted) update dtypes so a mixed-precision
    # (e.g. float32 v, float64 q/k) call does not trip lax.scan's carry-invariance check.
    s0 = jnp.zeros((d_k, d_v), dtype=jnp.result_type(kf.dtype, v.dtype))
    z0 = jnp.zeros((d_k,), dtype=kf.dtype)
    _, ys = jax.lax.scan(step, (s0, z0), (kf, v, qf))
    return ys


def linear_attention_parallel(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    feature_map: str | Callable[[jnp.ndarray], jnp.ndarray] = "elu",
    normalize: bool = True,
) -> jnp.ndarray:
    r"""Parallel linear attention as a masked matmul $Y = (L \circ (Q_\phi K_\phi^\top))V$.

    $L$ is the **all-ones causal mask** (1 on and below the diagonal). This is the
    `ch09:ssd-duality` form with the decay mask specialized to ones — the
    quadratic / "attention" face. $O(L^2 d)$ time. The normalizer is the masked
    row-sum $\sum_{j\le t}\phi(q_t)^\top\phi(k_j)$.

    Same parameters/returns as :func:`linear_attention_recurrent`; equals it to
    machine precision in float64 (Theorem ``ch11:recurrent-parallel-equivalence``).
    """
    _check_qkv(q, k, v)
    _require_positive_features(feature_map, normalize)
    phi = resolve_phi(feature_map)
    qf, kf = phi(q), phi(k)
    length = qf.shape[0]
    scores = qf @ kf.T  # (L, L); scores[t, j] = phi(q_t).phi(k_j)
    causal = jnp.tril(jnp.ones((length, length), dtype=scores.dtype))  # 1 for j <= t
    masked = scores * causal
    num = masked @ v  # (L, d_v)
    if normalize:
        den = masked.sum(axis=1, keepdims=True)  # (L, 1)
        return num / den
    return num


def linear_attention_state(
    k: jnp.ndarray,
    v: jnp.ndarray,
    feature_map: str | Callable[[jnp.ndarray], jnp.ndarray] = "elu",
) -> jnp.ndarray:
    r"""The final matrix state $S = \sum_{i} \phi(k_i)\,v_i^\top = \Phi_K^\top V$.

    A sum of $K$ rank-one outer products, so $\operatorname{rank} S \le
    \min(K, d_k, d_v)$ — the capacity bound of Proposition
    ``ch11:linattn-capacity``. Pinned in ``test_linattn_state_rank``.

    Parameters
    ----------
    k : jnp.ndarray, shape (K, d)
        Keys (pre-feature-map); $K$ is the number of stored associations.
    v : jnp.ndarray, shape (K, d_v)
        Values.
    feature_map : default "elu"

    Returns
    -------
    S : jnp.ndarray, shape (d_k, d_v)
    """
    if k.ndim != 2 or v.ndim != 2 or k.shape[0] != v.shape[0]:
        raise ValueError(f"k (K, d) and v (K, d_v) must share K; got {k.shape} and {v.shape}")
    phi = resolve_phi(feature_map)
    kf = phi(k)  # (K, d_k)
    return kf.T @ v  # (d_k, d_v)


def recurrent_parallel_residual(
    length: int,
    d: int = 8,
    d_v: int = 6,
    dtype: jnp.dtype = jnp.float64,
    seed: int = 0,
    feature_map: str = "elu",
) -> float:
    r"""Max ``|recurrent - parallel|`` of the *normalized* output at a given dtype.

    The recurrent form accumulates by sequential cumsum; the parallel form by a
    single matmul. In exact arithmetic they are identical (Theorem
    ``ch11:recurrent-parallel-equivalence``); in finite precision the differing
    summation orders of the large/large normalizer quotient diverge. At
    ``length = 512`` this residual is ``< 1e-12`` in float64 but ``~1e-7`` in
    float32 — the §11.2 precision point.

    Parameters
    ----------
    length : int
        Sequence length $L$.
    d, d_v : int
        Query/key and value dimensions.
    dtype : jnp.dtype, default float64
    seed : int, default 0
    feature_map : default "elu"

    Returns
    -------
    residual : float
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((length, d)), dtype=dtype)
    k = jnp.asarray(rng.standard_normal((length, d)), dtype=dtype)
    v = jnp.asarray(rng.standard_normal((length, d_v)), dtype=dtype)
    y_rec = linear_attention_recurrent(q, k, v, feature_map=feature_map)
    y_par = linear_attention_parallel(q, k, v, feature_map=feature_map)
    return float(jnp.max(jnp.abs(y_rec - y_par)))


# ---------------------------------------------------------------------------
# Figure: recurrent==parallel residual + the O(L) vs O(L^2) cost crossover
# ---------------------------------------------------------------------------

_LENGTHS = (16, 32, 64, 128, 256, 512)


def make_recurrent_parallel_figure() -> Figure:
    """Left: recurrent==parallel residual vs L (machine zero). Right: O(L) vs O(L^2) cost."""
    import numpy as np

    from companions._shared.plot_utils import (
        SSM_COLORS,
        apply_style,
        create_tufte_figure,
        set_tufte_labels,
        set_tufte_title,
    )

    apply_style()
    lengths = np.asarray(_LENGTHS)
    resid = np.asarray([recurrent_parallel_residual(int(L)) for L in lengths])

    fig, (ax1, ax2) = create_tufte_figure(ncols=2, figsize=(11.0, 4.2))

    ax1.semilogy(lengths, np.maximum(resid, 1e-18), "o-", color=SSM_COLORS["accent"])
    ax1.axhline(1e-12, color=SSM_COLORS["alert"], lw=0.8, ls="--", label=r"$10^{-12}$ pin")
    set_tufte_title(ax1, "Recurrent $\\equiv$ parallel (float64)")
    set_tufte_labels(ax1, xlabel=r"sequence length $L$", ylabel=r"$\max|y_{\rm rec}-y_{\rm par}|$")
    ax1.legend(loc="upper left", fontsize=8, frameon=False)

    # Cost model (relative units): recurrent ~ L, parallel ~ L^2.
    ax2.loglog(lengths, lengths.astype(float), "o-", color=SSM_COLORS["accent"],
               label=r"recurrent $O(L)$")
    ax2.loglog(lengths, lengths.astype(float) ** 2, "s-", color=SSM_COLORS["highlight"],
               label=r"parallel $O(L^2)$")
    set_tufte_title(ax2, "Two faces, two costs")
    set_tufte_labels(ax2, xlabel=r"sequence length $L$", ylabel="relative cost")
    ax2.legend(loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    return fig


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from companions._shared.plot_utils import save_figure

    print("Chapter 11 — linear_attention.py")
    print("=" * 64)

    # §11.2 recurrent == parallel (float64).
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal((64, 8)))
    k = jnp.asarray(rng.standard_normal((64, 8)))
    v = jnp.asarray(rng.standard_normal((64, 6)))
    # Normalized path: elu keeps the normalizer strictly positive.
    y_rec = linear_attention_recurrent(q, k, v, feature_map="elu")
    y_par = linear_attention_parallel(q, k, v, feature_map="elu")
    print(f"  phi=elu  (normalized):   max diff = {float(jnp.max(jnp.abs(y_rec - y_par))):.2e}  (Thm rec-par: ~0)")
    # Unnormalized path: the core identity holds for any phi (relu can zero the normalizer -> NaN).
    for fm in ("elu", "relu"):
        y_rec = linear_attention_recurrent(q, k, v, feature_map=fm, normalize=False)
        y_par = linear_attention_parallel(q, k, v, feature_map=fm, normalize=False)
        print(f"  phi={fm:4s} (unnormalized): max diff = {float(jnp.max(jnp.abs(y_rec - y_par))):.2e}")

    # §11.6 capacity: rank S = min(K, d_k) (take d_v large so the value axis does not cap it).
    print("  capacity: rank of S = sum phi(k_i) v_i^T")
    for n_kv in (4, 16, 48):
        kk = jnp.asarray(rng.standard_normal((n_kv, 32)))  # d_k = 32 after elu
        vv = jnp.asarray(rng.standard_normal((n_kv, 64)))  # d_v = 64 > d_k, so cap is min(K, d_k)
        s = linear_attention_state(kk, vv)
        r = int(jnp.linalg.matrix_rank(s))
        print(f"    K={n_kv:2d}, d_k=32 -> rank S = {r}  (= min(K, d_k) = {min(n_kv, 32)})")

    # §11.2 normalizer precision: f64 vs f32 at L = 512.
    r64 = recurrent_parallel_residual(512, dtype=jnp.float64)
    r32 = recurrent_parallel_residual(512, dtype=jnp.float32)
    print(f"  normalizer @ L=512: float64 residual = {r64:.2e}  (< 1e-12)")
    print(f"                      float32 residual = {r32:.2e}  (the precision symptom)")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_recurrent_parallel_figure()
    for p in save_figure(fig, _OUT_DIR / "recurrent-parallel", formats=("png",)):
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
